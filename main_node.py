#!/usr/bin/env python3

import sys
import os
import time
import json
import threading
from typing import Optional

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web3 import Web3
from ai_engine.train import train_local_standalone, load_and_preprocess_data
from ai_engine.model import SimpleNN, load_model_from_weights
from ai_engine.han_encryption import HANEncryption
from security.zk_proofs import generate_proof

class BankNode:
    def __init__(self, node_id: int, total_nodes: int, contract_address: str = None):
        self.node_id = node_id
        self.total_nodes = total_nodes
        self.w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
        self.account = self.w3.eth.accounts[node_id]  # Use different accounts for different nodes
        
        # Load data to determine input shape
        _, self.input_shape = load_and_preprocess_data(node_id, total_nodes)
        self.model = SimpleNN(input_size=self.input_shape, hidden_size=64, num_classes=2)
        self.encryption_handler = HANEncryption()
        self.is_running = False
        
        self._setup_blockchain_connection()
        self._deploy_contract()
        
    def _setup_blockchain_connection(self):
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {self.blockchain_url}")
        
        # Use Trust Agent account for admin operations
        if self.node_id == 0:  # Trust Agent gets special account
            self.account = "0x0798515C52d519f59daE18284DF4cd0f0CE6d389"
        else:
            self.account = self.w3.eth.accounts[self.node_id % len(self.w3.eth.accounts)]
        
        print(f"{'Trust Agent' if self.node_id == 0 else f'Bank Node {self.node_id}'} connected with account: {self.account}")
        
    def _get_contract_abi(self):
        return [
            {
                "inputs": [],
                "stateMutability": "nonpayable",
                "type": "constructor"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "address", "name": "nodeAddress", "type": "address"},
                    {"indexed": False, "internalType": "string", "name": "newHash", "type": "string"},
                    {"indexed": False, "internalType": "bytes", "name": "proof", "type": "bytes"},
                    {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                ],
                "name": "ModelUpdated",
                "type": "event"
            },
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "internalType": "address", "name": "badNode", "type": "address"},
                    {"indexed": False, "internalType": "uint256", "name": "timestamp", "type": "uint256"}
                ],
                "name": "NodeSlashed",
                "type": "event"
            },
            {
                "inputs": [
                    {"internalType": "uint256", "name": "nodeId", "type": "uint256"}
                ],
                "name": "registerNode",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "string", "name": "newHash", "type": "string"},
                    {"internalType": "bytes", "name": "proof", "type": "bytes"}
                ],
                "name": "uploadUpdate",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "getGlobalModelHash",
                "outputs": [
                    {"internalType": "string", "name": "", "type": "string"}
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "address", "name": "nodeAddr", "type": "address"}
                ],
                "name": "getNodeInfo",
                "outputs": [
                    {
                        "components": [
                            {"internalType": "uint256", "name": "id", "type": "uint256"},
                            {"internalType": "uint256", "name": "reputation", "type": "uint256"},
                            {"internalType": "bool", "name": "isBanned", "type": "bool"},
                            {"internalType": "address", "name": "nodeAddress", "type": "address"}
                        ],
                        "internalType": "struct FLRegistry.Node",
                        "name": "",
                        "type": "tuple"
                    }
                ],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [
                    {"internalType": "address", "name": "badNode", "type": "address"}
                ],
                "name": "slashNode",
                "outputs": [],
                "stateMutability": "nonpayable",
                "type": "function"
            }
        ]

    def _deploy_contract(self):
        contract_abi = self._get_contract_abi()
        
        # If not Node 1, wait for contract address
        if self.node_id != 1:
            self._wait_for_contract(contract_abi)
            return

        # Node 1 Deploys
        print(f"Node {self.node_id} deploying contract...")
        
        contract_bytecode = "608060405234801561001057600080fd5b5060016000556001600a60006101000a81548173ffffffffffffffffffffffffffffffffffffffff021916908373ffffffffffffffffffffffffffffffffffffffff1602179055506104b3806100626000396000f3fe608060405234801561001057600080fd5b50600436106100625760003560e01c8063a8b0577414610067578063b8b7a2f714610085578063c8f33c91146100a3578063d2f2b837146100c1578063e15f5bb7146100df578063f3b6f5fb146100fd575b600080fd5b61006f61011b565b60405161007c91906102a6565b60405180910390f35b61008d6101a1565b60405161009a91906102a6565b60405180910390f35b6100ab610227565b6040516100b891906102a6565b60405180910390f35b6100c96102ad565b6040516100d691906102a6565b60405180910390f35b6100e7610333565b6040516100f491906102a6565b60405180910390f35b610117600480360381019061011291906102f2565b6103a9565b005b60606000805461012890610351565b80601f016020809104026020016040519081016040528092919081815260200182805461015490610351565b80156101a15780601f10610176576101008083540402835291602001916101a1565b820191906000526020600020905b81548152906001019060200180831161018457829003601f168201915b5050505050905090565b60008054905090565b60008054905060005b81811015610221576000600160008585858181106101db576101da610383565b5b90506020020135815260200190815260200160002054111561020e576001600084848481811061020d5761020c610383565b5b5b508061021a906103b2565b90506101b4565b508091505090565b600160009054906101000a900473ffffffffffffffffffffffffffffffffffffffff1681565b60008054905060005b8181101561032e5760006001600085858581811061027257610271610383565b5b90506020020135815260200190815260200160002054141561031b57600160008484848181106102a4576102a3610383565b5b90506020020135815260200190815260200160002060008282546102c791906103ec565b9250508190555080806102d9906103b2565b915050610254565b508091505090565b60008054905090565b60008054905060005b818110156103a45760006001600085858581811061031857610317610383565b5b905060200201358152602001908152602001600020541415610391576001600084848481811061034b5761034a610383565b5b905060200201358152602001908152602001600020600082825461036e91906103ec565b925050819055508080610380906103b2565b9150506102fa565b508091505090565b6000819050919050565b6000604051905090565b600080fd5b600080fd5b600080fd5b600080fd5b6000601f19601f8301169050919050565b7f4e487b7100000000000000000000000000000000000000000000000000000000600052604160045260246000fd5b7f4e487b7100000000000000000000000000000000000000000000000000000000600052602260045260246000fd5b60008060006060848603121561044757600080fd5b600083013567ffffffffffffffff81111561046157600080fd5b61046d8682870161040c565b9350935050602084013561048081610497565b929592945050506040919091013590565b6000815190506104a081610497565b92915050565b6000602082840312156104b857600080fd5b60006104c684828501610491565b9150509291505056fea26469706673582212208a5b5c5d5a5b5c5d5a5b5c5d5a5b5c5d5a5b5c5d5a5b5c5d5a5b5c5d64736f6c63430008110033"
        
        try:
            Contract = self.w3.eth.contract(abi=contract_abi, bytecode=contract_bytecode)
            tx_hash = Contract.constructor().transact({
                'from': self.account,
                'gas': 2000000
            })
            
            tx_receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            self.contract_address = tx_receipt.contractAddress
            self.contract = self.w3.eth.contract(
                address=self.contract_address,
                abi=contract_abi
            )
            
            print(f"Contract deployed at address: {self.contract_address}")
            
            # Save contract address for other nodes
            with open("contract_address.json", "w") as f:
                json.dump({"address": self.contract_address, "timestamp": time.time()}, f)
            print("Contract address saved to contract_address.json")
            
        except Exception as e:
            print(f"Error deploying contract: {e}")
            raise

    def _wait_for_contract(self, contract_abi):
        print(f"Node {self.node_id} waiting for contract deployment by Node 1...")
        
        address_file = "contract_address.json"
        
        while True:
            if os.path.exists(address_file):
                try:
                    with open(address_file, "r") as f:
                        data = json.load(f)
                        address = data.get("address")
                        
                    if address:
                        # Verify it has code (part of current chain state)
                        code = self.w3.eth.get_code(address)
                        if code and len(code) > 0:
                            self.contract_address = address
                            self.contract = self.w3.eth.contract(address=address, abi=contract_abi)
                            print(f"Node {self.node_id} connected to contract at {address}")
                            break
                        else:
                            print("Contract file found but code is empty (stale file?). Waiting...")
                            
                except Exception as e:
                    print(f"Error checking contract file: {e}")
            
            time.sleep(2)

    
    def register_node(self):
        try:
            tx_hash = self.contract.functions.registerNode(self.node_id).transact({
                'from': self.account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Node {self.node_id} registered successfully!")
            print(f"Transaction hash: {tx_hash.hex()}")
            
        except Exception as e:
            print(f"Error registering node: {e}")
    
    def train_model(self) -> tuple:
        print(f"Training model on Bank Node {self.node_id}...")
        
        try:
            weights, model_hash = train_local_standalone(
                num_samples=1000,
                epochs=3,
                dp_noise_scale=0.01,
                node_id=self.node_id
            )
            
            print(f"Training completed. Model hash: {model_hash}")
            return weights, model_hash
            
        except Exception as e:
            print(f"Error training model: {e}")
            return [], ""
    
    def encrypt_weights(self, weights: list) -> list:
        try:
            encrypted_weights = self.encryption_handler.encrypt_weights(weights)
            print(f"Weights encrypted successfully!")
            return encrypted_weights
            
        except Exception as e:
            print(f"Error encrypting weights: {e}")
            return []
    
    def upload_to_blockchain(self, model_hash: str):
        try:
            # Generate ZK-Proof
            proof = generate_proof(model_hash)
            
            tx_hash = self.contract.functions.uploadUpdate(model_hash, proof).transact({
                'from': self.account,
                'gas': 300000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"Model update uploaded to blockchain!")
            print(f"Transaction hash: {tx_hash.hex()}")
            print(f"Block number: {receipt.blockNumber}")
            
        except Exception as e:
            print(f"Error uploading to blockchain: {e}")
    
    def get_global_model_hash(self) -> str:
        try:
            return self.contract.functions.getGlobalModelHash().call()
        except Exception as e:
            print(f"Error getting global model hash: {e}")
            return ""
    
    def run_federated_learning_round(self):
        print(f"\n=== Federated Learning Round - Bank Node {self.node_id} ===")
        
        weights, model_hash = self.train_model()
        if not weights:
            print("Training failed, skipping this round.")
            return
        
        encrypted_weights = self.encrypt_weights(weights)
        if not encrypted_weights:
            print("Encryption failed, skipping this round.")
            return
        
        self.upload_to_blockchain(model_hash)
        
        global_hash = self.get_global_model_hash()
        print(f"Current global model hash: {global_hash}")
        
        print("=== Round Complete ===\n")
    
    def start_continuous_learning(self, interval_seconds: int = 30):
        print(f"Starting continuous federated learning with {interval_seconds}s intervals...")
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_federated_learning_round()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\nStopping continuous learning...")
            self.is_running = False
        except Exception as e:
            print(f"Error in continuous learning: {e}")
            self.is_running = False
    
    def stop_learning(self):
        self.is_running = False

def main():
    try:
        print("=== Universal Decentralized Agentic Federated Learning Platform ===")
        
        node_id = 1
        if len(sys.argv) > 1:
            node_id = int(sys.argv[1])
        
        # Special handling for Trust Agent
        if node_id == 0:
            print("=== Trust Agent Starting ===")
            from agents.trust_agent import TrustAgent
            trust_agent = TrustAgent()
            trust_agent.monitor_events()
        else:
            print(f"=== Bank Node {node_id} Starting ===")
            bank_node = BankNode(node_id=node_id, total_nodes=2)
            bank_node.register_node()
            bank_node.run_federated_learning_round()
        
    except Exception as e:
        print(f"Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
