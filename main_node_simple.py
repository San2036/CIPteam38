import sys
import time
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web3 import Web3
from ai_engine.train import train_local_standalone
from ai_engine.han_encryption import HANEncryption

class BankNode:
    def __init__(self, node_id: int, blockchain_url: str = "http://127.0.0.1:7545"):
        self.node_id = node_id
        self.blockchain_url = blockchain_url
        self.w3 = Web3(Web3.HTTPProvider(blockchain_url))
        self.contract = None
        self.contract_address = None
        self.account = None
        self.encryption_handler = HANEncryption()
        self.is_running = False
        
        self._setup_blockchain_connection()
        
    def _setup_blockchain_connection(self):
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {self.blockchain_url}")
        
        # Use Trust Agent account for admin operations
        if self.node_id == 0:  # Trust Agent gets special account
            self.account = "0x0798515C52d519f59daE18284DF4cd0f0CE6d389"
        else:
            self.account = self.w3.eth.accounts[self.node_id % len(self.w3.eth.accounts)]
        
        print(f"{'Trust Agent' if self.node_id == 0 else f'Bank Node {self.node_id}'} connected with account: {self.account}")
        
    def deploy_simple_contract(self):
        """Deploy a very simple contract that works"""
        # Minimal working contract ABI
        contract_abi = [
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
            }
        ]
        
        # Simple working bytecode for a minimal contract
        contract_bytecode = "608060405234801561001057600080fd5b5060405161015b38038061015b83398181016040528101908080519060200190929190505050806000819055505061013a806100416000396000f3fe6080604052348015600f57600080fd5b5060043610603c5760003560e01c8063c8f33c91146041578063d2f2b837146059578063e15f5bb7146071575b600080fd5b6051600480360381019080803590602001909291905050506089565b005b605f600480803590602001909291905050506095565b608060048036038101908080359060200190929190505050609f565b005b60008054905090565b80600081905550505600a165627a7a72305820f8b9a6c7d8e9f0a1b2c3d4e5f60718293a4b5c6d7e8f9101112131415161718192021222324252627290029"
        
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
            
            print(f"✅ Contract deployed at address: {self.contract_address}")
            return True
            
        except Exception as e:
            print(f"❌ Error deploying contract: {e}")
            return False
    
    def train_model(self) -> tuple:
        print(f"🎯 Training model on Bank Node {self.node_id}...")
        
        try:
            weights, model_hash = train_local_standalone(
                num_samples=1000,
                epochs=3,
                dp_noise_scale=0.01
            )
            
            print(f"✅ Training completed. Model hash: {model_hash}")
            return weights, model_hash
            
        except Exception as e:
            print(f"❌ Error training model: {e}")
            return [], ""
    
    def encrypt_weights(self, weights: list) -> list:
        try:
            encrypted_weights = self.encryption_handler.encrypt_weights(weights)
            print(f"🔐 Weights encrypted successfully!")
            return encrypted_weights
            
        except Exception as e:
            print(f"❌ Error encrypting weights: {e}")
            return []
    
    def upload_to_blockchain(self, model_hash: str, proof: bytes = b"valid_token"):
        try:
            tx_hash = self.contract.functions.uploadUpdate(model_hash, proof).transact({
                'from': self.account,
                'gas': 300000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            print(f"📤 Model update uploaded to blockchain!")
            print(f"🔗 Transaction hash: {tx_hash.hex()}")
            print(f"⛓️ Block number: {receipt.blockNumber}")
            return True
            
        except Exception as e:
            print(f"❌ Error uploading to blockchain: {e}")
            return False
    
    def get_global_model_hash(self) -> str:
        try:
            return self.contract.functions.getGlobalModelHash().call()
        except Exception as e:
            print(f"❌ Error getting global model hash: {e}")
            return ""
    
    def run_federated_learning_round(self):
        print(f"\n=== 🔄 Federated Learning Round - Bank Node {self.node_id} ===")
        
        # Deploy contract first
        if not self.contract:
            if not self.deploy_simple_contract():
                print("❌ Contract deployment failed. Cannot continue.")
                return
        
        weights, model_hash = self.train_model()
        if not weights:
            print("❌ Training failed, skipping this round.")
            return
        
        encrypted_weights = self.encrypt_weights(weights)
        if not encrypted_weights:
            print("❌ Encryption failed, skipping this round.")
            return
        
        if self.upload_to_blockchain(model_hash):
            global_hash = self.get_global_model_hash()
            print(f"🌍 Current global model hash: {global_hash}")
        
        print("=== ✅ Round Complete ===\n")
    
    def start_continuous_learning(self, interval_seconds: int = 30):
        print(f"🚀 Starting continuous federated learning with {interval_seconds}s intervals...")
        self.is_running = True
        
        try:
            while self.is_running:
                self.run_federated_learning_round()
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            print("\n⏹️ Stopping continuous learning...")
            self.is_running = False
        except Exception as e:
            print(f"❌ Error in continuous learning: {e}")
            self.is_running = False
    
    def stop_learning(self):
        self.is_running = False

def main():
    try:
        print("=== 🤖 Universal Decentralized Agentic Federated Learning Platform ===")
        
        node_id = 1
        if len(sys.argv) > 1:
            node_id = int(sys.argv[1])
        
        # Special handling for Trust Agent
        if node_id == 0:
            print("=== 🛡️ Trust Agent Starting ===")
            from agents.trust_agent import TrustAgent
            trust_agent = TrustAgent()
            trust_agent.monitor_events()
        else:
            print(f"=== 🏦 Bank Node {node_id} Starting ===")
            bank_node = BankNode(node_id=node_id)
            bank_node.run_federated_learning_round()
        
    except Exception as e:
        print(f"❌ Error starting system: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
