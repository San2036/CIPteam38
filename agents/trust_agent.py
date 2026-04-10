import time
import json
from web3 import Web3
from typing import Dict, List, Optional
import logging
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from security.zk_proofs import verify_proof

class TrustAgent:
    def __init__(self, blockchain_url: str = "http://127.0.0.1:7545", contract_address: Optional[str] = None):
        self.blockchain_url = blockchain_url
        self.w3 = Web3(Web3.HTTPProvider(blockchain_url))
        self.contract_address = contract_address
        self.contract = None
        self.account = None
        self.is_running = False
        
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        self._setup_connection()
        
    def _setup_connection(self):
        if not self.w3.is_connected():
            raise ConnectionError(f"Failed to connect to blockchain at {self.blockchain_url}")
        
        self.account = self.w3.eth.accounts[0]
        self.logger.info(f"Trust Agent connected with account: {self.account}")
        
        if self.contract_address:
            self._load_contract()
    
    def _load_contract(self):
        contract_abi = [
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
                    {"internalType": "address", "name": "badNode", "type": "address"}
                ],
                "name": "slashNode",
                "outputs": [],
                "stateMutability": "nonpayable",
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
            }
        ]
        
        self.contract = self.w3.eth.contract(
            address=self.contract_address,
            abi=contract_abi
        )
        
        self.logger.info(f"Contract loaded at address: {self.contract_address}")
    
    def set_contract_address(self, contract_address: str):
        self.contract_address = contract_address
        self._load_contract()
    
    def validate_proof(self, proof: bytes, model_hash: str) -> bool:
        try:
            # Verify the ZK-Proof against the public input (model_hash)
            return verify_proof(proof, [model_hash])
        except Exception as e:
            self.logger.error(f"Error validating proof: {e}")
            return False
    
    def monitor_events(self):
        if not self.contract:
            self.logger.error("Contract not loaded. Cannot monitor events.")
            self.logger.info("Waiting for contract deployment...")
            
            # Wait for contract to be deployed by a bank node
            import time
            max_wait = 60  # Wait up to 60 seconds
            wait_time = 0
            
            while wait_time < max_wait:
                try:
                    # Try to find deployed contract by checking recent blocks
                    latest_block = self.w3.eth.get_block('latest', full_transactions=True)
                    
                    for tx in latest_block.transactions:
                        if hasattr(tx, 'contractAddress') and tx.contractAddress:
                            self.logger.info(f"Found deployed contract at: {tx.contractAddress}")
                            self.set_contract_address(tx.contractAddress)
                            break
                    
                    if self.contract:
                        break
                        
                    time.sleep(2)
                    wait_time += 2
                    self.logger.info(f"Waiting for contract deployment... ({wait_time}s)")
                    
                except Exception as e:
                    self.logger.error(f"Error checking for contract: {e}")
                    time.sleep(2)
                    wait_time += 2
            
            if not self.contract:
                self.logger.error("Contract deployment timeout. Please start a bank node first.")
                return
        
        self.logger.info("Trust Agent is actively monitoring for ModelUpdated events...")
        
        try:
            event_filter = self.contract.events.ModelUpdated.create_filter(fromBlock='latest')
            
            while True:
                for event in event_filter.get_new_entries():
                    self._handle_model_updated_event(event)
                
                time.sleep(2)
                
        except Exception as e:
            self.logger.error(f"Error monitoring events: {e}")
            self.is_running = False
    
    def _handle_model_updated_event(self, event):
        try:
            node_address = event['args']['nodeAddress']
            new_hash = event['args']['newHash']
            proof = event['args']['proof']
            timestamp = event['args']['timestamp']
            
            self.logger.info(f"Model update received from {node_address}")
            self.logger.info(f"Model hash: {new_hash}")
            self.logger.info(f"Timestamp: {timestamp}")
            
            self.logger.info(f"Timestamp: {timestamp}")
            
            is_valid = self.validate_proof(proof, new_hash)
            
            if is_valid:
                self.logger.info(f"Proof validated for node {node_address}")
            else:
                self.logger.warning(f"Invalid proof from node {node_address}. Slashing node...")
                self.slash_node(node_address)
                
        except Exception as e:
            self.logger.error(f"Error handling model update: {e}")
    
    def slash_node(self, node_address: str):
        try:
            if not self.contract:
                self.logger.error("Contract not loaded. Cannot slash node.")
                return
            
            tx_hash = self.contract.functions.slashNode(node_address).transact({
                'from': self.account,
                'gas': 200000
            })
            
            receipt = self.w3.eth.wait_for_transaction_receipt(tx_hash)
            
            self.logger.info(f"Node {node_address} slashed successfully!")
            self.logger.info(f"Transaction hash: {tx_hash.hex()}")
            self.logger.info(f"Block number: {receipt.blockNumber}")
            
        except Exception as e:
            self.logger.error(f"Error slashing node {node_address}: {e}")
    
    def get_node_reputation(self, node_address: str) -> int:
        try:
            if not self.contract:
                return -1
            
            node_info = self.contract.functions.getNodeInfo(node_address).call()
            return node_info[1]  # reputation is the second field
            
        except Exception as e:
            self.logger.error(f"Error getting node reputation: {e}")
            return -1
    
    def stop_monitoring(self):
        self.is_running = False
        self.logger.info("Stopping event monitoring...")
    
    def get_account_balance(self) -> float:
        try:
            balance_wei = self.w3.eth.get_balance(self.account)
            return self.w3.from_wei(balance_wei, 'ether')
        except Exception as e:
            self.logger.error(f"Error getting account balance: {e}")
            return 0.0

if __name__ == "__main__":
    try:
        trust_agent = TrustAgent()
        
        print("Trust Agent initialized successfully!")
        print(f"Account: {trust_agent.account}")
        print(f"Balance: {trust_agent.get_account_balance()} ETH")
        
        print(f"Balance: {trust_agent.get_account_balance()} ETH")
        
        print("Starting event monitoring...")
        trust_agent.monitor_events()
            
    except Exception as e:
        print(f"Error starting Trust Agent: {e}")
