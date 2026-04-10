
import json
import os
from web3 import Web3

# Configuration
GANACHE_URL = "http://127.0.0.1:7545"
CONTRACT_BUILD_PATH = os.path.join(os.path.dirname(__file__), '..', 'build', 'contracts', 'FLRegistry.json')

def deploy_contract():
    # 1. Connect to Blockchain
    w3 = Web3(Web3.HTTPProvider(GANACHE_URL))
    if not w3.is_connected():
        print(f"❌ Failed to connect to {GANACHE_URL}")
        return

    print(f"✅ Connected to Ganache. Network ID: {w3.net.version}")
    
    # 2. Load Contract Artifacts
    if not os.path.exists(CONTRACT_BUILD_PATH):
        print(f"❌ Contract JSON not found at: {CONTRACT_BUILD_PATH}")
        return

    with open(CONTRACT_BUILD_PATH, 'r') as f:
        contract_json = json.load(f)

    abi = contract_json['abi']
    bytecode = contract_json['bytecode']

    # 3. Get Account (Deployer)
    deployer_account = w3.eth.accounts[0]
    print(f"🚀 Deploying from account: {deployer_account}")

    # 4. Deploy Contract
    Contract = w3.eth.contract(abi=abi, bytecode=bytecode)
    
    # Build transaction
    tx_hash = Contract.constructor().transact({'from': deployer_account})
    print(f"⏳ Transaction Hash: {tx_hash.hex()}")
    
    # Wait for receipt
    tx_receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    contract_address = tx_receipt.contractAddress
    print(f"🎉 Contract Deployed to: {contract_address}")

    # 5. Update JSON Artifact with new address
    network_id = str(w3.net.version)
    
    if 'networks' not in contract_json:
        contract_json['networks'] = {}
        
    contract_json['networks'][network_id] = {
        "events": {},
        "links": {},
        "address": contract_address,
        "transactionHash": tx_hash.hex()
    }
    
    with open(CONTRACT_BUILD_PATH, 'w') as f:
        json.dump(contract_json, f, indent=2)
        
    print(f"📝 Updated artifacts at {CONTRACT_BUILD_PATH}")

if __name__ == "__main__":
    deploy_contract()
