from web3 import Web3
import sys

def check_connection():
    url = "http://127.0.0.1:7545"
    print(f"Attempting to connect to {url}...")
    
    try:
        w3 = Web3(Web3.HTTPProvider(url))
        
        if w3.is_connected():
            print("✅ CONNECTION SUCCESSFUL!")
            print(f"Network ID: {w3.net.version}")
            print(f"Block Number: {w3.eth.block_number}")
            print(f"Accounts: {len(w3.eth.accounts)}")
            if len(w3.eth.accounts) > 0:
                print(f"First Account: {w3.eth.accounts[0]}")
        else:
            print("❌ CONNECTION FAILED: w3.is_connected() returned False")
            print("Possible causes:")
            print("1. Ganache is not running.")
            print("2. Ganache is running on a different port (check 8545 vs 7545).")
            print("3. Firewall is blocking the connection.")
            
    except Exception as e:
        print(f"❌ EXCEPTION OCCURRED: {e}")

if __name__ == "__main__":
    check_connection()
