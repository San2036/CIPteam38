
import sys
import time

print("=== System Diagnostic Check ===")
print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print(f"Sys Path: {sys.path}")

# 1. Check Imports
print("\n1. Checking Dependencies...")
try:
    import web3
    print("   [OK] web3")
except ImportError as e:
    print(f"   [FAIL] web3: {e}")

try:
    import pandas
    print("   [OK] pandas")
except ImportError as e:
    print(f"   [FAIL] pandas: {e}")

try:
    import torch
    print("   [OK] torch")
except ImportError as e:
    print(f"   [FAIL] torch: {e}")

try:
    import streamlit
    print("   [OK] streamlit")
except ImportError as e:
    print(f"   [FAIL] streamlit: {e}")

# 2. Check Blockchain Connection
print("\n2. Checking Blockchain Connection (Ganache)...")
try:
    from web3 import Web3
    w3 = Web3(Web3.HTTPProvider("http://127.0.0.1:7545"))
    if w3.is_connected():
        print(f"   [OK] Connected to Ganache at http://127.0.0.1:7545")
        print(f"   [INFO] Block Number: {w3.eth.block_number}")
        print(f"   [INFO] Available Accounts: {len(w3.eth.accounts)}")
    else:
        print("   [FAIL] Could not connect to Ganache at http://127.0.0.1:7545")
        print("   -> Is Ganache running?")
        print("   -> Is it listening on port 7545?")
except Exception as e:
    print(f"   [FAIL] Connection Error: {e}")

print("\n=== Diagnostic Complete ===")
input("Press Enter to close...")
