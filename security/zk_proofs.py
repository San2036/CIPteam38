import hashlib
import json
import random

# Cryptographic Constants for True ZK-Proofs
# Using a 256-bit safe prime for demonstration speed 
P = 115792089237316195423570985008687907853269984665640564039457584007913129639747
G = 2

def generate_true_zk_proof(secret_weights: list) -> tuple:
    """
    Generates a true mathematical Non-Interactive Zero-Knowledge (NIZK) proof 
    using the Schnorr protocol + Fiat-Shamir heuristic.
    Proves knowledge of the model weights without revealing them!
    
    Returns: 
        public_y (str): The public key commitment to the weights.
        proof (bytes): The ZK proof payload.
    """
    # 1. Digest raw weights into an integer secret `x`
    weights_str = str(secret_weights)
    x_bytes = hashlib.sha256(weights_str.encode()).digest()
    x = int.from_bytes(x_bytes, byteorder='big')
    
    # 2. Public Key Y = G^x mod P
    Y = pow(G, x, P)
    
    # 3. Schnorr NIZK Protocol
    sys_rand = random.SystemRandom()
    r = sys_rand.randint(1, P-2)
    t = pow(G, r, P)
    
    # Fiat-Shamir challenge
    c_bytes = hashlib.sha256(f"{G}_{P}_{Y}_{t}".encode()).digest()
    c = int.from_bytes(c_bytes, byteorder='big')
    
    # Response
    s = (r + c * x) % (P - 1)
    
    proof_payload = json.dumps({"t": str(t), "s": str(s)})
    return str(Y), proof_payload.encode('utf-8')

def verify_true_zk_proof(proof: bytes, public_y: str) -> bool:
    """
    Mathematically verifies the NIZK proof using the public key commitment.
    """
    try:
        Y = int(public_y)
        proof_obj = json.loads(proof.decode('utf-8'))
        t = int(proof_obj["t"])
        s = int(proof_obj["s"])
        
        # Reconstruct challenge
        c_bytes = hashlib.sha256(f"{G}_{P}_{Y}_{t}".encode()).digest()
        c = int.from_bytes(c_bytes, byteorder='big')
        
        # Verify: G^s mod P == (t * Y^c) mod P
        lhs = pow(G, s, P)
        rhs = (t * pow(Y, c, P)) % P
        
        return lhs == rhs
    except Exception:
        return False
        
# Legacy wrappers to prevent import breaks elsewhere while refactoring
def generate_proof(model_hash: str) -> bytes:
    return b"LEGACY_REMOVED"
def verify_proof(proof: bytes, public_inputs: list) -> bool:
    return False
