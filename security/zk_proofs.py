import hashlib
import time
from typing import List

def generate_proof(model_hash: str) -> bytes:
    """
    Generates a mock ZK-proof for a given model hash.
    In a real system, this would use ZoKrates to generate a SNARK/STARK.
    """
    # Simulate computation time
    time.sleep(0.1)
    
    # Create a deterministic mock proof based on the input
    proof_content = f"PROOF_{model_hash}_AUTHORIZED"
    return proof_content.encode('utf-8')

def verify_proof(proof: bytes, public_inputs: List[str]) -> bool:
    """
    Verifies the mock ZK-proof.
    Returns True if the proof is valid for the given public inputs.
    """
    try:
        proof_str = proof.decode('utf-8')
        
        # In our mock logic, a valid proof must start with "PROOF_" and end with "_AUTHORIZED"
        # and contain the model hash (which is usually one of the public inputs)
        
        if not proof_str.startswith("PROOF_") or not proof_str.endswith("_AUTHORIZED"):
            return False
            
        # Extract the hash from the proof string
        # Format: PROOF_<hash>_AUTHORIZED
        embedded_hash = proof_str[6:-11]
        
        # Check if the embedded hash matches one of the public inputs (the model hash)
        if embedded_hash in public_inputs:
            return True
            
        return False
        
    except Exception:
        return False

if __name__ == "__main__":
    # Test the mock logic
    test_hash = "abc123456789"
    print(f"Testing with hash: {test_hash}")
    
    proof = generate_proof(test_hash)
    print(f"Generated proof: {proof}")
    
    is_valid = verify_proof(proof, [test_hash])
    print(f"Verification result (expected True): {is_valid}")
    
    is_valid_bad = verify_proof(proof, ["points_to_different_hash"])
    print(f"Verification result (expected False): {is_valid_bad}")
