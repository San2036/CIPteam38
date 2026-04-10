import numpy as np
from typing import List, Tuple

class HANEncryption:
    def __init__(self, key_size: int = 100):
        self.key_size = key_size
        self.encryption_key = self._generate_encryption_key()
        self.decryption_key = self._generate_decryption_key()
    
    def _generate_encryption_key(self) -> np.ndarray:
        np.random.seed(42)
        return np.random.randn(self.key_size, self.key_size)
    
    def _generate_decryption_key(self) -> np.ndarray:
        return np.linalg.inv(self.encryption_key)
    
    def encrypt_weights(self, weights: List[float]) -> List[float]:
        weights_array = np.array(weights)
        
        if len(weights_array) < self.key_size:
            padding_size = self.key_size - len(weights_array)
            weights_array = np.pad(weights_array, (0, padding_size), 'constant')
        elif len(weights_array) > self.key_size:
            weights_array = weights_array[:self.key_size]
        
        encrypted = np.dot(self.encryption_key, weights_array)
        return encrypted.tolist()
    
    def decrypt_weights(self, encrypted_weights: List[float]) -> List[float]:
        encrypted_array = np.array(encrypted_weights)
        decrypted = np.dot(self.decryption_key, encrypted_array)
        return decrypted.tolist()
    
    def get_public_key(self) -> np.ndarray:
        return self.encryption_key
    
    def set_public_key(self, public_key: np.ndarray):
        self.encryption_key = public_key
        self.decryption_key = np.linalg.inv(public_key)

class SimulatedHomomorphicEncryption:
    def __init__(self, noise_factor: float = 0.1):
        self.noise_factor = noise_factor
        self.secret_key = np.random.randn(100)
    
    def encrypt(self, values: List[float]) -> List[float]:
        encrypted = []
        for val in values:
            noise = np.random.normal(0, self.noise_factor)
            encrypted_val = val + noise + np.dot(self.secret_key, np.random.randn(100))
            encrypted.append(encrypted_val)
        return encrypted
    
    def decrypt(self, encrypted_values: List[float]) -> List[float]:
        decrypted = []
        for enc_val in encrypted_values:
            noise = np.random.normal(0, self.noise_factor)
            dec_val = enc_val - noise - np.dot(self.secret_key, np.random.randn(100))
            decrypted.append(dec_val)
        return decrypted
    
    def aggregate_encrypted(self, encrypted_lists: List[List[float]]) -> List[float]:
        if not encrypted_lists:
            return []
        
        max_len = max(len(lst) for lst in encrypted_lists)
        aggregated = []
        
        for i in range(max_len):
            sum_val = 0.0
            count = 0
            for lst in encrypted_lists:
                if i < len(lst):
                    sum_val += lst[i]
                    count += 1
            aggregated.append(sum_val / count if count > 0 else 0.0)
        
        return aggregated

def simulate_han_encryption(weights: List[float]) -> Tuple[List[float], HANEncryption]:
    han = HANEncryption()
    encrypted = han.encrypt_weights(weights)
    return encrypted, han

if __name__ == "__main__":
    import sys
    import os
    
    # Add parent directory to path to allow imports if running directly
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from ai_engine.train import load_and_preprocess_data, train_local
    from ai_engine.model import SimpleNN
    
    print("=== Testing HAN Encryption with Real Dataset ===")
    
    # 1. Load Data
    print("Loading dataset...")
    try:
        # Load a small partition (Node 1 of 2) just to get data
        train_loader, input_shape = load_and_preprocess_data(node_id=1, total_nodes=2)
        print(f"Dataset loaded. Input shape: {input_shape}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Ensure 'synthetic_fraud_dataset.csv' is in the project root.")
        sys.exit(1)
        
    # 2. Train a simple model to get realistic weights
    print("Training simple model to generate weights...")
    model = SimpleNN(input_size=input_shape, hidden_size=64, num_classes=2)
    weights = train_local(model, train_loader, epochs=1) 
    print(f"Generated {len(weights)} weights from model.")
    
    # 3. Encrypt Weights
    print("\n--- Encrypting Weights ---")
    # Note: HANEncryption output size defaults to 100. 
    # For this demo, we will use a key size closer to the weight count or truncate slightly
    # to show functionality without excessive computation for a demo.
    # Let's use a smaller key_size for the demo speed, but show it working on real weights.
    
    han = HANEncryption(key_size=200) 
    print(f"Initialized HANEncryption with key_size={han.key_size}")
    
    encrypted = han.encrypt_weights(weights)
    print(f"Encrypted weights count: {len(encrypted)}")
    
    # 4. Decrypt Weights
    decrypted = han.decrypt_weights(encrypted)
    print(f"Decrypted weights count: {len(decrypted)}")
    
    # 5. Verify
    print("\nOriginal (truncated to key_size) vs Decrypted difference (first 5):")
    # We compare against the first 'key_size' weights because encryption truncates if weights > key_size
    original_segment = weights[:han.key_size]
    
    for i in range(5):
        diff = abs(original_segment[i] - decrypted[i])
        print(f"Index {i}: Original={original_segment[i]:.4f}, Decrypted={decrypted[i]:.4f}, Diff={diff:.10f}")
        
    print("\n=== Test Complete ===")
