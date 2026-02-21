
import sys
import os
import torch
from ai_engine.train import load_and_preprocess_data, train_local
from ai_engine.model import SimpleNN

def verify_integration():
    print("=== Verifying Custom Dataset Integration ===")
    
    # 1. Test Data Loading for Node 1
    print("\n1. Testing Data Loading (Node 1/2)...")
    try:
        train_loader1, input_shape1 = load_and_preprocess_data(node_id=1, total_nodes=2)
        print(f"✅ Node 1 Data Loaded. Input Shape: {input_shape1}")
        
        # Check batch
        data, labels = next(iter(train_loader1))
        print(f"   Batch Data Shape: {data.shape}")
        print(f"   Batch Labels Shape: {labels.shape}")
        
    except Exception as e:
        print(f"❌ Node 1 Data Loading Failed: {e}")
        return

    # 2. Test Data Loading for Node 2
    print("\n2. Testing Data Loading (Node 2/2)...")
    try:
        train_loader2, input_shape2 = load_and_preprocess_data(node_id=2, total_nodes=2)
        print(f"✅ Node 2 Data Loaded. Input Shape: {input_shape2}")
    except Exception as e:
        print(f"❌ Node 2 Data Loading Failed: {e}")
        return

    # 3. Test Model Initialization
    print("\n3. Testing Model Initialization...")
    try:
        model = SimpleNN(input_size=input_shape1, hidden_size=64, num_classes=2)
        print(f"✅ Model Initialized with input_size={input_shape1}")
    except Exception as e:
        print(f"❌ Model Initialization Failed: {e}")
        return

    # 4. Test Training Step
    print("\n4. Testing Training Step...")
    try:
        weights = train_local(model, train_loader1, epochs=1)
        print(f"✅ Training completed successfully. Weights count: {len(weights)}")
    except Exception as e:
        print(f"❌ Training Failed: {e}")
        return

    print("\n=== Verification Successful ===")

if __name__ == "__main__":
    verify_integration()
