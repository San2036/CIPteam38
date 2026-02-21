#!/usr/bin/env python3

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    
    print(f"✅ PyTorch version: {torch.__version__}")
    print(f"✅ CUDA available: {torch.cuda.is_available()}")
    
    # Test basic neural network
    class TestNN(nn.Module):
        def __init__(self):
            super(TestNN, self).__init__()
            self.fc1 = nn.Linear(10, 5)
            self.fc2 = nn.Linear(5, 2)
            
        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = self.fc2(x)
            return x
    
    # Test model creation
    model = TestNN()
    print("✅ Model created successfully")
    
    # Test data creation
    X = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    print("✅ DataLoader created successfully")
    
    # Test forward pass
    with torch.no_grad():
        output = model(X[:5])
        print(f"✅ Forward pass successful. Output shape: {output.shape}")
    
    print("\n🎉 All PyTorch tests passed!")
    
except Exception as e:
    print(f"❌ Error: {e}")
    print("\n🔧 Try reinstalling PyTorch:")
    print("pip uninstall torch torchvision torchaudio")
    print("pip install torch==1.12.0 torchvision==0.13.0 --extra-index-url https://download.pytorch.org/whl/cpu")
