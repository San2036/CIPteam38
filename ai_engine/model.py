import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Tuple

class SimpleNN(nn.Module):
    def __init__(self, input_size=17, hidden_size=64, num_classes=2):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out
    
    def get_weights(self):
        weights = []
        for param in self.parameters():
            weights.extend(param.data.cpu().numpy().flatten().tolist())
        return weights

    def set_weights(self, weights):
        # Determine shapes first
        fc1_w_shape = self.fc1.weight.shape
        fc1_b_shape = self.fc1.bias.shape
        fc2_w_shape = self.fc2.weight.shape
        fc2_b_shape = self.fc2.bias.shape
        
        # Slicing indices
        fc1_w_end = fc1_w_shape[0] * fc1_w_shape[1]
        fc1_b_end = fc1_w_end + fc1_b_shape[0]
        fc2_w_end = fc1_b_end + fc2_w_shape[0] * fc2_w_shape[1]
        
        # Reconstruct tensors
        self.fc1.weight.data = torch.tensor(weights[:fc1_w_end]).reshape(fc1_w_shape).float()
        self.fc1.bias.data = torch.tensor(weights[fc1_w_end:fc1_b_end]).reshape(fc1_b_shape).float()
        self.fc2.weight.data = torch.tensor(weights[fc1_b_end:fc2_w_end]).reshape(fc2_w_shape).float()
        self.fc2.bias.data = torch.tensor(weights[fc2_w_end:]).reshape(fc2_b_shape).float()
    
    def get_model_hash(self):
        import hashlib
        weights_str = str(self.get_weights())
        return hashlib.sha256(weights_str.encode()).hexdigest()

def create_model(input_size=17, hidden_size=64, num_classes=2):
    return SimpleNN(input_size, hidden_size, num_classes)

def load_model_from_weights(weights: List[float], input_size: int = 17, hidden_size: int = 64, num_classes: int = 2) -> SimpleNN:
    model = SimpleNN(input_size, hidden_size, num_classes)
    model.set_weights(weights)
    return model
