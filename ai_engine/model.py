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
        state_dict = self.state_dict()
        idx = 0
        for key in state_dict.keys():
            param = state_dict[key]
            num_elements = param.numel()
            # Precisely take only what we need for this parameter
            flat_slice = weights[idx : idx + num_elements]
            # Convert to tensor and reshape
            state_dict[key] = torch.tensor(flat_slice).reshape(param.shape).float()
            idx += num_elements
        self.load_state_dict(state_dict)
    
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
