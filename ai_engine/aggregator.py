import numpy as np
from typing import List
import os
import torch

def federated_average(weights_list: List[List[float]]) -> List[float]:
    """
    Performs Federated Averaging (FedAvg) on a list of model weights.
    Each element in weights_list is a flat list of floats representing a model.
    """
    if not weights_list:
        return []
    
    # Convert to numpy for efficient averaging
    arr = np.array(weights_list)
    avg_weights = np.mean(arr, axis=0)
    
    return avg_weights.tolist()

def save_global_model(weights: List[float], filename: str = "global_model_weights.pt"):
    """Saves the aggregated weights to a file for nodes to download."""
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    torch.save(weights, save_path)
    print(f"✅ Global Model saved to: {save_path}")

def load_global_model(filename: str = "global_model_weights.pt") -> List[float]:
    """Loads the latest global model weights."""
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", filename)
    if os.path.exists(save_path):
        return torch.load(save_path)
    return None

if __name__ == "__main__":
    # Test Averaging
    w1 = [1.0, 2.0, 3.0]
    w2 = [2.0, 4.0, 6.0]
    avg = federated_average([w1, w2])
    print(f"Test Avg: {avg}") # Expect [1.5, 3.0, 4.5]
