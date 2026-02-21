import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Tuple, List
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from .model import SimpleNN
from .han_encryption import HANEncryption
import os

# --- Configuration ---
CSV_PATH = "g:/CIP/Decentralized_AI_Platform/synthetic_fraud_dataset.csv"
TARGET_COLUMN = "Fraud_Label"
DROP_COLUMNS = ["Transaction_ID", "User_ID", "Timestamp"]
CATEGORICAL_COLUMNS = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type", "Authentication_Method"]

def load_and_preprocess_data(node_id: int = 1, total_nodes: int = 2, batch_size: int = 32) -> Tuple[DataLoader, int]:
    """
    Loads the fraud dataset, preprocesses it, and returns a DataLoader for the specific node's partition.
    Returns: (DataLoader, input_shape)
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")

    print(f"Node {node_id}: Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Drop irrelevant columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors='ignore')
    
    # 2. Encode Categorical Variables
    label_encoders = {}
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
            
    # 3. Separate Features and Target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int64)
    
    # 4. Normalize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5. Partition Data for Federated Learning
    # Simple partitioning: Split data into `total_nodes` chunks and pick the `node_id-1`-th chunk
    total_samples = len(X)
    partition_size = total_samples // total_nodes
    start_idx = (node_id - 1) * partition_size
    end_idx = start_idx + partition_size
    
    # Handle the last node taking the remainder
    if node_id == total_nodes:
        end_idx = total_samples
        
    X_node = X[start_idx:end_idx]
    y_node = y[start_idx:end_idx]
    
    print(f"Node {node_id}: Training on samples {start_idx} to {end_idx} (Total: {len(X_node)})")
    
    # 6. Create TensorDataset and DataLoader
    tensor_x = torch.Tensor(X_node)
    tensor_y = torch.LongTensor(y_node)
    
    dataset = TensorDataset(tensor_x, tensor_y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    input_shape = X.shape[1]
    return dataloader, input_shape

def add_differential_privacy_noise(weights: List[float], noise_scale: float = 0.01) -> List[float]:
    noisy_weights = []
    for weight in weights:
        noise = np.random.normal(0, noise_scale)
        noisy_weights.append(weight + noise)
    return noisy_weights

def train_local(model: nn.Module, train_loader: DataLoader, epochs: int = 1, lr: float = 0.01, dp_noise_scale: float = 0.01) -> Tuple[List[float], float]:
    """
    Trains the model locally and returns the updated weights and average loss.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    model.train()
    avg_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / num_batches
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
            
    accuracy = total_correct / total_samples if total_samples > 0 else 0
    print(f"Training Accuracy: {accuracy:.4f}")
    
    weights = model.get_weights()
    noisy_weights = add_differential_privacy_noise(weights, dp_noise_scale)
    
    return noisy_weights, avg_loss, accuracy

def train_local_standalone_v2(input_size: int = 17, hidden_size: int = 64, num_classes: int = 2, 
                          data_node_id: int = 1, total_nodes: int = 2, epochs: int = 5, dp_noise_scale: float = 0.01) -> Tuple[List[float], str, float, float, List[float]]:
    """
    Trains a model locally for a specific node, encrypts the weights, and generates a hash.
    Returns: (encrypted_weights, model_hash, average_loss, accuracy, trained_weights)
    """
    from .model import create_model
    
    # 1. Initialize Model
    model = create_model(input_size, hidden_size, num_classes)
    
    # 2. Load Real Dataset
    try:
        train_loader, _ = load_and_preprocess_data(node_id=data_node_id, total_nodes=total_nodes)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return [], "", 0.0, 0.0, []
    
    # 3. Train Locally
    # Update unpacking to include accuracy
    trained_weights, avg_loss, accuracy = train_local(model, train_loader, epochs, dp_noise_scale=dp_noise_scale)
    
    # 4. Encrypt with HAN
    print("🔒 Encrypting weights with HAN...")
    han = HANEncryption(key_size=2000)
    encrypted_weights = han.encrypt_weights(trained_weights)
    
    # 5. Generate Secure Hash
    # We hash the ENCRYPTED weights so the blockchain sees the secure version
    import hashlib
    weights_str = str(encrypted_weights)
    model_hash = hashlib.sha256(weights_str.encode()).hexdigest()
    
    return encrypted_weights, model_hash, avg_loss, accuracy, trained_weights

if __name__ == "__main__":
    weights, model_hash = train_local_standalone()
    print(f"Training completed!")
    print(f"Model hash: {model_hash}")
    print(f"Number of weights: {len(weights)}")
    print(f"Sample weights (first 5): {weights[:5]}")
