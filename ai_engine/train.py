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

def load_and_preprocess_data(node_id: int = 1, total_nodes: int = 2, batch_size: int = 32, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, int]:
    """
    Loads the fraud dataset, preprocesses it, partitions it, and then
    splits the node's partition into Train and Validation sets.
    Returns: (train_loader, val_loader, input_shape)
    """
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError(f"Dataset not found at {CSV_PATH}")

    print(f"Node {node_id}: Loading dataset from {CSV_PATH}...")
    df = pd.read_csv(CSV_PATH)
    
    # 1. Drop irrelevant columns
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors='ignore')
    
    # 2. Encode Categorical Variables
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    # 3. Separate Features and Target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in dataset.")
        
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int64)
    
    # 4. Normalize Features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5. Partition Data for Federated Learning (Horizontal Split)
    total_samples = len(X)
    partition_size = total_samples // total_nodes
    start_idx = (node_id - 1) * partition_size
    end_idx = start_idx + partition_size
    if node_id == total_nodes:
        end_idx = total_samples
        
    X_node = X[start_idx:end_idx]
    y_node = y[start_idx:end_idx]
    
    # 6. Internal Train/Test Split (The "Final Exam")
    X_train, X_val, y_train, y_val = train_test_split(X_node, y_node, test_size=test_size, random_state=42)
    
    print(f"Node {node_id}: Data Partitioned -> Train: {len(X_train)}, Val: {len(X_val)}")
    
    # 7. Create DataLoaders
    train_loader = DataLoader(TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val)), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X.shape[1]

def add_differential_privacy_noise(weights: List[float], noise_scale: float = 0.01) -> List[float]:
    noisy_weights = []
    for weight in weights:
        noise = np.random.normal(0, noise_scale)
        noisy_weights.append(weight + noise)
    return noisy_weights

def train_local(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 1, lr: float = 0.01, dp_noise_scale: float = 0.01) -> Tuple[List[float], float, float, float]:
    """
    Trains the model locally and returns the updated weights, average loss, train_acc, and val_acc.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    # Training Loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        total_correct = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            total_correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / num_batches
        train_accuracy = total_correct / total_samples if total_samples > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}")

    # Validation Phase (The "Final Exam")
    model.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_accuracy = val_correct / val_total if val_total > 0 else 0
    print(f"Validation Accuracy: {val_accuracy:.4f}")
    
    weights = model.get_weights()
    noisy_weights = add_differential_privacy_noise(weights, dp_noise_scale)
    
    return noisy_weights, avg_loss, train_accuracy, val_accuracy

def train_local_standalone_v2(input_size: int = 17, hidden_size: int = 64, num_classes: int = 2, 
                          data_node_id: int = 1, total_nodes: int = 2, epochs: int = 5, dp_noise_scale: float = 0.01) -> Tuple[List[float], str, float, float, List[float]]:
    """
    Trains a model locally with a validation split.
    Returns: (encrypted_weights, model_hash, average_loss, val_accuracy, trained_weights)
    """
    from .model import create_model
    model = create_model(input_size, hidden_size, num_classes)
    
    # 1. Load Data with Split
    train_loader, val_loader, _ = load_and_preprocess_data(node_id=data_node_id, total_nodes=total_nodes)
    
    # 2. Train and Validate
    trained_weights, avg_loss, train_acc, val_acc = train_local(model, train_loader, val_loader, epochs, dp_noise_scale=dp_noise_scale)
    
    # 3. Encrypt and Hash (Using HAN key size 2000 for blockchain uniformity)
    han = HANEncryption(key_size=2000)
    encrypted_weights = han.encrypt_weights(trained_weights)
    
    import hashlib
    model_hash = hashlib.sha256(str(encrypted_weights).encode()).hexdigest()
    
    return encrypted_weights, model_hash, avg_loss, val_acc, trained_weights

if __name__ == "__main__":
    weights, model_hash = train_local_standalone()
    print(f"Training completed!")
    print(f"Model hash: {model_hash}")
    print(f"Number of weights: {len(weights)}")
    print(f"Sample weights (first 5): {weights[:5]}")
