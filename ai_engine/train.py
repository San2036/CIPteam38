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
from opacus import PrivacyEngine

class EarlyStopping:
    def __init__(self, patience=2, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# --- Multi-Bank Dataset Registry ---
# Each node_id maps to its own independent bank dataset.
# Bank 1: Original synthetic fraud dataset
# Bank 2: Adapted EU Credit Card fraud dataset (creditcard.csv)
# Bank 3: Adapted PaySim Mobile Money fraud dataset (PaySim CSV)

import sys as _sys
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATASET_REGISTRY = {
    1: os.path.join(_BASE_DIR, "synthetic_fraud_dataset.csv"),
    2: os.path.join(_BASE_DIR, "bank2_adapted.csv"),
    3: os.path.join(_BASE_DIR, "bank3_adapted.csv"),
}

TARGET_COLUMN = "Fraud_Label"
# Columns to drop if present (IDs/timestamps not in adapted datasets, but safe to try)
DROP_COLUMNS = ["Transaction_ID", "User_ID", "Timestamp"]
# Categorical columns that need label encoding — only those that are string-typed will be encoded
CATEGORICAL_COLUMNS = ["Transaction_Type", "Device_Type", "Location", "Merchant_Category", "Card_Type", "Authentication_Method"]

def load_and_preprocess_data(node_id: int = 1, total_nodes: int = 3, batch_size: int = 32, test_size: float = 0.2) -> Tuple[DataLoader, DataLoader, int]:
    """
    Loads the bank's own private dataset (based on node_id), preprocesses it,
    and splits it into Train and Validation sets.
    
    Each bank/node has its own independent dataset — no row-splitting across banks.
    Returns: (train_loader, val_loader, input_shape)
    """
    # Resolve dataset path from registry
    csv_path = DATASET_REGISTRY.get(node_id)
    if csv_path is None:
        # Fallback: if node_id not in registry, cycle through available datasets
        fallback_id = (node_id % len(DATASET_REGISTRY)) + 1
        csv_path = DATASET_REGISTRY.get(fallback_id, list(DATASET_REGISTRY.values())[0])
        print(f"Node {node_id}: No dataset registered. Using fallback: {csv_path}")
    
    if not os.path.exists(csv_path):
        # If adapted dataset not yet generated, run the adapter automatically
        print(f"Node {node_id}: Dataset not found at {csv_path}. Attempting to generate...")
        _run_adapter_for_node(node_id)
        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"Dataset not found at {csv_path}.\n"
                f"Please run: python dataset_adapter.py  (from project root)"
            )

    print(f"Node {node_id}: Loading its OWN private dataset from {os.path.basename(csv_path)}...")
    df = pd.read_csv(csv_path)
    print(f"Node {node_id}: Loaded {len(df):,} rows, {len(df.columns)} columns")
    
    # 1. Drop irrelevant ID/timestamp columns if present
    df = df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], errors='ignore')
    
    # 2. Encode any remaining string categorical columns
    for col in df.columns:
        if df[col].dtype == object:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            
    # 3. Separate Features and Target
    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in dataset at {csv_path}.\n"
            f"Available columns: {list(df.columns)}"
        )
        
    X = df.drop(columns=[TARGET_COLUMN]).values.astype(np.float32)
    y = df[TARGET_COLUMN].values.astype(np.int64)
    
    # 4. Normalize Features (each bank's scaler is private — that's the FL point!)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # 5. Train/Validation split on the FULL private bank dataset
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Node {node_id} [{os.path.basename(csv_path)}]: Train={len(X_train):,}, Val={len(X_val):,}, Features={X.shape[1]}")
    
    # 6. Create DataLoaders
    train_loader = DataLoader(
        TensorDataset(torch.Tensor(X_train), torch.LongTensor(y_train)),
        batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(torch.Tensor(X_val), torch.LongTensor(y_val)),
        batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader, X.shape[1]


def _run_adapter_for_node(node_id: int):
    """Auto-run dataset_adapter.py if adapted dataset is missing."""
    try:
        adapter_path = os.path.join(_BASE_DIR, "dataset_adapter.py")
        if os.path.exists(adapter_path):
            print(f"Running dataset_adapter.py to generate missing datasets...")
            import subprocess
            result = subprocess.run(
                [_sys.executable, adapter_path],
                cwd=_BASE_DIR,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode == 0:
                print("Dataset adapter completed successfully.")
            else:
                print(f"Adapter error: {result.stderr}")
    except Exception as e:
        print(f"Could not auto-run adapter: {e}")

from sklearn.metrics import precision_score, recall_score, f1_score

def train_local(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, epochs: int = 1, lr: float = 0.01, dp_noise_scale: float = 0.01) -> Tuple[List[float], float, float, float, float, float, float]:
    """
    Trains the model locally with Opacus DP and Early Stopping.
    """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    
    privacy_engine = PrivacyEngine()
    model, optimizer, train_loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=train_loader,
        noise_multiplier=dp_noise_scale,
        max_grad_norm=1.0,
    )
    
    early_stopping = EarlyStopping(patience=2)
    
    # Training Loop
    for epoch in range(epochs):
        model.train()
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
        
        # Validation Phase per epoch
        model.eval()
        val_loss_total = 0.0
        val_batches = 0
        val_correct = 0
        val_total_samples = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs)
                v_loss = criterion(outputs, labels)
                val_loss_total += v_loss.item()
                val_batches += 1
                
                _, predicted = torch.max(outputs.data, 1)
                val_total_samples += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
        val_accuracy = val_correct / val_total_samples if val_total_samples > 0 else 0
        val_epoch_loss = val_loss_total / (val_batches if val_batches > 0 else 1)
        
        precision = float(precision_score(all_labels, all_preds, zero_division=0))
        recall = float(recall_score(all_labels, all_preds, zero_division=0))
        f1 = float(f1_score(all_labels, all_preds, zero_division=0))
        
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_accuracy:.4f}")
        
        early_stopping(val_epoch_loss)
        if early_stopping.early_stop:
            print(f"Early stopping triggered at Epoch {epoch+1}!")
            break
            
    print(f"Final Metrics -> Accuracy: {val_accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
    
    try:
        epsilon = privacy_engine.get_epsilon(delta=1e-5)
        print(f"DP Epsilon Spent: {epsilon:.2f}")
    except:
        pass
        
    weights = model._module.get_weights() if hasattr(model, '_module') else model.get_weights()
    return weights, avg_loss, train_accuracy, val_accuracy, precision, recall, f1

def train_local_standalone_v2(input_size: int = 17, hidden_size: int = 64, num_classes: int = 2, 
                          data_node_id: int = 1, total_nodes: int = 3, epochs: int = 5, dp_noise_scale: float = 0.01,
                          initial_weights: List[float] = None) -> Tuple[List[float], str, float, dict, List[float]]:
    """
    Trains a model locally with a validation split and returns advanced metrics.
    Returns: (encrypted_weights, model_hash, average_loss, metrics_dict, trained_weights)
    """
    from .model import create_model
    model = create_model(input_size, hidden_size, num_classes)
    
    if initial_weights is not None:
        print(f"Node {data_node_id}: Initializing model with Federated Global Weights...")
        model.set_weights(initial_weights)
    
    # 1. Load Data with Split
    train_loader, val_loader, _ = load_and_preprocess_data(node_id=data_node_id, total_nodes=total_nodes)
    
    # 2. Train and Validate
    results = train_local(model, train_loader, val_loader, epochs, dp_noise_scale=dp_noise_scale)
    trained_weights, avg_loss, train_acc, val_acc, prec, rec, f1 = results
    
    metrics = {
        "accuracy": val_acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1
    }
    
    # 3. Encrypt and Hash
    print(f"Node {data_node_id}: Applying HAN Encryption to {len(trained_weights)} weights...")
    han = HANEncryption(key_size=2000)
    encrypted_weights = han.encrypt_weights(trained_weights)
    
    import hashlib
    model_hash = hashlib.sha256(str(encrypted_weights).encode()).hexdigest()
    
    return encrypted_weights, model_hash, avg_loss, metrics, trained_weights

if __name__ == "__main__":
    results = train_local_standalone_v2()
    weights, model_hash, avg_loss, metrics, raw_weights = results
    print(f"Training completed!")
    print(f"Model hash: {model_hash}")
    print(f"Number of weights: {len(weights)}")
    print(f"Sample weights (first 5): {weights[:5]}")
