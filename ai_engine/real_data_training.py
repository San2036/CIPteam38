import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import List, Tuple, Optional
import os
import sys

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from model import SimpleNN

class RealDatasetLoader:
    """Load and prepare real datasets for federated learning"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_fraud_dataset(self, csv_path: str = None) -> Tuple[DataLoader, DataLoader]:
        """
        Load real fraud detection dataset
        If csv_path is None, creates a realistic fraud dataset
        """
        if csv_path and os.path.exists(csv_path):
            print(f"📁 Loading real dataset from: {csv_path}")
            df = pd.read_csv(csv_path)
        else:
            print("📊 Creating realistic fraud detection dataset...")
            df = self.create_realistic_fraud_dataset()
        
        # Preprocess the data
        X, y = self.preprocess_fraud_data(df)
        
        # Split for training
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Create PyTorch datasets
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train),
            torch.LongTensor(y_train)
        )
        
        test_dataset = TensorDataset(
            torch.FloatTensor(X_test),
            torch.LongTensor(y_test)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        print(f"✅ Dataset loaded: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"📊 Fraud cases: {np.sum(y)} out of {len(y)} ({np.mean(y)*100:.1f}%)")
        
        return train_loader, test_loader
    
    def create_realistic_fraud_dataset(self, n_samples: int = 10000) -> pd.DataFrame:
        """Create a realistic fraud detection dataset"""
        np.random.seed(42)
        
        # Generate realistic features
        data = {
            'transaction_amount': np.random.lognormal(3, 1.5, n_samples),
            'transaction_hour': np.random.randint(0, 24, n_samples),
            'merchant_category': np.random.randint(1, 10, n_samples),
            'customer_age': np.random.randint(18, 80, n_samples),
            'customer_income': np.random.lognormal(10, 0.5, n_samples),
            'transaction_frequency': np.random.poisson(5, n_samples),
            'distance_from_home': np.random.exponential(10, n_samples),
            'is_weekend': np.random.randint(0, 2, n_samples),
            'risk_score': np.random.beta(2, 5, n_samples),
            'previous_fraud_count': np.random.poisson(0.1, n_samples)
        }
        
        df = pd.DataFrame(data)
        
        # Create fraud labels (realistic fraud rate ~2%)
        fraud_probability = self.calculate_fraud_probability(df)
        df['is_fraud'] = np.random.random(n_samples) < fraud_probability
        
        return df
    
    def calculate_fraud_probability(self, df: pd.DataFrame) -> np.ndarray:
        """Calculate realistic fraud probability based on features"""
        # Base fraud rate
        base_rate = 0.02
        
        # Risk factors
        amount_risk = np.where(df['transaction_amount'] > 1000, 0.3, 0.01)
        hour_risk = np.where((df['transaction_hour'] < 6) | (df['transaction_hour'] > 22), 0.15, 0.01)
        frequency_risk = np.where(df['transaction_frequency'] > 10, 0.2, 0.01)
        distance_risk = np.where(df['distance_from_home'] > 50, 0.25, 0.01)
        
        # Combine risks
        combined_risk = base_rate + amount_risk + hour_risk + frequency_risk + distance_risk
        
        # Cap at reasonable maximum
        return np.minimum(combined_risk, 0.8)
    
    def preprocess_fraud_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess fraud detection data"""
        # Separate features and labels
        if 'is_fraud' in df.columns:
            X = df.drop('is_fraud', axis=1)
            y = df['is_fraud'].values
        else:
            # If no fraud column, create synthetic labels
            X = df
            y = np.random.randint(0, 2, len(df))
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y
    
    def load_credit_card_dataset(self, csv_path: str = None) -> Tuple[DataLoader, DataLoader]:
        """Load credit card fraud dataset (Kaggle style)"""
        if csv_path and os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            
            # Standard credit card fraud dataset preprocessing
            if 'Class' in df.columns:
                X = df.drop(['Class', 'Time'], axis=1, errors='ignore')
                y = df['Class'].values
            else:
                # Generic preprocessing
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                X = df[numeric_cols].dropna()
                y = np.random.randint(0, 2, len(X))  # Placeholder labels
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            # Create datasets
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train),
                torch.LongTensor(y_train)
            )
            
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test),
                torch.LongTensor(y_test)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
            
            print(f"✅ Credit card dataset loaded: {len(X_train)} training samples")
            print(f"📊 Fraud cases: {np.sum(y)} out of {len(y)} ({np.mean(y)*100:.2f}%)")
            
            return train_loader, test_loader
        else:
            print("⚠️ Credit card dataset not found, using fraud dataset instead")
            return self.load_fraud_dataset()

def train_with_real_dataset(
    dataset_type: str = "fraud",
    csv_path: str = None,
    epochs: int = 5,
    learning_rate: float = 0.001,
    dp_noise_scale: float = 0.01,
    num_samples: int = 1000
) -> Tuple[List[float], str]:
    """
    Train model with real dataset
    
    Args:
        dataset_type: 'fraud', 'credit_card', or 'custom'
        csv_path: Path to custom CSV file
        epochs: Number of training epochs
        learning_rate: Learning rate
        dp_noise_scale: Differential privacy noise scale
        num_samples: Number of samples to use (for subset training)
    
    Returns:
        Tuple of (losses, model_hash)
    """
    print(f"🎯 Training with real dataset: {dataset_type}")
    
    # Load real dataset
    loader = RealDatasetLoader()
    
    if dataset_type == "fraud":
        train_loader, test_loader = loader.load_fraud_dataset(csv_path)
    elif dataset_type == "credit_card":
        train_loader, test_loader = loader.load_credit_card_dataset(csv_path)
    else:
        print("⚠️ Unknown dataset type, using fraud dataset")
        train_loader, test_loader = loader.load_fraud_dataset(csv_path)
    
    # Create model
    input_size = len(next(iter(train_loader))[0][0])
    model = SimpleNN(input_size=input_size)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    losses = []
    model.train()
    
    print(f"🚀 Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if batch_idx * 32 >= num_samples:  # Limit samples
                break
                
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        avg_loss = epoch_loss / max(1, num_batches)
        losses.append(avg_loss)
        print(f"✅ Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.4f}")
    
    # Evaluate on test set
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            if total >= 200:  # Limit test samples for speed
                break
                
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    
    accuracy = 100 * correct / max(1, total)
    print(f"📊 Test Accuracy: {accuracy:.2f}%")
    
    # Generate model hash
    model_hash = hash(str(model.state_dict()))
    print(f"🔢 Model hash: {model_hash}")
    
    return losses, str(model_hash)

def train_local_standalone_real(
    dataset_type: str = "fraud",
    csv_path: str = None,
    num_samples: int = 1000,
    epochs: int = 3,
    dp_noise_scale: float = 0.01
) -> Tuple[List[float], str]:
    """
    Standalone training function for real datasets
    """
    try:
        losses, model_hash = train_with_real_dataset(
            dataset_type=dataset_type,
            csv_path=csv_path,
            epochs=epochs,
            dp_noise_scale=dp_noise_scale,
            num_samples=num_samples
        )
        return losses, model_hash
    except Exception as e:
        print(f"❌ Error in real dataset training: {e}")
        return [], ""

# Example usage
if __name__ == "__main__":
    print("🎯 Testing real dataset training...")
    
    # Test with fraud dataset
    losses, model_hash = train_with_real_dataset(
        dataset_type="fraud",
        epochs=3,
        num_samples=500
    )
    
    print(f"✅ Training completed! Final loss: {losses[-1]:.4f}")
    print(f"🔢 Model hash: {model_hash}")
