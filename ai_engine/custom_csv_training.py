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

class CustomCSVTrainer:
    """Train models with custom CSV datasets"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
    
    def load_custom_csv(self, csv_path: str, target_column: str = None) -> Tuple[DataLoader, DataLoader]:
        """
        Load and prepare custom CSV dataset
        
        Args:
            csv_path: Path to your CSV file
            target_column: Name of the target column (if None, will try to detect)
        
        Returns:
            Tuple of (train_loader, test_loader)
        """
        print(f"📁 Loading custom CSV: {csv_path}")
        
        # Load CSV
        try:
            df = pd.read_csv(csv_path)
            print(f"✅ CSV loaded successfully!")
            print(f"📊 Dataset shape: {df.shape}")
            print(f"📋 Columns: {list(df.columns)}")
        except Exception as e:
            print(f"❌ Error loading CSV: {e}")
            return None, None
        
        # Detect target column if not specified
        if target_column is None:
            target_column = self._detect_target_column(df)
            print(f"🎯 Auto-detected target column: {target_column}")
        
        # Preprocess data
        X, y = self._preprocess_custom_data(df, target_column)
        
        if X is None or y is None:
            return None, None
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y if len(np.unique(y)) > 1 else None
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
        
        print(f"✅ Dataset prepared: {len(X_train)} training samples, {len(X_test)} test samples")
        print(f"📊 Classes: {np.unique(y, return_counts=True)}")
        
        return train_loader, test_loader
    
    def _detect_target_column(self, df: pd.DataFrame) -> str:
        """Automatically detect the target column"""
        # Common target column names
        target_candidates = [
            'target', 'label', 'class', 'y', 'output', 'result',
            'is_fraud', 'fraud', 'default', 'churn', 'survived',
            'diagnosis', 'disease', 'outcome', 'status'
        ]
        
        # Check for exact matches
        for col in df.columns:
            if col.lower() in target_candidates:
                return col
        
        # Check for columns with few unique values (likely categorical target)
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10:
                if col not in ['id', 'ID', 'index', 'name', 'date', 'time']:
                    return col
        
        # If no clear target, use the last column
        return df.columns[-1]
    
    def _preprocess_custom_data(self, df: pd.DataFrame, target_column: str) -> Tuple[np.ndarray, np.ndarray]:
        """Preprocess custom CSV data"""
        try:
            # Separate features and target
            if target_column not in df.columns:
                print(f"❌ Target column '{target_column}' not found in dataset")
                return None, None
            
            X = df.drop(target_column, axis=1)
            y = df[target_column]
            
            print(f"📊 Features shape: {X.shape}")
            print(f"🎯 Target column: {target_column}")
            
            # Handle categorical features
            categorical_cols = X.select_dtypes(include=['object', 'category']).columns
            for col in categorical_cols:
                if X[col].nunique() <= 10:  # Only encode low-cardinality columns
                    X[col] = self.label_encoder.fit_transform(X[col].astype(str))
                else:
                    # Drop high-cardinality categorical columns
                    X = X.drop(col, axis=1)
                    print(f"⚠️ Dropped high-cardinality column: {col}")
            
            # Handle missing values
            X = X.fillna(X.mean())
            
            # Convert target to numeric
            if y.dtype == 'object':
                y = self.label_encoder.fit_transform(y.astype(str))
            elif y.dtype == 'bool':
                y = y.astype(int)
            elif y.dtype not in ['int64', 'float64']:
                y = pd.to_numeric(y, errors='coerce').fillna(0).astype(int)
            
            # Scale features
            X_scaled = self.scaler.fit_transform(X)
            
            print(f"✅ Data preprocessing completed")
            print(f"📊 Final features shape: {X_scaled.shape}")
            print(f"🎯 Unique classes: {np.unique(y)}")
            
            return X_scaled, y.values
            
        except Exception as e:
            print(f"❌ Error preprocessing data: {e}")
            return None, None
    
    def train_custom_model(
        self,
        csv_path: str = "g:/CIP/Decentralized_AI_Platform/synthetic_fraud_dataset.csv", # Hardcoded default
        target_column: str = "Fraud_Label", # Hardcoded default
        epochs: int = 5,
        learning_rate: float = 0.001,
        num_samples: int = 1000
    ) -> Tuple[List[float], str]:
        """
        Train model with specific fraud dataset
        
        Args:
            csv_path: Path to your CSV file
            target_column: Target column name
            epochs: Number of training epochs
            learning_rate: Learning rate
            num_samples: Maximum number of samples to use
        
        Returns:
            Tuple of (losses, model_hash)
        """
        print(f"🎯 Training with dataset: {csv_path}")
        
        if not os.path.exists(csv_path):
             print(f"❌ Dataset not found at {csv_path}")
             return [], ""

        # Load data
        train_loader, test_loader = self.load_custom_csv(csv_path, target_column)
        
        if train_loader is None:
            print("❌ Failed to load CSV data")
            return [], ""
        
        # Create model - using new defaults (29 inputs, 2 classes)
        input_size = len(next(iter(train_loader))[0][0])
        model = SimpleNN(input_size=input_size, hidden_size=64, num_classes=2)
        
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

def train_with_custom_csv(
    csv_path: str = "g:/CIP/Decentralized_AI_Platform/synthetic_fraud_dataset.csv",
    target_column: str = "Fraud_Label",
    epochs: int = 5,
    num_samples: int = 1000
) -> Tuple[List[float], str]:
    """
    Convenience function to train with the fraud dataset
    """
    trainer = CustomCSVTrainer()
    return trainer.train_custom_model(
        csv_path=csv_path,
        target_column=target_column,
        epochs=epochs,
        num_samples=num_samples
    )

def analyze_csv_dataset(csv_path: str):
    """Analyze CSV dataset before training"""
    print(f"🔍 Analyzing CSV dataset: {csv_path}")
    
    try:
        df = pd.read_csv(csv_path)
        
        print(f"\n📊 Dataset Overview:")
        print(f"Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        
        print(f"\n📋 Column Details:")
        for col in df.columns:
            dtype = df[col].dtype
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            
            print(f"  {col}:")
            print(f"    Type: {dtype}")
            print(f"    Unique values: {unique_count}")
            print(f"    Missing: {missing_count}")
            
            if unique_count <= 10:
                print(f"    Values: {df[col].value_counts().to_dict()}")
        
        print(f"\n🎯 Recommended target columns:")
        trainer = CustomCSVTrainer()
        target_col = trainer._detect_target_column(df)
        print(f"  Primary: {target_col}")
        
        # Show other potential targets
        target_candidates = []
        for col in df.columns:
            if df[col].dtype in ['object', 'category'] or df[col].nunique() <= 10:
                if col not in ['id', 'ID', 'index', 'name', 'date', 'time']:
                    target_candidates.append(col)
        
        print(f"  Other options: {target_candidates}")
        
    except Exception as e:
        print(f"❌ Error analyzing CSV: {e}")

# Example usage
if __name__ == "__main__":
    # Example with specific fraud dataset - now defaults are hardcoded
    csv_file = "g:/CIP/Decentralized_AI_Platform/synthetic_fraud_dataset.csv" 
    
    # First analyze the dataset
    analyze_csv_dataset(csv_file)
    
    # Then train the model
    losses, model_hash = train_with_custom_csv() # Uses defaults
    
    print(f"✅ Training completed! Final loss: {losses[-1]:.4f}")
    print(f"🔢 Model hash: {model_hash}")
