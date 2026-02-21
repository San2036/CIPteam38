import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from ai_engine.model import SimpleNN, load_model_from_weights
from ai_engine.train import create_dummy_dataset
from typing import List, Tuple, Dict

class FraudDetector:
    def __init__(self, model_weights: List[float] = None):
        if model_weights:
            self.model = load_model_from_weights(model_weights)
        else:
            self.model = SimpleNN()
        self.model.eval()
    
    def predict(self, transaction_features: List[float]) -> Dict:
        """
        Predict if transaction is fraudulent
        
        Args:
            transaction_features: List of transaction features
            [amount, balance, time, age, account_age, daily_count, avg_amount, distance, 
             transaction_type_encoded, merchant_category_encoded, device_type_encoded, segment_encoded]
        
        Returns:
            Dict with prediction details
        """
        try:
            # Convert to tensor
            features_tensor = torch.FloatTensor(transaction_features).unsqueeze(0)
            
            # Get prediction
            with torch.no_grad():
                outputs = self.model(features_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
            
            fraud_probability = probabilities[0][1].item()  # Probability of class 1 (fraud)
            is_fraud = predicted.item() == 1
            
            return {
                'is_fraud': is_fraud,
                'fraud_probability': fraud_probability,
                'confidence': confidence.item(),
                'prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
                'raw_probabilities': probabilities[0].tolist()
            }
            
        except Exception as e:
            return {
                'error': str(e),
                'is_fraud': False,
                'fraud_probability': 0.0,
                'prediction': 'ERROR'
            }
    
    def batch_predict(self, transactions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict fraud for multiple transactions
        
        Args:
            transactions_df: DataFrame with transaction features
            
        Returns:
            DataFrame with predictions added
        """
        results = []
        
        for idx, row in transactions_df.iterrows():
            # Extract features (ensure correct order)
            features = [
                row.get('transaction_amount', 0),
                row.get('account_balance', 0),
                row.get('transaction_time', 0),
                row.get('customer_age', 0),
                row.get('account_age_days', 0),
                row.get('daily_transaction_count', 0),
                row.get('avg_transaction_amount', 0),
                row.get('distance_from_home', 0),
                row.get('transaction_type_encoded', 0),
                row.get('merchant_category_encoded', 0),
                row.get('device_type_encoded', 0),
                row.get('customer_segment_encoded', 0)
            ]
            
            prediction = self.predict(features)
            prediction['transaction_id'] = row.get('transaction_id', idx)
            results.append(prediction)
        
        return pd.DataFrame(results)
    
    def evaluate_model(self, test_data: pd.DataFrame, true_labels: List[int]) -> Dict:
        """
        Evaluate model performance
        
        Args:
            test_data: Test transaction features
            true_labels: Actual fraud labels (0 or 1)
            
        Returns:
            Performance metrics
        """
        predictions = []
        
        for idx, row in test_data.iterrows():
            features = [
                row.get('transaction_amount', 0),
                row.get('account_balance', 0),
                row.get('transaction_time', 0),
                row.get('customer_age', 0),
                row.get('account_age_days', 0),
                row.get('daily_transaction_count', 0),
                row.get('avg_transaction_amount', 0),
                row.get('distance_from_home', 0),
                row.get('transaction_type_encoded', 0),
                row.get('merchant_category_encoded', 0),
                row.get('device_type_encoded', 0),
                row.get('customer_segment_encoded', 0)
            ]
            
            pred = self.predict(features)
            predictions.append(1 if pred['is_fraud'] else 0)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
        
        accuracy = accuracy_score(true_labels, predictions)
        precision = precision_score(true_labels, predictions)
        recall = recall_score(true_labels, predictions)
        f1 = f1_score(true_labels, predictions)
        cm = confusion_matrix(true_labels, predictions)
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm.tolist(),
            'total_predictions': len(predictions),
            'fraud_predictions': sum(predictions),
            'actual_fraud': sum(true_labels)
        }

def create_sample_transactions() -> pd.DataFrame:
    """
    Create sample transactions for testing
    """
    np.random.seed(42)
    
    # Generate legitimate transactions
    legitimate_data = []
    for i in range(100):
        legitimate_data.append({
            'transaction_id': f"LEGIT_{i:03d}",
            'transaction_amount': np.random.normal(100, 50),  # Normal amounts
            'account_balance': np.random.normal(5000, 2000),
            'transaction_time': np.random.randint(0, 86400),  # Time of day
            'customer_age': np.random.randint(18, 80),
            'account_age_days': np.random.randint(30, 3650),
            'daily_transaction_count': np.random.randint(1, 10),
            'avg_transaction_amount': np.random.normal(80, 30),
            'distance_from_home': np.random.exponential(5),  # Usually close
            'transaction_type_encoded': np.random.randint(0, 3),
            'merchant_category_encoded': np.random.randint(0, 5),
            'device_type_encoded': np.random.randint(0, 2),
            'customer_segment_encoded': np.random.randint(0, 2),
            'is_fraud': 0
        })
    
    # Generate fraudulent transactions
    fraud_data = []
    for i in range(20):
        fraud_data.append({
            'transaction_id': f"FRAUD_{i:03d}",
            'transaction_amount': np.random.normal(500, 300),  # Higher amounts
            'account_balance': np.random.normal(2000, 1500),  # Lower balance
            'transaction_time': np.random.randint(0, 86400),
            'customer_age': np.random.randint(18, 80),
            'account_age_days': np.random.randint(1, 365),  # Newer accounts
            'daily_transaction_count': np.random.randint(1, 20),  # More activity
            'avg_transaction_amount': np.random.normal(150, 100),
            'distance_from_home': np.random.exponential(50),  # Far from home
            'transaction_type_encoded': np.random.randint(0, 3),
            'merchant_category_encoded': np.random.randint(0, 5),
            'device_type_encoded': np.random.randint(0, 2),
            'customer_segment_encoded': np.random.randint(0, 2),
            'is_fraud': 1
        })
    
    # Combine and shuffle
    all_data = legitimate_data + fraud_data
    df = pd.DataFrame(all_data)
    return df.sample(frac=1).reset_index(drop=True)

def test_fraud_detection():
    """
    Test fraud detection with sample data
    """
    print("🔍 Testing Fraud Detection System")
    print("=" * 50)
    
    # Create sample data
    test_data = create_sample_transactions()
    
    # Create detector with dummy weights (simulating trained model)
    detector = FraudDetector()
    
    # Make predictions
    print("📊 Making predictions...")
    predictions_df = detector.batch_predict(test_data)
    
    # Evaluate performance
    true_labels = test_data['is_fraud'].tolist()
    test_features = test_data.drop('is_fraud', axis=1)
    
    metrics = detector.evaluate_model(test_features, true_labels)
    
    # Display results
    print("\n🎯 Model Performance:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-Score: {metrics['f1_score']:.2%}")
    
    print(f"\n📈 Prediction Summary:")
    print(f"Total transactions: {metrics['total_predictions']}")
    print(f"Predicted as fraud: {metrics['fraud_predictions']}")
    print(f"Actual fraud cases: {metrics['actual_fraud']}")
    
    print(f"\n🔢 Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"              Predicted")
    print(f"              Legitimate  Fraud")
    print(f"Actual Legitimate  {cm[0][0]:10d}  {cm[0][1]:6d}")
    print(f"       Fraud        {cm[1][0]:10d}  {cm[1][1]:6d}")
    
    # Show some sample predictions
    print(f"\n📋 Sample Predictions:")
    sample_predictions = predictions_df.head(10)
    for idx, row in sample_predictions.iterrows():
        status = "🚨" if row['is_fraud'] else "✅"
        print(f"{status} {row['transaction_id']}: {row['prediction']} ({row['fraud_probability']:.1%})")
    
    return metrics, predictions_df

if __name__ == "__main__":
    test_fraud_detection()
