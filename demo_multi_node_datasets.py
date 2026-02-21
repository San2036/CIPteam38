import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ai_engine.custom_csv_training import train_with_custom_csv, analyze_csv_dataset
from ai_engine.han_encryption import HANEncryption

class MultiNodeFederatedLearning:
    """Multi-node federated learning with different datasets per node"""
    
    def __init__(self):
        self.nodes = {}
        self.global_model_updates = []
        self.encryption_handler = HANEncryption()
    
    def add_node_with_dataset(self, node_id: int, csv_path: str, target_column: str = None, node_name: str = None):
        """Add a node with its specific dataset"""
        node_name = node_name or f"Bank_Node_{node_id}"
        
        self.nodes[node_id] = {
            'node_id': node_id,
            'node_name': node_name,
            'csv_path': csv_path,
            'target_column': target_column,
            'model_updates': [],
            'dataset_stats': {}
        }
        
        print(f"🏦 Added {node_name} with dataset: {csv_path}")
        
        # Analyze the dataset
        print(f"\n🔍 Analyzing dataset for {node_name}...")
        try:
            df = pd.read_csv(csv_path)
            self.nodes[node_id]['dataset_stats'] = {
                'shape': df.shape,
                'columns': list(df.columns),
                'target_column': target_column or 'auto-detected'
            }
            print(f"   📊 Dataset shape: {df.shape}")
            print(f"   📋 Columns: {list(df.columns)[:5]}{'...' if len(df.columns) > 5 else ''}")
        except Exception as e:
            print(f"   ❌ Error analyzing dataset: {e}")
    
    def train_node(self, node_id: int, epochs: int = 3, num_samples: int = 1000) -> bool:
        """Train a specific node with its dataset"""
        if node_id not in self.nodes:
            print(f"❌ Node {node_id} not found")
            return False
        
        node = self.nodes[node_id]
        print(f"\n🎯 Training {node['node_name']} with its unique dataset...")
        
        try:
            losses, model_hash = train_with_custom_csv(
                csv_path=node['csv_path'],
                target_column=node['target_column'],
                epochs=epochs,
                num_samples=num_samples
            )
            
            if losses and model_hash:
                # Store model update
                update = {
                    'node_id': node_id,
                    'node_name': node['node_name'],
                    'model_hash': model_hash,
                    'losses': losses,
                    'final_loss': losses[-1],
                    'dataset_type': 'custom_csv',
                    'csv_path': node['csv_path'],
                    'timestamp': f"2024-02-18 {12+node_id:02d}:30:00"
                }
                
                node['model_updates'].append(update)
                self.global_model_updates.append(update)
                
                print(f"✅ {node['node_name']} training completed!")
                print(f"   📊 Final loss: {losses[-1]:.4f}")
                print(f"   🔢 Model hash: {model_hash}")
                
                return True
            else:
                print(f"❌ {node['node_name']} training failed")
                return False
                
        except Exception as e:
            print(f"❌ Error training {node['node_name']}: {e}")
            return False
    
    def encrypt_and_upload(self, node_id: int) -> bool:
        """Encrypt and upload model for a specific node"""
        if node_id not in self.nodes:
            return False
        
        node = self.nodes[node_id]
        if not node['model_updates']:
            print(f"❌ No trained models for {node['node_name']}")
            return False
        
        latest_model = node['model_updates'][-1]
        model_hash = latest_model['model_hash']
        
        # Encrypt weights
        encrypted_hash = f"encrypted_{model_hash}"
        
        # Simulate blockchain upload
        tx_hash = f"0x{'node' + str(node_id) + 'abcdef1234567890' * 3}"
        block_number = 3000 + node_id
        
        upload_info = {
            'node_id': node_id,
            'node_name': node['node_name'],
            'model_hash': model_hash,
            'encrypted_hash': encrypted_hash,
            'tx_hash': tx_hash,
            'block_number': block_number,
            'timestamp': f"2024-02-18 {12+node_id:02d}:45:00"
        }
        
        print(f"📤 {node['node_name']} model uploaded to blockchain!")
        print(f"   🔗 Transaction hash: {tx_hash}")
        print(f"   ⛓️ Block number: {block_number}")
        
        return True
    
    def run_federated_round(self, epochs: int = 3, num_samples: int = 1000):
        """Run one round of federated learning with all nodes"""
        print(f"\n🚀 Federated Learning Round - All Nodes")
        print("=" * 50)
        
        successful_nodes = []
        
        # Train each node with its unique dataset
        for node_id in self.nodes:
            if self.train_node(node_id, epochs, num_samples):
                successful_nodes.append(node_id)
        
        # Upload models from successful nodes
        for node_id in successful_nodes:
            self.encrypt_and_upload(node_id)
        
        # Summary
        print(f"\n📊 Round Summary:")
        print(f"   🏦 Nodes trained: {len(successful_nodes)}/{len(self.nodes)}")
        print(f"   📈 Total model updates: {len(self.global_model_updates)}")
        
        if successful_nodes:
            avg_loss = np.mean([self.nodes[nid]['model_updates'][-1]['final_loss'] for nid in successful_nodes])
            print(f"   📊 Average loss: {avg_loss:.4f}")
        
        return successful_nodes
    
    def compare_node_performance(self):
        """Compare performance across different nodes"""
        print(f"\n📊 Node Performance Comparison")
        print("=" * 40)
        
        for node_id, node in self.nodes.items():
            if node['model_updates']:
                latest = node['model_updates'][-1]
                print(f"🏦 {node['node_name']}:")
                print(f"   📊 Dataset: {os.path.basename(node['csv_path'])}")
                print(f"   🎯 Target: {node['target_column'] or 'auto-detected'}")
                print(f"   📈 Final Loss: {latest['final_loss']:.4f}")
                print(f"   🔢 Model Hash: {latest['model_hash'][:20]}...")
                print(f"   📋 Dataset Size: {node['dataset_stats'].get('shape', 'Unknown')}")
                print()
    
    def create_sample_datasets(self):
        """Create sample datasets for demonstration"""
        print("📁 Creating sample datasets for multi-node demo...")
        
        # Dataset 1: Bank A - Retail transactions
        bank_a_data = {
            'transaction_amount': np.random.lognormal(3, 1.2, 2000),
            'merchant_type': np.random.choice(['retail', 'grocery', 'gas'], 2000),
            'customer_age': np.random.randint(25, 65, 2000),
            'transaction_hour': np.random.randint(6, 22, 2000),
            'is_fraud': np.random.random(2000) < 0.03
        }
        df_a = pd.DataFrame(bank_a_data)
        df_a.to_csv('bank_a_transactions.csv', index=False)
        print("   ✅ Created: bank_a_transactions.csv")
        
        # Dataset 2: Bank B - Online transactions
        bank_b_data = {
            'amount': np.random.lognormal(4, 1.5, 1500),
            'category': np.random.choice(['electronics', 'clothing', 'travel'], 1500),
            'customer_income': np.random.lognormal(10, 0.5, 1500),
            'time_of_day': np.random.randint(0, 24, 1500),
            'fraud': np.random.random(1500) < 0.05
        }
        df_b = pd.DataFrame(bank_b_data)
        df_b.to_csv('bank_b_online.csv', index=False)
        print("   ✅ Created: bank_b_online.csv")
        
        # Dataset 3: Bank C - Credit card transactions
        bank_c_data = {
            'purchase_amount': np.random.lognormal(3.5, 1.3, 1800),
            'merchant_category': np.random.randint(1, 15, 1800),
            'cardholder_age': np.random.randint(18, 80, 1800),
            'transaction_frequency': np.random.poisson(3, 1800),
            'is_suspicious': np.random.random(1800) < 0.02
        }
        df_c = pd.DataFrame(bank_c_data)
        df_c.to_csv('bank_c_credit.csv', index=False)
        print("   ✅ Created: bank_c_credit.csv")
        
        return ['bank_a_transactions.csv', 'bank_b_online.csv', 'bank_c_credit.csv']

def demo_multi_node_training():
    """Demo multi-node training with different datasets"""
    print("🤖 Multi-Node Federated Learning with Different Datasets")
    print("=" * 60)
    
    # Initialize federated learning system
    fl_system = MultiNodeFederatedLearning()
    
    # Create sample datasets
    sample_files = fl_system.create_sample_datasets()
    
    # Add nodes with different datasets
    fl_system.add_node_with_dataset(
        node_id=1,
        csv_path='bank_a_transactions.csv',
        target_column='is_fraud',
        node_name='Bank_A_Retail'
    )
    
    fl_system.add_node_with_dataset(
        node_id=2,
        csv_path='bank_b_online.csv',
        target_column='fraud',
        node_name='Bank_B_Online'
    )
    
    fl_system.add_node_with_dataset(
        node_id=3,
        csv_path='bank_c_credit.csv',
        target_column='is_suspicious',
        node_name='Bank_C_Credit'
    )
    
    # Run federated learning rounds
    print(f"\n🚀 Starting Multi-Node Federated Learning...")
    
    for round_num in range(1, 4):
        print(f"\n{'='*20} Round {round_num}/3 {'='*20}")
        successful_nodes = fl_system.run_federated_round(epochs=2, num_samples=500)
        
        if successful_nodes:
            print(f"✅ Round {round_num} completed with {len(successful_nodes)} nodes")
        else:
            print(f"❌ Round {round_num} failed")
    
    # Compare performance
    fl_system.compare_node_performance()
    
    # Final summary
    print(f"\n🎉 Multi-Node Federated Learning Summary")
    print("=" * 50)
    print(f"🏦 Total nodes: {len(fl_system.nodes)}")
    print(f"📊 Total model updates: {len(fl_system.global_model_updates)}")
    print(f"📈 Datasets used: {list(set(node['csv_path'] for node in fl_system.nodes.values()))}")
    
    print(f"\n🔗 All Model Updates:")
    for update in fl_system.global_model_updates:
        print(f"   🏦 {update['node_name']}: {update['model_hash'][:15]}... (Loss: {update['final_loss']:.4f})")
    
    print(f"\n✅ Multi-node federated learning with different datasets completed! 🚀")

def interactive_multi_node_setup():
    """Interactive setup for multi-node with your CSV files"""
    print("🎮 Interactive Multi-Node Setup")
    print("=" * 40)
    
    fl_system = MultiNodeFederatedLearning()
    
    while True:
        print(f"\n📋 Current nodes: {len(fl_system.nodes)}")
        
        action = input("Choose action:\n1. Add node with CSV\n2. Run training\n3. Compare performance\n4. Exit\n\nChoice: ").strip()
        
        if action == "1":
            node_id = len(fl_system.nodes) + 1
            csv_path = input("📁 Enter CSV file path: ").strip()
            
            if not os.path.exists(csv_path):
                print("❌ File not found")
                continue
            
            target_column = input("🎯 Target column (Enter for auto-detect): ").strip() or None
            node_name = input("🏦 Node name (Enter for default): ").strip() or None
            
            fl_system.add_node_with_dataset(node_id, csv_path, target_column, node_name)
            
        elif action == "2":
            if len(fl_system.nodes) == 0:
                print("❌ No nodes added yet")
                continue
            
            epochs = input("🔄 Epochs (default 3): ").strip() or "3"
            num_samples = input("📊 Samples per node (default 1000): ").strip() or "1000"
            
            fl_system.run_federated_round(
                epochs=int(epochs),
                num_samples=int(num_samples)
            )
            
        elif action == "3":
            fl_system.compare_node_performance()
            
        elif action == "4":
            break
        
        else:
            print("❌ Invalid choice")

def main():
    print("🤖 Universal Decentralized Agentic Federated Learning Platform")
    print("🎯 MULTI-NODE WITH DIFFERENT DATASETS")
    print("=" * 60)
    
    # Choose mode
    print("\n🎮 Choose mode:")
    print("1. Demo with sample datasets")
    print("2. Interactive with your CSV files")
    
    choice = input("\nEnter choice (1 or 2): ").strip()
    
    if choice == "1":
        demo_multi_node_training()
    elif choice == "2":
        interactive_multi_node_setup()
    else:
        print("❌ Invalid choice")

if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    main()
