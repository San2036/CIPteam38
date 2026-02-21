# Universal Decentralized Agentic Federated Learning Platform

A comprehensive federated learning platform that combines blockchain technology, differential privacy, homomorphic encryption, and intelligent agents for secure and decentralized machine learning.

## 🏗️ Architecture

The platform consists of 5 distinct layers:

### 1. Blockchain Layer
- **Smart Contract**: `FLRegistry.sol` - Manages node registration, model updates, and reputation system
- **Features**: Node slashing, global model hash storage, event emission for monitoring

### 2. AI Engine Layer
- **Model**: `SimpleNN` - PyTorch neural network for classification tasks
- **Training**: Differential privacy simulation with Gaussian noise
- **Encryption**: Homomorphic encryption simulation using matrix operations

### 3. Security Layer
- **Trust Agent**: Monitors blockchain events, validates proofs, and slashes malicious nodes
- **Healer Agent**: Detects anomalous models and performs healing operations

### 4. Agentic Layer
- **Trust Agent**: Real-time monitoring and validation of federated learning participants
- **Healer Agent**: Model quality assessment and automatic recovery mechanisms

### 5. Dashboard Layer
- **Streamlit App**: Real-time monitoring, training visualization, and system management

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Ganache (local blockchain)
- Node.js (for blockchain development tools)

### Installation

1. **Clone the repository and install dependencies:**
```bash
cd Decentralized_AI_Platform
pip install -r requirements.txt
```

2. **Start Ganache:**
- Open Ganache desktop application
- Create a new workspace
- Note the RPC URL (usually `http://127.0.0.1:7545`)

3. **Run the main node:**
```bash
python main_node.py
```

4. **Launch the dashboard (optional):**
```bash
streamlit run dashboard/app.py
```

## 📁 Project Structure

```
Decentralized_AI_Platform/
├── blockchain/
│   └── FLRegistry.sol          # Smart contract for federated learning registry
├── ai_engine/
│   ├── model.py               # PyTorch neural network model
│   ├── train.py               # Training logic with differential privacy
│   └── han_encryption.py      # Homomorphic encryption simulation
├── agents/
│   ├── trust_agent.py         # Blockchain monitoring and validation
│   └── healer_agent.py        # Model quality assessment and healing
├── dashboard/
│   └── app.py                 # Streamlit dashboard
├── main_node.py               # Entry point for bank nodes
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## 🔧 Usage

### Running a Single Bank Node
```bash
# Default node ID 1
python main_node.py

# Specific node ID
python main_node.py 2
```

### Running Multiple Nodes
Open multiple terminal windows and run:
```bash
# Terminal 1
python main_node.py 1

# Terminal 2
python main_node.py 2

# Terminal 3
python main_node.py 3
```

### Trust Agent Monitoring
```bash
python agents/trust_agent.py
```

### Dashboard Access
1. Start the dashboard: `streamlit run dashboard/app.py`
2. Open browser to `http://localhost:8501`
3. Connect to Ganache and monitor the system

## 🛡️ Security Features

### Differential Privacy
- Gaussian noise added to model weights during training
- Configurable noise scale for privacy-utility trade-off

### Homomorphic Encryption
- Matrix-based encryption simulation
- Secure weight aggregation without decryption

### Blockchain Security
- Immutable audit trail of model updates
- Smart contract-based reputation system
- Automatic slashing of malicious nodes

### Agent-Based Monitoring
- Real-time proof validation
- Anomaly detection in model updates
- Automatic recovery mechanisms

## 📊 Key Components

### FLRegistry Smart Contract
```solidity
struct Node {
    uint256 id;
    uint256 reputation;
    bool isBanned;
    address nodeAddress;
}
```

### SimpleNN Model
```python
class SimpleNN(nn.Module):
    def __init__(self, input_size=784, hidden_size=128, num_classes=10):
        # 3-layer neural network for classification
```

### Trust Agent
- Monitors `ModelUpdated` events
- Validates cryptographic proofs
- Executes slashing for malicious behavior

### Healer Agent
- Detects anomalous model updates
- Performs statistical analysis
- Generates healed model weights

## 🔍 Monitoring and Visualization

The Streamlit dashboard provides:
- Real-time training metrics
- Blockchain transaction monitoring
- Security agent status
- Node health visualization
- Encryption demonstrations

## 🧪 Testing

### Run Individual Components
```bash
# Test AI engine
python ai_engine/train.py

# Test encryption
python ai_engine/han_encryption.py

# Test healer agent
python agents/healer_agent.py
```

## 📈 Performance Metrics

The platform tracks:
- Training loss over time
- Model accuracy
- Node reputation scores
- Security incidents
- Blockchain gas usage

## 🔄 Workflow

1. **Node Registration**: Banks register on the blockchain
2. **Local Training**: Each node trains on private data
3. **Encryption**: Model weights are encrypted
4. **Blockchain Upload**: Encrypted weights uploaded with proofs
5. **Validation**: Trust agent validates proofs
6. **Aggregation**: Global model updated
7. **Monitoring**: Continuous security monitoring

## 🛠️ Configuration

### Blockchain Settings
- RPC URL: `http://127.0.0.1:7545` (Ganache default)
- Gas limit: Configurable per transaction
- Contract address: Auto-deployed on first run

### Training Parameters
- Epochs: 5 (configurable)
- Batch size: 32
- Learning rate: 0.001
- DP noise scale: 0.01

### Security Thresholds
- Proof validation: `"valid_token"` (mock)
- Anomaly detection: Z-score > 2.0
- Reputation penalty: -10 points
- Reputation reward: +5 points

## 🚨 Troubleshooting

### Common Issues

1. **Blockchain Connection Failed**
   - Ensure Ganache is running
   - Check RPC URL in configuration
   - Verify network connectivity

2. **Contract Deployment Failed**
   - Check gas limit settings
   - Ensure sufficient ETH in account
   - Verify contract bytecode

3. **Training Errors**
   - Check PyTorch installation
   - Verify data loader configuration
   - Ensure sufficient memory

4. **Dashboard Connection Issues**
   - Check Streamlit version
   - Verify all dependencies installed
   - Check browser console for errors

## 📝 License

This project is licensed under the MIT License.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For issues and questions:
- Check the troubleshooting section
- Review the code documentation
- Open an issue on the repository

---

**Universal Decentralized Agentic Federated Learning Platform** - Secure, Private, and Intelligent Federated Learning
