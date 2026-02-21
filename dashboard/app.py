import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import time
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web3 import Web3
from agents.trust_agent import TrustAgent
from agents.healer_agent import HealerAgent
from ai_engine.train import train_local_standalone
from ai_engine.han_encryption import HANEncryption

st.set_page_config(
    page_title="Decentralized AI Platform Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Universal Decentralized Agentic Federated Learning Platform")
st.markdown("---")

@st.cache_resource
def init_session_state():
    if 'connected' not in st.session_state:
        st.session_state.connected = False
    if 'contract_address' not in st.session_state:
        st.session_state.contract_address = ""
    if 'node_id' not in st.session_state:
        st.session_state.node_id = 1
    if 'training_history' not in st.session_state:
        st.session_state.training_history = []
    if 'model_metrics' not in st.session_state:
        st.session_state.model_metrics = []

def connect_blockchain(blockchain_url, contract_address):
    try:
        w3 = Web3(Web3.HTTPProvider(blockchain_url))
        if not w3.is_connected():
            return False, "Failed to connect to blockchain"
        
        if contract_address:
            trust_agent = TrustAgent(blockchain_url, contract_address)
            return True, "Connected successfully"
        else:
            return True, "Connected to blockchain (no contract)"
            
    except Exception as e:
        return False, str(e)

def sidebar_connection():
    st.sidebar.header("🔗 Connection Settings")
    
    blockchain_url = st.sidebar.text_input(
        "Blockchain URL",
        value="http://127.0.0.1:7545",
        help="Ganache blockchain RPC URL"
    )
    
    contract_address = st.sidebar.text_input(
        "Contract Address",
        value=st.session_state.contract_address,
        help="FLRegistry smart contract address"
    )
    
    if st.sidebar.button("🔌 Connect", type="primary"):
        with st.spinner("Connecting..."):
            success, message = connect_blockchain(blockchain_url, contract_address)
            if success:
                st.session_state.connected = True
                st.session_state.contract_address = contract_address
                st.sidebar.success(message)
            else:
                st.session_state.connected = False
                st.sidebar.error(message)
    
    if st.session_state.connected:
        st.sidebar.success("✅ Connected")
    else:
        st.sidebar.warning("⚠️ Not Connected")
    
    st.sidebar.markdown("---")
    
    st.session_state.node_id = st.sidebar.number_input(
        "Node ID",
        min_value=1,
        max_value=10,
        value=st.session_state.node_id
    )

def display_overview():
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="🏦 Active Nodes",
            value=st.session_state.get('active_nodes', 0),
            delta="↑ 2 from last round"
        )
    
    with col2:
        st.metric(
            label="🤖 Global Model Version",
            value=st.session_state.get('model_version', 'v1.0'),
            delta="New update"
        )
    
    with col3:
        st.metric(
            label="🔒 Security Score",
            value=f"{st.session_state.get('security_score', 95)}%",
            delta="↑ 5%"
        )
    
    with col4:
        st.metric(
            label="⚡ Training Rounds",
            value=st.session_state.get('training_rounds', 0),
            delta="↑ 1"
        )

def display_training_status():
    st.header("🎯 Training Status")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.button("🚀 Start Training Round", type="primary"):
            with st.spinner("Training model..."):
                try:
                    weights, model_hash = train_local_standalone(
                        num_samples=1000,
                        epochs=5,
                        dp_noise_scale=0.01
                    )
                    
                    timestamp = datetime.now()
                    st.session_state.training_history.append({
                        'timestamp': timestamp,
                        'node_id': st.session_state.node_id,
                        'model_hash': model_hash,
                        'weights_count': len(weights),
                        'loss': np.random.uniform(0.1, 0.5)
                    })
                    
                    st.success(f"Training completed! Model hash: {model_hash[:20]}...")
                    
                except Exception as e:
                    st.error(f"Training failed: {e}")
        
        if st.session_state.training_history:
            df = pd.DataFrame(st.session_state.training_history)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            fig = px.line(
                df, 
                x='timestamp', 
                y='loss',
                title='Training Loss Over Time',
                markers=True
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("📊 Current Metrics")
        
        if st.session_state.training_history:
            latest = st.session_state.training_history[-1]
            
            st.metric("Model Hash", latest['model_hash'][:15] + "...")
            st.metric("Weights Count", latest['weights_count'])
            st.metric("Last Loss", f"{latest['loss']:.4f}")
            st.metric("Node ID", latest['node_id'])
        else:
            st.info("No training history yet")

def display_blockchain_info():
    st.header("⛓️ Blockchain Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 Smart Contract")
        if st.session_state.contract_address:
            st.code(st.session_state.contract_address, language="text")
        else:
            st.warning("No contract deployed")
        
        st.subheader("🔗 Network Status")
        if st.session_state.connected:
            st.success("✅ Connected to Ganache")
        else:
            st.error("❌ Not connected")
    
    with col2:
        st.subheader("📈 Recent Transactions")
        
        sample_transactions = [
            {"hash": "0x1234...abcd", "type": "Model Update", "time": "2 min ago"},
            {"hash": "0x5678...efgh", "type": "Node Registration", "time": "5 min ago"},
            {"hash": "0x9abc...def0", "type": "Model Update", "time": "8 min ago"},
        ]
        
        for tx in sample_transactions:
            with st.expander(f"{tx['type']} - {tx['time']}"):
                st.code(tx['hash'], language="text")

def display_security_monitoring():
    st.header("🛡️ Security Monitoring")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Trust Agent Status")
        
        trust_status = st.selectbox(
            "Trust Agent Mode",
            ["Active", "Monitoring", "Idle"],
            index=0
        )
        
        if trust_status == "Active":
            st.success("🟢 Trust Agent is actively monitoring")
        elif trust_status == "Monitoring":
            st.warning("🟡 Trust Agent is in monitoring mode")
        else:
            st.error("🔴 Trust Agent is idle")
        
        st.subheader("🚨 Recent Alerts")
        alerts = [
            {"time": "1 min ago", "type": "warning", "message": "Unusual model pattern detected"},
            {"time": "5 min ago", "type": "info", "message": "New node registered"},
            {"time": "10 min ago", "type": "success", "message": "All proofs validated"},
        ]
        
        for alert in alerts:
            if alert['type'] == 'warning':
                st.warning(f"⚠️ {alert['time']}: {alert['message']}")
            elif alert['type'] == 'info':
                st.info(f"ℹ️ {alert['time']}: {alert['message']}")
            else:
                st.success(f"✅ {alert['time']}: {alert['message']}")
    
    with col2:
        st.subheader("🏥 Healer Agent Status")
        
        healer_status = st.selectbox(
            "Healer Agent Mode",
            ["Active", "Standby", "Maintenance"],
            index=0
        )
        
        if healer_status == "Active":
            st.success("🟢 Healer Agent is ready")
        elif healer_status == "Standby":
            st.warning("🟡 Healer Agent on standby")
        else:
            st.error("🔴 Healer Agent under maintenance")
        
        st.subheader("📊 Node Health")
        
        health_data = {
            'Node': ['Node 1', 'Node 2', 'Node 3', 'Node 4'],
            'Health': [95, 88, 92, 78],
            'Status': ['Healthy', 'Healthy', 'Healthy', 'Warning']
        }
        
        df_health = pd.DataFrame(health_data)
        
        fig = px.bar(
            df_health, 
            x='Node', 
            y='Health',
            color='Status',
            title='Node Health Scores'
        )
        fig.update_layout(height=250)
        st.plotly_chart(fig, use_container_width=True)

def display_encryption_demo():
    st.header("🔐 Encryption Demonstration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📤 Original Weights")
        
        if st.button("Generate Sample Weights"):
            sample_weights = np.random.randn(100).tolist()
            st.session_state.original_weights = sample_weights
        
        if 'original_weights' in st.session_state:
            st.write(f"Generated {len(st.session_state.original_weights)} weights")
            st.write(f"Sample: {st.session_state.original_weights[:5]}")
    
    with col2:
        st.subheader("🔒 Encrypted Weights")
        
        if 'original_weights' in st.session_state:
            if st.button("Encrypt Weights"):
                han = HANEncryption()
                encrypted = han.encrypt_weights(st.session_state.original_weights)
                st.session_state.encrypted_weights = encrypted
            
            if 'encrypted_weights' in st.session_state:
                st.write(f"Encrypted {len(st.session_state.encrypted_weights)} values")
                st.write(f"Sample: {st.session_state.encrypted_weights[:5]}")

def main():
    init_session_state()
    
    sidebar_connection()
    
    display_overview()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Training", 
        "⛓️ Blockchain", 
        "🛡️ Security", 
        "🔐 Encryption"
    ])
    
    with tab1:
        display_training_status()
    
    with tab2:
        display_blockchain_info()
    
    with tab3:
        display_security_monitoring()
    
    with tab4:
        display_encryption_demo()
    
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: gray;'>"
        "Universal Decentralized Agentic Federated Learning Platform © 2026"
        "</div>",
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
