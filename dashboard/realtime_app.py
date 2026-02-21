import streamlit as st
import sys
import os
import time
import threading
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="Real-time Federated Learning Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Real-time Universal Decentralized Agentic Federated Learning Platform")

# Initialize session state safely
def init_session_state():
    if 'system_state' not in st.session_state:
        st.session_state.system_state = {
            'active_nodes': 0,
            'training_rounds': 0,
            'global_model_version': 'v1.0',
            'security_score': 95,
            'training_loss': [],
            'recent_transactions': [],
            'node_health': {},
            'model_updates': [],
            'trust_agent_active': False,
            'healer_agent_active': False,
            'fraud_metrics': None
        }
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False

# Initialize on first load
init_session_state()

def simulate_real_time_data():
    """Simulate real-time data from the running system"""
    while st.session_state.is_running:
        time.sleep(2)
        
        # Simulate nodes joining
        if st.session_state.system_state['active_nodes'] < 3:
            st.session_state.system_state['active_nodes'] += 1
            st.session_state.system_state['node_health'][f'Node_{st.session_state.system_state["active_nodes"]}'] = np.random.randint(85, 100)
        
        # Simulate training rounds
        if np.random.random() > 0.7:
            st.session_state.system_state['training_rounds'] += 1
            loss = np.random.uniform(1.0, 2.5)
            st.session_state.system_state['training_loss'].append(loss)
            
            # Keep only last 20 loss values
            if len(st.session_state.system_state['training_loss']) > 20:
                st.session_state.system_state['training_loss'] = st.session_state.system_state['training_loss'][-20:]
        
        # Simulate transactions
        if np.random.random() > 0.8:
            transaction = {
                'hash': f"0x{'1234567890abcdef' * 4}",
                'type': 'Model Update',
                'node': f"Node_{np.random.randint(1, 4)}",
                'time': datetime.now().strftime('%H:%M:%S')
            }
            st.session_state.system_state['recent_transactions'].insert(0, transaction)
            
            # Keep only last 10 transactions
            if len(st.session_state.system_state['recent_transactions']) > 10:
                st.session_state.system_state['recent_transactions'] = st.session_state.system_state['recent_transactions'][:10]
        
        # Simulate model updates
        if np.random.random() > 0.9:
            model_update = {
                'node': f"Node_{np.random.randint(1, 4)}",
                'hash': f"model_{int(time.time())}",
                'timestamp': datetime.now().strftime('%H:%M:%S'),
                'loss': np.random.uniform(1.0, 2.5)
            }
            st.session_state.system_state['model_updates'].insert(0, model_update)
            
            # Keep only last 10 updates
            if len(st.session_state.system_state['model_updates']) > 10:
                st.session_state.system_state['model_updates'] = st.session_state.system_state['model_updates'][:10]
        
        # Update global model version
        if st.session_state.system_state['training_rounds'] % 5 == 0 and st.session_state.system_state['training_rounds'] > 0:
            version_num = st.session_state.system_state['training_rounds'] // 5 + 1
            st.session_state.system_state['global_model_version'] = f'v{version_num}.0'
        
        # Update security score
        st.session_state.system_state['security_score'] = min(100, 85 + st.session_state.system_state['active_nodes'] * 5)
        
        # Simulate agent activity
        st.session_state.system_state['trust_agent_active'] = np.random.random() > 0.3
        st.session_state.system_state['healer_agent_active'] = np.random.random() > 0.4

def start_real_time_simulation():
    """Start real-time simulation in background"""
    if not st.session_state.is_running:
        st.session_state.is_running = True
        thread = threading.Thread(target=simulate_real_time_data, daemon=True)
        thread.start()

def stop_real_time_simulation():
    """Stop real-time simulation"""
    st.session_state.is_running = False

def display_overview():
    st.header("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "🏦 Active Nodes", 
            st.session_state.system_state['active_nodes'],
            delta=None
        )
    
    with col2:
        st.metric(
            "🤖 Global Model", 
            st.session_state.system_state['global_model_version'],
            delta="Updated" if st.session_state.system_state['training_rounds'] > 0 else None
        )
    
    with col3:
        st.metric(
            "📈 Training Rounds", 
            st.session_state.system_state['training_rounds'],
            delta=None
        )
    
    with col4:
        st.metric(
            "🔒 Security Score", 
            f"{st.session_state.system_state['security_score']}%",
            delta=None
        )

def display_training_status():
    st.header("🎯 Training Status")
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("📈 Training Progress")
        
        if st.session_state.system_state['training_loss']:
            df_loss = pd.DataFrame({
                'Round': range(1, len(st.session_state.system_state['training_loss']) + 1),
                'Loss': st.session_state.system_state['training_loss']
            })
            
            fig = px.line(
                df_loss, 
                x='Round', 
                y='Loss',
                title='Training Loss Over Time',
                labels={'value': 'Loss'}
            )
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("🔄 Waiting for training data...")
    
    with col2:
        st.subheader("📊 Current Metrics")
        
        if st.session_state.system_state['training_loss']:
            latest_loss = st.session_state.system_state['training_loss'][-1]
            avg_loss = np.mean(st.session_state.system_state['training_loss'])
            
            st.metric("Latest Loss", f"{latest_loss:.3f}")
            st.metric("Average Loss", f"{avg_loss:.3f}")
            
            if latest_loss < avg_loss:
                st.success("📉 Model improving!")
            else:
                st.warning("📈 Model needs more training")
        else:
            st.info("⏳ No training data yet")
    
    # Training controls
    st.subheader("🎮 Training Controls")
    col_a, col_b, col_c = st.columns(3)
    
    with col_a:
        if st.button("🚀 Start Training", type="primary"):
            start_real_time_simulation()
            st.success("🎯 Real-time simulation started!")
    
    with col_b:
        if st.button("⏸️ Pause Training"):
            stop_real_time_simulation()
            st.warning("⏸️ Simulation paused")
    
    with col_c:
        if st.button("🔄 Reset Data"):
            st.session_state.system_state['training_loss'] = []
            st.session_state.system_state['training_rounds'] = 0
            st.rerun()

def display_blockchain_info():
    st.header("⛓️ Blockchain Activity")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📋 Recent Transactions")
        
        if st.session_state.system_state['recent_transactions']:
            df_tx = pd.DataFrame(st.session_state.system_state['recent_transactions'])
            st.dataframe(df_tx, use_container_width=True)
        else:
            st.info("⏳ Waiting for transactions...")
    
    with col2:
        st.subheader("📊 Transaction Stats")
        
        if st.session_state.system_state['recent_transactions']:
            total_tx = len(st.session_state.system_state['recent_transactions'])
            model_updates = len([tx for tx in st.session_state.system_state['recent_transactions'] if tx['type'] == 'Model Update'])
            
            st.metric("Total Transactions", total_tx)
            st.metric("Model Updates", model_updates)
            st.metric("Update Rate", f"{model_updates/total_tx*100:.1f}%")
        else:
            st.info("📊 No transaction data yet")

def display_security_monitoring():
    st.header("🛡️ Security Monitoring")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🤖 Agent Status")
        
        # Trust Agent Status
        trust_status = "🟢 Active" if st.session_state.system_state['trust_agent_active'] else "🔴 Inactive"
        st.metric("🛡️ Trust Agent", trust_status)
        
        # Healer Agent Status
        healer_status = "🟢 Active" if st.session_state.system_state['healer_agent_active'] else "🔴 Inactive"
        st.metric("🏥 Healer Agent", healer_status)
        
        # Recent Alerts
        st.subheader("🚨 Recent Alerts")
        alerts = []
        
        if st.session_state.system_state['trust_agent_active']:
            alerts.append("🛡️ Trust Agent monitoring events")
        
        if st.session_state.system_state['healer_agent_active']:
            alerts.append("🏥 Healer Agent analyzing models")
        
        if st.session_state.system_state['security_score'] < 90:
            alerts.append("⚠️ Security score below optimal")
        
        if alerts:
            for alert in alerts:
                st.info(alert)
        else:
            st.success("✅ All systems operating normally")
    
    with col2:
        st.subheader("📊 Node Health")
        
        if st.session_state.system_state['node_health']:
            health_data = []
            for node, health in st.session_state.system_state['node_health'].items():
                health_data.append({
                    'Node': node,
                    'Health': health,
                    'Status': 'Good' if health > 90 else 'Fair' if health > 70 else 'Poor'
                })
            
            df_health = pd.DataFrame(health_data)
            
            fig = px.bar(
                df_health, 
                x='Node', 
                y='Health',
                color='Status',
                title='Node Health Scores',
                color_discrete_map={'Good': 'green', 'Fair': 'orange', 'Poor': 'red'}
            )
            fig.update_layout(height=250)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("📊 No node health data yet")

def display_fraud_detection():
    st.header("🔍 Fraud Detection Testing")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🧪 Test Model Performance")
        
        if st.button("🚀 Run Fraud Detection Test", type="primary"):
            with st.spinner("Testing fraud detection..."):
                try:
                    # Simulate fraud detection test
                    accuracy = np.random.uniform(0.85, 0.95)
                    precision = np.random.uniform(0.80, 0.90)
                    recall = np.random.uniform(0.75, 0.85)
                    f1_score = np.random.uniform(0.80, 0.90)
                    
                    # Store metrics
                    st.session_state.system_state['fraud_metrics'] = {
                        'accuracy': accuracy,
                        'precision': precision,
                        'recall': recall,
                        'f1_score': f1_score
                    }
                    
                    st.success("🎉 Fraud detection test completed!")
                    
                except Exception as e:
                    st.error(f"❌ Error running test: {e}")
        
        if st.session_state.system_state['fraud_metrics']:
            metrics = st.session_state.system_state['fraud_metrics']
            
            st.subheader("📊 Model Performance Metrics")
            
            col_a, col_b, col_c, col_d = st.columns(4)
            
            with col_a:
                st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            with col_b:
                st.metric("Precision", f"{metrics['precision']:.2%}")
            with col_c:
                st.metric("Recall", f"{metrics['recall']:.2%}")
            with col_d:
                st.metric("F1-Score", f"{metrics['f1_score']:.2%}")
            
            # Confusion Matrix
            st.subheader("🔢 Confusion Matrix")
            
            # Generate sample confusion matrix
            cm = np.array([
                [np.random.randint(80, 100), np.random.randint(0, 20)],
                [np.random.randint(0, 20), np.random.randint(60, 90)]
            ])
            
            fig_cm = go.Figure(data=go.Heatmap(
                z=cm,
                x=['Legitimate', 'Fraud'],
                y=['Legitimate', 'Fraud'],
                colorscale='Blues',
                text=cm,
                texttemplate="%{text}"
            ))
            
            fig_cm.update_layout(
                title='Confusion Matrix',
                xaxis_title='Predicted',
                yaxis_title='Actual'
            )
            
            st.plotly_chart(fig_cm, use_container_width=True)
        else:
            st.info("🧪 Click 'Run Fraud Detection Test' to see performance metrics")
    
    with col2:
        st.subheader("📋 Sample Predictions")
        
        if st.session_state.system_state['fraud_metrics']:
            # Generate sample predictions
            predictions = []
            for i in range(10):
                is_fraud = np.random.random() > 0.8
                prob = np.random.uniform(0.1, 0.9)
                
                predictions.append({
                    'Transaction': f"TX_{1000 + i}",
                    'Prediction': 'FRAUD' if is_fraud else 'LEGITIMATE',
                    'Probability': f"{prob:.1%}",
                    'Status': '🚨' if is_fraud else '✅'
                })
            
            df_pred = pd.DataFrame(predictions)
            st.dataframe(df_pred, use_container_width=True)
        else:
            st.info("📋 Run fraud detection test to see sample predictions")

def sidebar_connection():
    st.sidebar.header("🔗 Connection Settings")
    
    # Connection status
    connection_status = "🟢 Connected" if st.session_state.is_running else "🔴 Disconnected"
    st.sidebar.metric("🌐 Status", connection_status)
    
    st.sidebar.text("Blockchain URL:")
    st.sidebar.code("http://127.0.0.1:7545")
    
    st.sidebar.text("Contract Address:")
    st.sidebar.code("0x1234567890abcdef1234567890abcdef1234567890")
    
    st.sidebar.text("Node ID:")
    st.sidebar.code("1")
    
    # System controls
    st.sidebar.subheader("🎮 System Controls")
    
    if st.sidebar.button("🔄 Refresh Dashboard"):
        st.rerun()
    
    if st.sidebar.button("📊 Export Data"):
        # Export current system state
        data = {
            'timestamp': datetime.now().isoformat(),
            'system_state': st.session_state.system_state
        }
        st.sidebar.download_button(
            "Download System Data",
            data=str(data),
            file_name=f"system_state_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )

def main():
    # Initialize session state
    if 'system_state' not in st.session_state:
        st.session_state.system_state = {
            'active_nodes': 0,
            'training_rounds': 0,
            'global_model_version': 'v1.0',
            'security_score': 95,
            'training_loss': [],
            'recent_transactions': [],
            'node_health': {},
            'model_updates': [],
            'trust_agent_active': False,
            'healer_agent_active': False,
            'fraud_metrics': None
        }
    
    if 'is_running' not in st.session_state:
        st.session_state.is_running = False
    
    # Sidebar
    sidebar_connection()
    
    # Main content
    display_overview()
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Training", 
        "⛓️ Blockchain", 
        "🛡️ Security",
        "🔍 Fraud Detection"
    ])
    
    with tab1:
        display_training_status()
    
    with tab2:
        display_blockchain_info()
    
    with tab3:
        display_security_monitoring()
    
    with tab4:
        display_fraud_detection()
    
    # Auto-refresh
    if st.session_state.is_running:
        time.sleep(2)
        st.rerun()

if __name__ == "__main__":
    main()
