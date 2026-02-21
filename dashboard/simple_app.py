import streamlit as st

import sys

import os

import pandas as pd

import numpy as np

import plotly.express as px

from datetime import datetime



st.set_page_config(

    page_title="Decentralized AI Platform",

    page_icon="🤖",

    layout="wide",

    initial_sidebar_state="expanded"

)



st.title("🤖 Universal Decentralized Agentic Federated Learning Platform")

st.markdown("---")



def display_overview():

    col1, col2, col3, col4 = st.columns(4)

    

    with col1:

        st.metric("🏦 Active Nodes", "3", delta="↑ 2 from last round")

    

    with col2:

        st.metric("🤖 Global Model Version", "v1.0", delta="New update")

    

    with col3:

        st.metric("🔒 Security Score", "95%", delta="↑ 5%")

    

    with col4:

        st.metric("⚡ Training Rounds", "5", delta="↑ 1")



def sidebar_connection():

    st.sidebar.header("🔗 Connection Settings")

    

    blockchain_url = st.sidebar.text_input(

        "Blockchain URL",

        value="http://127.0.0.1:7545",

        help="Ganache blockchain RPC URL"

    )

    

    contract_address = st.sidebar.text_input(

        "Contract Address",

        value="",

        help="FLRegistry smart contract address"

    )

    

    if st.sidebar.button("🔌 Connect", type="primary"):

        st.session_state.connected = True

        st.session_state.contract_address = contract_address

        st.sidebar.success("✅ Connected to Ganache")

    

    if st.session_state.connected:

        st.sidebar.success("✅ Connected")

    else:

        st.sidebar.warning("⚠️ Not Connected")



def display_training_status():

    st.header("🎯 Training Status")

    

    col1, col2 = st.columns([2, 1])

    

    with col1:

        if st.button("🚀 Start Training Round", type="primary"):

            with st.spinner("Training model..."):

                # Simulate training

                import time

                time.sleep(2)

                

                # Generate fake training metrics

                loss = np.random.uniform(0.1, 0.5)

                accuracy = np.random.uniform(0.85, 0.95)

                

                st.success(f"Training completed! Loss: {loss:.4f}, Accuracy: {accuracy:.2%}")

                

                # Store in session state

                if 'training_history' not in st.session_state:

                    st.session_state.training_history = []

                

                st.session_state.training_history.append({

                    'timestamp': datetime.now(),

                    'loss': loss,

                    'accuracy': accuracy

                })

        

        if 'training_history' in st.session_state and st.session_state.training_history:

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

        

        if 'training_history' in st.session_state and st.session_state.training_history:

            latest = st.session_state.training_history[-1]

            st.metric("Latest Loss", f"{latest['loss']:.4f}")

            st.metric("Latest Accuracy", f"{latest['accuracy']:.2%}")

            st.metric("Training Rounds", len(st.session_state.training_history))

        else:

            st.info("No training history yet")



def display_blockchain_info():

    st.header("⛓️ Blockchain Information")

    

    col1, col2 = st.columns(2)

    

    with col1:

        st.subheader("📋 Smart Contract")

        if st.session_state.get('contract_address'):

            st.code(st.session_state.contract_address, language="text")

        else:

            st.warning("No contract deployed")

        

        st.subheader("🔗 Network Status")

        if st.session_state.get('connected'):

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



def display_fraud_detection():

    st.header("🔍 Fraud Detection Testing")

    

    col1, col2 = st.columns([2, 1])

    

    with col1:

        st.subheader("🧪 Test Model Performance")

        

        if st.button("🚀 Run Fraud Detection Test", type="primary"):

            with st.spinner("Testing fraud detection..."):

                try:

                    # Import and run fraud detection test

                    import sys

                    import os

                    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

                    

                    from fraud_detector import test_fraud_detection

                    metrics, predictions_df = test_fraud_detection()

                    

                    # Store results in session state

                    st.session_state.fraud_metrics = metrics

                    st.session_state.fraud_predictions = predictions_df

                    

                    st.success("Fraud detection test completed!")

                    

                except Exception as e:

                    st.error(f"Error running fraud detection: {e}")

        

        if 'fraud_metrics' in st.session_state:

            metrics = st.session_state.fraud_metrics

            

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

            cm = metrics['confusion_matrix']

            

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

            

            # Summary Statistics

            st.subheader("📈 Prediction Summary")

            col_e, col_f, col_g = st.columns(3)

            

            with col_e:

                st.metric("Total Transactions", metrics['total_predictions'])

            with col_f:

                st.metric("Predicted Fraud", metrics['fraud_predictions'])

            with col_g:

                st.metric("Actual Fraud", metrics['actual_fraud'])

    

    with col2:

        st.subheader("📋 Sample Predictions")

        

        if 'fraud_predictions' in st.session_state:

            predictions_df = st.session_state.fraud_predictions

            

            # Show sample predictions

            sample_size = min(10, len(predictions_df))

            sample_df = predictions_df.head(sample_size)

            

            for idx, row in sample_df.iterrows():

                if row.get('error'):

                    st.error(f"❌ {row['transaction_id']}: {row['error']}")

                else:

                    if row['is_fraud']:

                        st.warning(f"🚨 {row['transaction_id']}: FRAUD ({row['fraud_probability']:.1%})")

                    else:

                        st.success(f"✅ {row['transaction_id']}: LEGITIMATE ({row['fraud_probability']:.1%})")

        

        st.subheader("🎯 How to Interpret Results")

        

        st.info("""

        **Fraud Detection Guide:**

        

        🚨 **FRAUD** - High probability transaction, investigate immediately

        ✅ **LEGITIMATE** - Low probability, normal processing

        📊 **Accuracy** - Overall correct prediction rate

        🎯 **Precision** - Fraud predictions that are actually fraud

        🔍 **Recall** - Actual fraud cases detected

        """)

        

        st.subheader("🔧 Model Improvement Tips")

        

        tips = [

            "📈 Add more training data for better patterns",

            "🏷️ Include more transaction features",

            "⚖️ Balance legitimate vs fraud samples",

            "🔄 Regular model retraining",

            "🛡️ Add anomaly detection layers"

        ]

        

        for tip in tips:

            st.write(tip)



def init_session_state():

    if 'connected' not in st.session_state:

        st.session_state.connected = False

    if 'contract_address' not in st.session_state:

        st.session_state.contract_address = ""

    if 'node_id' not in st.session_state:

        st.session_state.node_id = 1



def main():

    init_session_state()

    

    sidebar_connection()

    

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

    

    st.markdown("---")

    st.markdown(

        "<div style='text-align: center; color: gray;'>"

        "Universal Decentralized Agentic Federated Learning Platform © 2026"

        "</div>",

        unsafe_allow_html=True

    )



if __name__ == "__main__":

    main()

