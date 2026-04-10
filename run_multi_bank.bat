@echo off
title Multi-Bank Federated Learning Platform (3 Banks)
color 0A

echo ╔══════════════════════════════════════════════════════════╗
echo ║   Multi-Bank Federated Learning Platform                 ║
echo ║   3 Independent Banks — Real Datasets — Privacy Preserved║
echo ╚══════════════════════════════════════════════════════════╝
echo.
echo ─────────────────────────────────────────────────────────────
echo  STEP 1: Generating adapted datasets (safe to skip if done)
echo ─────────────────────────────────────────────────────────────
echo.
python dataset_adapter.py
echo.
echo ─────────────────────────────────────────────────────────────
echo  STEP 2: Starting Local Blockchain (Ganache)
echo ─────────────────────────────────────────────────────────────
echo.
echo Starting Ganache on port 7545...
start "Ganache Blockchain" cmd /k ganache-cli -p 7545 -m "candy maple cake sugar pudding cream honey rich smooth crumble sweet treat"
timeout /t 5 >nul

echo.
echo ─────────────────────────────────────────────────────────────
echo  STEP 3: Starting all platform components...
echo ─────────────────────────────────────────────────────────────
echo.

:: 1. Start Trust Agent (Fraud Monitor)
echo [1/5] Starting Trust Agent...
start "Trust Agent" cmd /k "cd /d %~dp0 && python agents/trust_agent.py"
timeout /t 3 >nul

:: 2. Start Bank Node 1 (Synthetic Fraud Dataset)
echo [2/5] Starting Bank Node 1 [synthetic_fraud_dataset.csv]...
start "Bank 1 - Synthetic Fraud" cmd /k "cd /d %~dp0 && python main_node.py 1"
timeout /t 2 >nul

:: 3. Start Bank Node 2 (EU Credit Card Dataset)
echo [3/5] Starting Bank Node 2 [EU Credit Card - creditcard.csv]...
start "Bank 2 - EU Credit Card" cmd /k "cd /d %~dp0 && python main_node.py 2"
timeout /t 2 >nul

:: 4. Start Bank Node 3 (Mobile Money / PaySim Dataset)
echo [4/5] Starting Bank Node 3 [PaySim Mobile Money]...
start "Bank 3 - Mobile Money" cmd /k "cd /d %~dp0 && python main_node.py 3"
timeout /t 2 >nul

:: 5. Start Dashboard
echo [5/5] Starting Dashboard...
start "FL Dashboard" cmd /k "cd /d %~dp0 && streamlit run dashboard/real_system_app.py --server.fileWatcherType none --server.port 8501"
timeout /t 3 >nul

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║  All components launched successfully!                   ║
echo ║                                                          ║
echo ║  Bank 1 (Node 1): synthetic_fraud_dataset.csv            ║
echo ║  Bank 2 (Node 2): bank2_adapted.csv (EU Credit Card)     ║
echo ║  Bank 3 (Node 3): bank3_adapted.csv (Mobile Money)       ║
echo ║                                                          ║
echo ║  Dashboard: http://localhost:8501                        ║
echo ║  P2P Ports: 8001 (Bank1), 8002 (Bank2), 8003 (Bank3)    ║
echo ╚══════════════════════════════════════════════════════════╝
echo.
pause
