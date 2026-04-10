@echo off
title Decentralized AI Security Demo
echo ===================================================
echo   Security and Attack Simulation Demo
echo ===================================================
echo.
echo Launching Dashboard Nodes...
echo.

:: 1. Start Node 1 (Honest Validator)
start "Node 1 (HONEST)" cmd /k "streamlit run dashboard/real_system_app.py --server.port 8502 -- --node_id 1"

:: 2. Start Node 2 (Potentially Malicious)
start "Node 2 (ATTACKER)" cmd /k "streamlit run dashboard/real_system_app.py --server.port 8503 -- --node_id 2"

echo.
echo ---------------------------------------------------
echo INSTRUCTIONS:
echo 1. Open http://localhost:8502 (Node 1)
echo 2. Open http://localhost:8503 (Node 2)
echo 3. On Node 2, check "Simulate Attack"
echo 4. Click "Start Real System" on both!
echo ---------------------------------------------------
pause
