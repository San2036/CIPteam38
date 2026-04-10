@echo off
title True P2P AI Mesh Network and ZKP Demo
echo ===================================================
echo   Live Blockchain Network with True ZK-Proofs
echo ===================================================
echo.
echo Launching Visual Dashboard Engines...
echo.

:: 1. Start Node 1 Dashboard
start "Node 1 Monitor (UI)" cmd /k "streamlit run dashboard/real_system_app.py --server.port 8502 -- --node_id 1"

:: 2. Start Node 2 Dashboard
start "Node 2 Monitor (UI)" cmd /k "streamlit run dashboard/real_system_app.py --server.port 8503 -- --node_id 2"

echo.
echo ---------------------------------------------------
echo INSTRUCTIONS TO WATCH THE CHARTS:
echo 1. Wait for the browser tabs (http://localhost:8502 and 8503) to open.
echo 2. Go to the dashboard and simply click "Start Real System"!
echo 3. The Dashboard Engine will now automatically run the Training, generate the True Mathematical ZKP, and live-update your charts!
echo ---------------------------------------------------
pause
