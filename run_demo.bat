@echo off
title Decentralized AI Platform Launcher

echo ===================================================
echo   Universal Decentralized Agentic FL Platform
echo ===================================================
echo.
echo Starting Components...
echo.

:: 1. Start Trust Agent
start "Trust Agent" cmd /k "python agents/trust_agent.py"
timeout /t 2 >nul

:: 2. Start Node 1
start "Bank Node 1" cmd /k "python main_node.py 1"
timeout /t 2 >nul

:: 3. Start Node 2
start "Bank Node 2" cmd /k "python main_node.py 2"
timeout /t 2 >nul

:: 4. Start Dashboard
start "Dashboard" cmd /k "streamlit run dashboard/app.py --server.fileWatcherType none"

echo.
echo All components executed. 
echo - Trust Agent: Verifying proofs
echo - Node 1 & 2: Training on fraud dataset
echo - Dashboard: Visualizing progress
echo.
pause
