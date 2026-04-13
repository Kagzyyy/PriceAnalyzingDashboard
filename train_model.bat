@echo off
title Train XGBoost Model - NSE Dashboard
echo.
echo =============================================
echo   NSE XGBoost Model Training
echo   This will fetch 2 years of data and train.
echo   Expected time: 10-30 minutes.
echo =============================================
echo.

if not exist "venv\" (
    python -m venv venv
)
call venv\Scripts\activate.bat
pip install -r requirements.txt -q

set PYTHONPATH=%~dp0
python -m backend.ml.model

echo.
echo Training complete! You can now start the dashboard.
pause
