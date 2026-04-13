@echo off
cd /d C:\Users\ASUS\Desktop\PriceAnalyzingDashboard

echo ================================
echo  NSE Predict - Starting up...
echo ================================
echo.

:: Only install packages if requirements changed since last install
set REQ=requirements.txt
set MARKER=data\.packages_ok

if not exist "%MARKER%" goto install
for /f %%i in ('powershell -NoProfile -Command "(Get-Item \"%REQ%\").LastWriteTime -gt (Get-Item \"%MARKER%\").LastWriteTime"') do set NEWER=%%i
if "%NEWER%"=="True" goto install
echo [1/3] Packages already up-to-date. Skipping install.
goto start_backend

:install
echo [1/3] Installing Python packages (first run or requirements changed)...
C:\Python314\python.exe -m pip install -r requirements.txt -q --disable-pip-version-check
if errorlevel 1 (
    echo ERROR: pip install failed. Check requirements.txt.
    pause
    exit /b 1
)
if not exist data mkdir data
echo. > "%MARKER%"
echo Done.

:start_backend
echo.
echo [2/3] Starting backend...
start "NSE Backend" cmd /k "cd /d C:\Users\ASUS\Desktop\PriceAnalyzingDashboard && C:\Python314\python.exe -m uvicorn backend.main:app --port 8000 --host 0.0.0.0"

echo Waiting for backend to start...
timeout /t 6 /nobreak >nul

echo [3/3] Starting frontend...
start "NSE Frontend" cmd /k "cd /d C:\Users\ASUS\Desktop\PriceAnalyzingDashboard\frontend && npm run dev"

timeout /t 4 /nobreak >nul

echo.
echo Opening dashboard in Chrome...
start chrome http://localhost:5173

echo.
echo ================================
echo  Dashboard ready!
echo  Data auto-refreshes at 8:45 AM.
echo  Click Refresh to force update.
echo ================================
