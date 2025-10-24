@echo off
echo ========================================
echo  Regression Analysis Hub - Installation
echo ========================================
echo.

echo Checking Python installation...
python --version
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH!
    echo Please install Python 3.8 or higher from python.org
    pause
    exit /b 1
)
echo.

echo Installing required packages...
echo This may take a few minutes...
echo.

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

if errorlevel 1 (
    echo.
    echo ERROR: Installation failed!
    echo Please check your internet connection and try again.
    pause
    exit /b 1
)

echo.
echo ========================================
echo  Installation Complete! 
echo ========================================
echo.
echo You can now run the app by:
echo 1. Double-clicking run_app.bat
echo 2. Or running: streamlit run regression_analysis_app.py
echo.
pause

