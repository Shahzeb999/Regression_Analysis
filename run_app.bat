@echo off
echo ========================================
echo  Regression Analysis Hub
echo ========================================
echo.
echo Starting Streamlit app...
echo The app will open in your default browser.
echo.
echo To stop the app, press Ctrl+C in this window.
echo ========================================
echo.

streamlit run regression_analysis_app.py

if errorlevel 1 (
    echo.
    echo ERROR: Failed to start the app!
    echo.
    echo Possible solutions:
    echo 1. Run install.bat first to install dependencies
    echo 2. Make sure regression_analysis_app.py exists
    echo 3. Check if Streamlit is installed: pip install streamlit
    echo.
    pause
)

