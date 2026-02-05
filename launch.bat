@echo off
echo ============================================
echo  Suspension Setup Evaluator
echo ============================================
echo.

REM Verifica daca exista venv
if not exist "venv" (
    echo Creez mediul virtual...
    python -m venv venv
)

REM Activeaza venv
echo Activez mediul virtual...
call venv\Scripts\activate.bat

REM Verifica daca sunt instalate pachetele
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo Instalez dependintele...
    pip install -r requirements.txt
)

REM Porneste aplicatia
echo.
echo ============================================
echo  Pornesc aplicatia...
echo  Acceseaza: http://localhost:8501
echo ============================================
echo.

streamlit run main_app.py

pause