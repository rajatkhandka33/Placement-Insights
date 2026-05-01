@echo off
cd /d "%~dp0"

if not exist ".venv" (
    echo Creating virtual environment...
    python -m venv .venv
)

echo Activating virtual environment...
call .venv\Scripts\activate.bat

echo Upgrading pip...
python -m pip install --upgrade pip

echo Installing dependencies...
python -m pip install --only-binary :all: greenlet
python -m pip install -r requirements.txt

echo Opening browser...
start http://127.0.0.1:8000/

echo Starting Uvicorn server...
python -m uvicorn app:app --host 127.0.0.1 --port 8000

pause
