@echo off

echo Activating virtual environment...
call myenv\Scripts\activate

echo Starting FastAPI backend...
start cmd /k uvicorn backend:app --reload

echo Starting frontend server...
start cmd /k python -m http.server 5500

echo ===================================
echo App is running:
echo Backend:  http://127.0.0.1:8000
echo Frontend: http://127.0.0.1:5500
echo ===================================

pause