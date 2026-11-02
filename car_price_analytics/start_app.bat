@echo off
echo ==============================================================
echo Starting Used Car Price Prediction & Automobile Analytics System
echo ==============================================================
echo.
cd /d "%~dp0"
echo Opening browser at http://127.0.0.1:8000 ...
start http://127.0.0.1:8000
echo Running Django Server...
python manage.py runserver 127.0.0.1:8000
pause
