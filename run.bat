@echo off
title Academic RAG Assistant
echo ========================================================
echo   Launching Academic RAG Assistant Website
echo   http://127.0.0.1:8000
echo ========================================================
echo.
cd /d "%~dp0"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
pause

