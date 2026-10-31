@echo off
title Academic RAG Test Suite
echo ========================================================
echo   Running Academic RAG Test Suite
echo ========================================================
echo.
cd /d "%~dp0"
.\.venv\Scripts\python.exe -m pytest tests/test_guardrails.py -v
pause

