@echo off
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%"
set "PYTHONPATH=%PROJECT_ROOT%.vendor;%PROJECT_ROOT%src"
py -m uvicorn src.main:app --host 127.0.0.1 --port 8000 1>> "%PROJECT_ROOT%backend.log" 2>> "%PROJECT_ROOT%backend.err.log"
