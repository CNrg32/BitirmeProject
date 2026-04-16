@echo off
set "PROJECT_ROOT=%~dp0"
cd /d "%PROJECT_ROOT%mobile"
"%PROJECT_ROOT%.flutter_sdk\bin\flutter.bat" run -d web-server --web-hostname 127.0.0.1 --web-port 8080 1>> "%PROJECT_ROOT%frontend.log" 2>> "%PROJECT_ROOT%frontend.err.log"
