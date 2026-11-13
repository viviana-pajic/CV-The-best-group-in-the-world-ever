@echo off
setlocal
where py.exe >nul 2>nul
if errorlevel 1 (
    python "%~dp0open_dashboard.py" %*
) else (
    py -3 "%~dp0open_dashboard.py" %*
)
if errorlevel 1 pause
