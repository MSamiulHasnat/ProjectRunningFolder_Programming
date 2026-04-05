@echo off
REM Auto-start RuFlo daemon for CT-MUSIQ project
REM Run this batch file to ensure the daemon is always running

cd /d "%~dp0"

REM Check if daemon is running
echo Checking RuFlo daemon status...
claude-flow daemon status > nul 2>&1

REM Start daemon if not running
if errorlevel 1 (
    echo Starting RuFlo daemon...
    claude-flow daemon start
) else (
    echo RuFlo daemon is already running.
)

echo.
echo RuFlo is ready for CT-MUSIQ thesis project.
echo.
pause
