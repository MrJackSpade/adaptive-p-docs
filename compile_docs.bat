@echo off
REM ============================================================================
REM Adaptive-P Documentation Compiler
REM ============================================================================
REM This batch file wraps the PowerShell script that compiles documentation.
REM 
REM Usage:
REM   compile_docs.bat                    - Outputs to Documentation.md
REM   compile_docs.bat CustomOutput.md    - Outputs to specified file
REM ============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%compile_docs.ps1"
set "OUTPUT_FILE=%~1"

if "%OUTPUT_FILE%"=="" (
    set "OUTPUT_FILE=Documentation.md"
)

echo.
echo ============================================
echo   Adaptive-P Documentation Compiler
echo ============================================
echo.

REM Check if PowerShell script exists
if not exist "%PS_SCRIPT%" (
    echo ERROR: PowerShell script not found: %PS_SCRIPT%
    echo.
    pause
    exit /b 1
)

REM Run the PowerShell script with execution policy bypass
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%PS_SCRIPT%" -OutputFile "%OUTPUT_FILE%"

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Documentation compilation failed.
    pause
    exit /b 1
)

echo.
echo Press any key to exit...
pause >nul
