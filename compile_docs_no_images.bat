@echo off
REM ============================================================================
REM Adaptive-P Documentation Compiler (NO IMAGES VERSION)
REM ============================================================================
REM This batch file wraps the PowerShell script that compiles documentation
REM WITHOUT embedding images (smaller file size, text-only placeholders).
REM 
REM Usage:
REM   compile_docs_no_images.bat                    - Outputs to Documentation_NoImages.md
REM   compile_docs_no_images.bat CustomOutput.md    - Outputs to specified file
REM ============================================================================

setlocal

set "SCRIPT_DIR=%~dp0"
set "PS_SCRIPT=%SCRIPT_DIR%compile_docs_no_images.ps1"
set "OUTPUT_FILE=%~1"

if "%OUTPUT_FILE%"=="" (
    set "OUTPUT_FILE=Documentation_NoImages.md"
)

echo.
echo ============================================
echo   Adaptive-P Documentation Compiler
echo   (NO IMAGES - Text Only Version)
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
