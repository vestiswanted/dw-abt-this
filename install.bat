@echo off
setlocal enabledelayedexpansion

echo =============================================================
echo  Rofus AMD Edition  --  Installation Script
echo =============================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python not found in PATH.
    echo         Get Python 3.10 from: https://www.python.org/downloads/
    echo         Make sure to tick "Add Python to PATH" during install.
    pause
    exit /b 1
)

python --version

:: Upgrade pip quietly
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip -q

:: Install main requirements
echo [2/4] Installing requirements...
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] pip install failed. Check your internet connection and try again.
    pause
    exit /b 1
)

:: Install BetterCam (requires git)
echo [3/4] Installing BetterCam...
where git >nul 2>&1
if errorlevel 1 (
    echo [WARN] git not found – skipping BetterCam install.
    echo        Download git from: https://git-scm.com/download/win
    echo        Then run:  pip install git+https://github.com/RootKit-Org/BetterCam.git
) else (
    pip install git+https://github.com/RootKit-Org/BetterCam.git -q
    echo        BetterCam installed.
)

:: Create lib directory
if not exist "lib\config" mkdir lib\config

echo [4/4] Done!
echo.
echo =============================================================
echo  NEXT STEPS
echo =============================================================
echo  1. Copy your model file:
echo       best.pt  -^>  lib\best.pt
echo.
echo  2. Export it to ONNX (one-time, needs torch installed):
echo       pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
echo       python export_model.py
echo       (after export you can uninstall torch to save space)
echo.
echo  3. Launch:
echo       start.bat
echo =============================================================
pause
