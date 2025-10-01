@echo off
REM Complete pipeline script for Windows
REM This script trains the model and generates samples

echo ========================================
echo Transformer Training Pipeline
echo ========================================
echo.

REM Check if virtual environment is activated
python -c "import sys; sys.exit(0 if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix) else 1)" 2>nul
if errorlevel 1 (
    echo WARNING: Virtual environment not detected!
    echo Please activate your venv first: venv\Scripts\activate
    echo.
    pause
    exit /b 1
)

echo Virtual environment: Active
echo.

REM Check PyTorch installation
echo Checking PyTorch installation...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')" 2>nul
if errorlevel 1 (
    echo ERROR: PyTorch not installed or import failed
    echo Please install requirements: pip install -r requirements.txt
    pause
    exit /b 1
)
echo.

echo ========================================
echo Step 1: Training Model
echo ========================================
echo.
python train.py
if errorlevel 1 (
    echo ERROR: Training failed!
    pause
    exit /b 1
)
echo.
echo Training complete!
echo.

echo ========================================
echo Step 2: Generating Text Samples
echo ========================================
echo.
python generate.py --samples
if errorlevel 1 (
    echo ERROR: Sample generation failed!
    pause
    exit /b 1
)
echo.
echo Sample generation complete!
echo.

echo ========================================
echo Pipeline Complete!
echo ========================================
echo.
echo Generated files:
echo   - best_model.pt (trained model)
echo   - training_log_*.csv (training metrics)
echo   - sample_romeo.txt (ROMEO sample)
echo   - sample_juliet.txt (JULIET sample)
echo.
echo To generate more samples:
echo   python generate.py --prompt "YOUR_PROMPT:" --tokens 200
echo.
echo To use interactive mode:
echo   python generate.py --interactive
echo.
pause