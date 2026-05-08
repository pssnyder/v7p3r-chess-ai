@echo off
REM Quick Start Script for V7P3R Docker Training
REM Automates setup and launch for 48-hour training run

echo ================================================================================
echo V7P3R AI - Docker Training Quick Start
echo ================================================================================
echo.

REM Check Docker is running
docker info >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Docker is not running!
    echo Please start Docker Desktop and try again.
    pause
    exit /b 1
)

echo [1/5] Checking Docker installation...
docker --version
docker-compose --version
echo.

REM Check for GPU support
echo [2/5] Checking GPU support...
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo ✓ GPU support available - will use GPU training
    set USE_GPU=true
) else (
    echo ⚠ No GPU support - will use CPU training
    set USE_GPU=false
)
echo.

REM Create necessary directories
echo [3/5] Creating directories...
if not exist "data" mkdir data
if not exist "models" mkdir models
if not exist "logs" mkdir logs
if not exist "tensorboard_logs" mkdir tensorboard_logs
if not exist "checkpoints" mkdir checkpoints
echo ✓ Directories created
echo.

REM Build Docker image
echo [4/5] Building Docker image (this may take 5-10 minutes)...
docker-compose build --progress=plain 2>&1
if %errorlevel% neq 0 (
    echo.
    echo ERROR: Docker build failed!
    echo.
    echo Common fixes:
    echo   1. Check Docker Desktop is running and up to date
    echo   2. Free up disk space (need ~10GB for build)
    echo   3. Check internet connection for package downloads
    echo   4. Try: docker system prune -a
    echo.
    echo To see full error, run: docker-compose build --progress=plain
    echo.
    pause
    exit /b 1
)
echo ✓ Docker image built successfully
echo.

REM Start training
echo [5/5] Starting 48-hour training...
echo.
echo Training will run in the background. You can:
echo   - Monitor logs:      docker-compose logs -f training
echo   - Check health:      docker-compose exec training python scripts/training_health_check.py
echo   - View TensorBoard:  http://localhost:6006
echo   - Stop training:     docker-compose down
echo.
echo Press Ctrl+C during this script to cancel startup (training has not started yet)
timeout /t 5

if "%USE_GPU%"=="true" (
    docker-compose up -d
) else (
    docker-compose -f docker-compose.yml -f docker-compose.cpu.yml up -d
)

if %errorlevel% neq 0 (
    echo ERROR: Failed to start training container!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo ✓ Training started successfully!
echo ================================================================================
echo.
echo Monitoring commands:
echo   docker-compose logs -f training          ^| View live logs
echo   docker-compose exec training python scripts/training_health_check.py
echo.
echo TensorBoard: http://localhost:6006
echo.
echo To stop training: docker-compose down
echo.
echo Opening TensorBoard in browser...
timeout /t 3
start http://localhost:6006

echo.
echo Press any key to view live logs (Ctrl+C to exit logs)...
pause >nul

docker-compose logs -f training
