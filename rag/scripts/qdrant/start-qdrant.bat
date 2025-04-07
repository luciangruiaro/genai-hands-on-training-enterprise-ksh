@echo off
echo Starting Qdrant using Docker Compose...

:: Check if Docker is installed
where docker >nul 2>nul
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker and try again.
    exit /b 1
)

:: Start Qdrant with Docker Compose
docker-compose up -d

:: Wait for a moment and check if the container is running
timeout /t 5 /nobreak >nul

docker ps | findstr "qdrant-local" >nul
if %errorlevel% equ 0 (
    echo Qdrant is running at http://localhost:6333
) else (
    echo Failed to start Qdrant.
    exit /b 1
)
