#!/bin/bash

echo "Starting Qdrant using Docker Compose..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "Docker is not installed. Please install Docker and try again."
    exit 1
fi

# Start Qdrant with Docker Compose
docker-compose up -d

# Check if Qdrant is running
if docker ps | grep -q "genai-handson-qdrant"; then
    echo "Qdrant is running at http://localhost:6333"
else
    echo "Failed to start Qdrant."
    exit 1
fi
