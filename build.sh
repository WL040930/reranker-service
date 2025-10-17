#!/bin/bash
# Docker build script with error handling and debugging

set -e  # Exit on any error

echo "🔧 Building reranker-service Docker image..."

# Check if Docker is available
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed or not in PATH"
    exit 1
fi

# Build with detailed output
echo "📦 Starting Docker build..."
docker build \
    --progress=plain \
    --no-cache \
    -t reranker-service \
    . 2>&1 | tee build.log

if [ $? -eq 0 ]; then
    echo "✅ Build successful!"
    echo "🚀 You can now run: docker run -p 8000:8000 reranker-service"
else
    echo "❌ Build failed. Check build.log for details."
    exit 1
fi