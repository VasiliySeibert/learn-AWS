#!/bin/bash
# Helper script to run the learn-AWS application with proper environment configuration

echo "============================================"
echo "  learn-AWS Study Application"
echo "============================================"
echo ""
echo "⚠️  IMPORTANT: Before running, make sure:"
echo "   1. LM Studio is running on Windows"
echo "   2. The local server is started in LM Studio"
echo "   3. Model loaded: nvidia/llama-3.1-nemotron-nano-8b-v1"
echo ""
echo "Starting application..."
echo ""

# Set LM Studio URL (update this if your IP changes)
export LM_STUDIO_URL="http://192.168.2.57:1234/v1"

# Activate virtual environment
source .venv/bin/activate

# Run the application
python learning.py LearningMaterials/clf-c02.json "$@"
