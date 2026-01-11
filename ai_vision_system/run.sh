#!/bin/bash

# Quick Start Script for YOLO AI Vision System
# ============================================

cd "$(dirname "$0")"

echo "========================================="
echo "YOLO AI Vision System - Quick Start"
echo "========================================="
echo ""

# Check if in correct directory
if [ ! -f "main.py" ]; then
    echo "Error: main.py not found!"
    echo "Please run this script from the ai_vision_system directory"
    exit 1
fi

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found!"
    exit 1
fi

# Check for GPU environment
if command -v conda &> /dev/null; then
    if conda env list | grep -q "yolo_gpu"; then
        echo "GPU environment 'yolo_gpu' found. Activating..."
        source "$(conda info --base)/etc/profile.d/conda.sh"
        conda activate yolo_gpu 2>/dev/null || echo "Warning: Could not activate yolo_gpu environment"
    fi
fi

echo "Starting YOLO AI Vision System..."
echo ""
echo "Controls:"
echo "  Q/ESC - Quit"
echo "  S - Save frame"
echo "  D - Toggle detections"
echo "  V - Change view mode"
echo "  +/- - Adjust confidence"
echo ""
echo "Press Ctrl+C to stop"
echo ""

# Run the main program
python3 main.py "$@"
