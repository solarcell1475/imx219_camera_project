#!/bin/bash
# Setup GPU Environment for YOLO on Jetson
# This script installs PyTorch with CUDA support for Jetson devices

set -e

echo "=========================================="
echo "GPU Environment Setup for YOLO"
echo "=========================================="
echo ""

# Check JetPack version
echo "Checking JetPack version..."
JETPACK_VERSION=$(cat /etc/nv_tegra_release | grep -o "R[0-9]*" | head -1)
echo "JetPack version: $JETPACK_VERSION"
echo ""

# Check CUDA version
echo "Checking CUDA version..."
CUDA_VERSION=$(nvcc --version | grep "release" | grep -o "[0-9]\+\.[0-9]\+" | head -1)
echo "CUDA version: $CUDA_VERSION"
echo ""

# Activate conda environment
echo "Activating conda environment: yolo_gpu"
source /home/jetson/miniconda/etc/profile.d/conda.sh
conda activate yolo_gpu

# Check Python version
PYTHON_VERSION=$(python --version | grep -o "[0-9]\+\.[0-9]\+" | head -1)
echo "Python version: $PYTHON_VERSION"
echo ""

# Set up library paths
export LD_LIBRARY_PATH=/usr/local/cuda-${CUDA_VERSION}/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

echo "=========================================="
echo "Installing PyTorch with CUDA support"
echo "=========================================="
echo ""
echo "For JetPack 6 (R36), you need to download PyTorch from NVIDIA's repository."
echo "Please visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo ""
echo "Or try downloading directly:"
echo ""

# Try to download PyTorch 2.3.0 (newer version)
PYTORCH_URL="https://nvidia.box.com/shared/static/xxxxxxxxx.whl"
PYTORCH_VERSION="2.3.0"

echo "Attempting to download PyTorch ${PYTORCH_VERSION}..."
cd /tmp

# Note: The actual URL needs to be obtained from NVIDIA forums
# For now, we'll provide instructions
echo ""
echo "Manual installation required:"
echo "1. Visit: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048"
echo "2. Find the PyTorch wheel for JetPack ${JETPACK_VERSION} with Python ${PYTHON_VERSION}"
echo "3. Download the wheel file"
echo "4. Run: python -m pip install /path/to/torch-*.whl"
echo ""
echo "Alternatively, if you have the wheel file, install it now:"
echo "python -m pip install /tmp/torch-*.whl"
echo ""

# Check if wheel exists
if ls /tmp/torch-*-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_aarch64.whl 1> /dev/null 2>&1; then
    echo "Found PyTorch wheel, installing..."
    python -m pip install /tmp/torch-*-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_aarch64.whl
else
    echo "No PyTorch wheel found in /tmp"
    echo "Please download it manually from NVIDIA forums"
fi

echo ""
echo "Installing torchvision..."
# Try to install torchvision (may need to be downloaded separately)
if ls /tmp/torchvision-*-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_aarch64.whl 1> /dev/null 2>&1; then
    python -m pip install /tmp/torchvision-*-cp${PYTHON_VERSION//./}-cp${PYTHON_VERSION//./}-linux_aarch64.whl
else
    echo "Torchvision wheel not found. Install manually after PyTorch."
fi

echo ""
echo "=========================================="
echo "Verifying installation"
echo "=========================================="
python -c "
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU count:', torch.cuda.device_count())
else:
    print('CUDA not available - check installation')
"

echo ""
echo "Setup complete!"
echo "To use this environment, run: conda activate yolo_gpu"
