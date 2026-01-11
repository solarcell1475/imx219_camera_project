#!/bin/bash
# Setup library symlinks for PyTorch compatibility
# This script creates symlinks for older library versions that PyTorch wheels expect

set -e

echo "Setting up library symlinks for PyTorch..."

# Create symlinks for MPI libraries
if [ -f /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 ] && [ ! -f /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20 ]; then
    echo "Creating symlink for libmpi_cxx.so.20"
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.40 /usr/lib/aarch64-linux-gnu/libmpi_cxx.so.20
fi

if [ -f /usr/lib/aarch64-linux-gnu/libmpi.so.40 ] && [ ! -f /usr/lib/aarch64-linux-gnu/libmpi.so.20 ]; then
    echo "Creating symlink for libmpi.so.20"
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libmpi.so.40 /usr/lib/aarch64-linux-gnu/libmpi.so.20
fi

# Create symlinks for CUDA libraries
if [ -f /usr/local/cuda-12.6/lib64/libcufft.so.11 ] && [ ! -f /usr/local/cuda-12.6/lib64/libcufft.so.10 ]; then
    echo "Creating symlink for libcufft.so.10"
    sudo ln -sf /usr/local/cuda-12.6/lib64/libcufft.so.11 /usr/local/cuda-12.6/lib64/libcufft.so.10
fi

# Create symlinks for cuDNN (PyTorch may expect cuDNN 8 but system has cuDNN 9)
if [ -f /usr/lib/aarch64-linux-gnu/libcudnn.so.9 ] && [ ! -f /usr/lib/aarch64-linux-gnu/libcudnn.so.8 ]; then
    echo "Creating symlink for libcudnn.so.8"
    sudo ln -sf /usr/lib/aarch64-linux-gnu/libcudnn.so.9 /usr/lib/aarch64-linux-gnu/libcudnn.so.8
fi

# Create symlinks for CUDA runtime (PyTorch expects CUDA 10.2, system has CUDA 12.6)
if [ -f /usr/local/cuda-12.6/lib64/libcudart.so.12 ] && [ ! -f /usr/local/cuda-12.6/lib64/libcudart.so.10.2 ]; then
    echo "Creating symlink for libcudart.so.10.2"
    sudo ln -sf /usr/local/cuda-12.6/lib64/libcudart.so.12 /usr/local/cuda-12.6/lib64/libcudart.so.10.2
fi

# Create symlinks for other CUDA libraries that PyTorch might expect
for lib in libcurand libcusolver libcusparse libcublas; do
    if [ -f /usr/local/cuda-12.6/lib64/${lib}.so.12 ] && [ ! -f /usr/local/cuda-12.6/lib64/${lib}.so.10 ]; then
        echo "Creating symlink for ${lib}.so.10"
        sudo ln -sf /usr/local/cuda-12.6/lib64/${lib}.so.12 /usr/local/cuda-12.6/lib64/${lib}.so.10
    fi
done

# Add CUDA to library path
if ! grep -q "CUDA.*lib64" /etc/ld.so.conf.d/*.conf 2>/dev/null; then
    echo "Adding CUDA to library path..."
    echo "/usr/local/cuda-12.6/lib64" | sudo tee /etc/ld.so.conf.d/cuda.conf > /dev/null
    sudo ldconfig
fi

echo "Library setup complete!"
