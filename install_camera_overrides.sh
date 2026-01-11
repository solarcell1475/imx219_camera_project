#!/bin/bash

# Script to install camera_overrides.isp file
# This fixes reddish color issues with IMX219 cameras
# Source: Waveshare IMX219-83 Installation Guide

echo "========================================="
echo "Camera ISP Override Installation"
echo "========================================="
echo ""
echo "This script downloads and installs the camera_overrides.isp file"
echo "from Waveshare to fix color balance issues (especially reddish tint)."
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then 
    echo "❌ This script must be run with sudo"
    echo "Usage: sudo $0"
    exit 1
fi

# Create temp directory
TEMP_DIR="/tmp/camera_isp_install"
mkdir -p "$TEMP_DIR"
cd "$TEMP_DIR" || exit 1

echo "Step 1: Downloading camera_overrides.isp..."
echo "────────────────────────────────────────"

# Download file
wget http://www.waveshare.net/w/upload/e/eb/Camera_overrides.tar.gz

if [ $? -ne 0 ]; then
    echo "❌ Failed to download file"
    echo "Please check your internet connection"
    exit 1
fi

echo "✓ Download complete"
echo ""

echo "Step 2: Extracting archive..."
echo "────────────────────────────────────────"

tar zxvf Camera_overrides.tar.gz

if [ $? -ne 0 ]; then
    echo "❌ Failed to extract archive"
    exit 1
fi

echo "✓ Extraction complete"
echo ""

echo "Step 3: Creating nvcam settings directory..."
echo "────────────────────────────────────────"

# Create directory if it doesn't exist
mkdir -p /var/nvidia/nvcam/settings/

echo "✓ Directory ready"
echo ""

echo "Step 4: Installing camera_overrides.isp..."
echo "────────────────────────────────────────"

# Check if file exists before copying
if [ ! -f "camera_overrides.isp" ]; then
    echo "❌ camera_overrides.isp file not found in archive"
    echo "Archive contents:"
    ls -la
    exit 1
fi

# Backup existing file if present
if [ -f "/var/nvidia/nvcam/settings/camera_overrides.isp" ]; then
    echo "Backing up existing file..."
    cp /var/nvidia/nvcam/settings/camera_overrides.isp \
       /var/nvidia/nvcam/settings/camera_overrides.isp.backup.$(date +%Y%m%d_%H%M%S)
fi

# Copy file
cp camera_overrides.isp /var/nvidia/nvcam/settings/

if [ $? -ne 0 ]; then
    echo "❌ Failed to copy file"
    exit 1
fi

echo "✓ File installed"
echo ""

echo "Step 5: Setting file permissions..."
echo "────────────────────────────────────────"

# Set permissions
chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp

echo "✓ Permissions set"
echo ""

echo "Step 6: Verifying installation..."
echo "────────────────────────────────────────"

if [ -f "/var/nvidia/nvcam/settings/camera_overrides.isp" ]; then
    echo "✓ File successfully installed:"
    ls -lh /var/nvidia/nvcam/settings/camera_overrides.isp
else
    echo "❌ File not found after installation"
    exit 1
fi

echo ""

# Cleanup
echo "Step 7: Cleaning up temporary files..."
echo "────────────────────────────────────────"
cd /
rm -rf "$TEMP_DIR"
echo "✓ Cleanup complete"
echo ""

echo "========================================="
echo "Installation Complete!"
echo "========================================="
echo ""
echo "The camera_overrides.isp file has been installed to:"
echo "  /var/nvidia/nvcam/settings/camera_overrides.isp"
echo ""
echo "This file will be automatically used by the camera system"
echo "to improve color balance and reduce reddish tint."
echo ""
echo "⚠️  You may need to reboot for changes to take full effect:"
echo "   sudo reboot"
echo ""
echo "After rebooting, test your cameras again:"
echo "  DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0"
echo "  DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=1"
echo ""
