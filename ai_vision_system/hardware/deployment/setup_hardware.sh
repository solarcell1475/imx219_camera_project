#!/bin/bash

# One-click Hardware Setup Script
# =================================
# Automates hardware setup and verification for IMX219 dual cameras

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

echo "========================================="
echo "IMX219 Dual Camera Hardware Setup"
echo "========================================="
echo ""

# Step 1: Check if cameras are already activated
echo "Step 1: Checking camera activation..."
if grep -q "DEFAULT JetsonIO" /boot/extlinux/extlinux.conf 2>/dev/null; then
    echo "✓ Cameras appear to be activated in boot config"
else
    echo "⚠ Cameras may not be activated"
    echo "  Run: sudo $PROJECT_ROOT/activate_imx219_cameras.sh"
fi

echo ""

# Step 2: Check for video devices
echo "Step 2: Checking for video devices..."
VIDEO_DEVICES=$(ls /dev/video* 2>/dev/null | wc -l)
if [ "$VIDEO_DEVICES" -ge 2 ]; then
    echo "✓ Found $VIDEO_DEVICES video device(s)"
    ls -lh /dev/video* | head -2
else
    echo "✗ Found only $VIDEO_DEVICES video device(s) (expected at least 2)"
    echo "  You may need to reboot after running activation script"
fi

echo ""

# Step 3: Run Python detection script
echo "Step 3: Running camera detection..."
cd "$SCRIPT_DIR/../installation"
if python3 camera_detection.py; then
    echo "✓ Camera detection completed"
else
    echo "⚠ Camera detection had issues"
fi

echo ""

# Step 4: Run driver health check
echo "Step 4: Checking driver health..."
cd "$SCRIPT_DIR/../drivers"
if python3 camera_driver.py; then
    echo "✓ Driver health check completed"
else
    echo "⚠ Driver health check had issues"
fi

echo ""

# Step 5: Quick hardware test
echo "Step 5: Running quick hardware test..."
read -p "Run comprehensive hardware test? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    cd "$SCRIPT_DIR/../testing"
    python3 hardware_test_suite.py
else
    echo "Skipping hardware test"
fi

echo ""
echo "========================================="
echo "Hardware setup verification complete!"
echo "========================================="
