#!/bin/bash

# Script to activate IMX219-83 Stereo Camera on Jetson Orin Nano
# This script updates the boot configuration to enable dual IMX219 cameras

echo "========================================="
echo "IMX219-83 Stereo Camera Activation Script"
echo "========================================="
echo ""

# Backup the current configuration
echo "Creating backup of extlinux.conf..."
cp /boot/extlinux/extlinux.conf /boot/extlinux/extlinux.conf.backup.$(date +%Y%m%d_%H%M%S)

# Update the DEFAULT boot option to JetsonIO (which has the camera overlay)
echo "Updating boot configuration to enable dual IMX219 cameras..."
sed -i 's/^DEFAULT primary/DEFAULT JetsonIO/' /boot/extlinux/extlinux.conf

# Verify the change
echo ""
echo "Current boot configuration:"
grep -A 1 "^DEFAULT" /boot/extlinux/extlinux.conf
echo ""

# Check if the JetsonIO label exists
if grep -q "LABEL JetsonIO" /boot/extlinux/extlinux.conf; then
    echo "✓ Camera overlay configuration found in boot config"
    echo ""
    echo "Configuration details:"
    grep -A 7 "LABEL JetsonIO" /boot/extlinux/extlinux.conf | grep -E "(MENU LABEL|OVERLAYS)"
else
    echo "✗ Warning: JetsonIO configuration not found!"
    exit 1
fi

echo ""
echo "========================================="
echo "Configuration updated successfully!"
echo "========================================="
echo ""
echo "IMPORTANT: You need to REBOOT the system for the cameras to be activated."
echo ""
echo "After reboot, you can test the cameras with:"
echo "  ls /dev/video*"
echo "  v4l2-ctl --list-devices"
echo ""
echo "To test camera capture:"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink"
echo ""
read -p "Would you like to reboot now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Rebooting in 5 seconds... Press Ctrl+C to cancel"
    sleep 5
    reboot
else
    echo "Please reboot manually when ready: sudo reboot"
fi
