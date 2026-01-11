#!/bin/bash

# Script to test IMX219-83 Stereo Camera functionality
# Run this after rebooting with the camera overlay enabled

echo "========================================="
echo "IMX219-83 Stereo Camera Test Script"
echo "========================================="
echo ""

# Check for video devices
echo "1. Checking for video devices..."
if ls /dev/video* &> /dev/null; then
    echo "✓ Video devices found:"
    ls -l /dev/video*
else
    echo "✗ No video devices found. Make sure you have rebooted after running the activation script."
    exit 1
fi

echo ""
echo "2. Listing V4L2 devices..."
v4l2-ctl --list-devices

echo ""
echo "3. Checking for IMX219 sensors in kernel messages..."
dmesg | grep -i imx219 | tail -10

echo ""
echo "4. Camera sensor information..."
echo "Camera 0 (sensor-id=0):"
v4l2-ctl --device=/dev/video0 --all 2>/dev/null | head -20

echo ""
echo "Camera 1 (sensor-id=1):"
v4l2-ctl --device=/dev/video1 --all 2>/dev/null | head -20

echo ""
echo "========================================="
echo "Camera Test Commands"
echo "========================================="
echo ""
echo "Test Camera 0 (Left):"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink"
echo ""
echo "Test Camera 1 (Right):"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink"
echo ""
echo "Capture image from Camera 0:"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=3280,height=2464' ! nvjpegenc ! filesink location=camera0.jpg"
echo ""
echo "Capture image from Camera 1:"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=1 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=3280,height=2464' ! nvjpegenc ! filesink location=camera1.jpg"
echo ""
echo "Display both cameras side by side:"
echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=960,height=540' ! nvvidconv ! 'video/x-raw,width=960,height=540' ! videobox left=0 ! comp. nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=960,height=540' ! nvvidconv ! 'video/x-raw,width=960,height=540' ! videobox left=-960 ! comp. compositor name=comp ! xvimagesink"
echo ""
