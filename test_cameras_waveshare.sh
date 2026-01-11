#!/bin/bash

# Enhanced Camera Test Script - Following Waveshare Guide
# This script tests IMX219-83 Stereo Camera using methods from Waveshare documentation

echo "========================================="
echo "IMX219-83 Camera Test (Waveshare Method)"
echo "========================================="
echo ""

# Check if display is available
if [ -z "$DISPLAY" ]; then
    echo "⚠️  Warning: DISPLAY environment variable not set"
    echo "   Setting DISPLAY=:0.0"
    export DISPLAY=:0.0
fi

# Function to check if cameras are detected
check_cameras() {
    echo "Step 1: Checking for video devices..."
    echo "────────────────────────────────────────"
    if ls /dev/video* &> /dev/null; then
        echo "✓ Video devices found:"
        ls -l /dev/video*
        echo ""
        return 0
    else
        echo "✗ No video devices found!"
        echo ""
        echo "This means the cameras are not detected by the system."
        echo "Please run the activation script and reboot:"
        echo "  sudo ./activate_imx219_cameras.sh"
        echo "  sudo reboot"
        echo ""
        return 1
    fi
}

# Function to show device info
show_device_info() {
    echo "Step 2: Camera Device Information"
    echo "────────────────────────────────────────"
    
    if command -v v4l2-ctl &> /dev/null; then
        echo "Using v4l2-ctl to get device info..."
        echo ""
        
        v4l2-ctl --list-devices
        
        echo ""
        echo "Camera 0 capabilities:"
        v4l2-ctl --device=/dev/video0 --all 2>/dev/null | head -30
        
        echo ""
        echo "Camera 1 capabilities:"
        v4l2-ctl --device=/dev/video1 --all 2>/dev/null | head -30
    else
        echo "v4l2-ctl not available"
    fi
    echo ""
}

# Function to test with nvgstcapture (Waveshare method)
test_nvgstcapture() {
    local sensor_id=$1
    local camera_name=$2
    
    echo "────────────────────────────────────────"
    echo "Testing $camera_name with nvgstcapture-1.0"
    echo "────────────────────────────────────────"
    echo ""
    echo "Command: DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=$sensor_id"
    echo ""
    echo "This will open a preview window on your display."
    echo "Press Ctrl+C to stop the preview."
    echo ""
    
    read -p "Press Enter to start camera test (or Ctrl+C to skip)..." -r
    
    # Run nvgstcapture
    DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=$sensor_id
    
    echo ""
    echo "$camera_name test completed."
    echo ""
}

# Main execution
echo "This script follows the Waveshare installation guide for testing cameras."
echo ""

# Check if cameras are detected
if ! check_cameras; then
    exit 1
fi

# Show device info
show_device_info

# Ask user which test to run
echo "========================================="
echo "Select Test Method:"
echo "========================================="
echo "1) Test Camera 0 (Left) with nvgstcapture"
echo "2) Test Camera 1 (Right) with nvgstcapture"
echo "3) Test both cameras sequentially"
echo "4) Show GStreamer test commands"
echo "5) Show Python test commands"
echo "6) Exit"
echo ""

read -p "Enter your choice [1-6]: " choice

case $choice in
    1)
        test_nvgstcapture 0 "Camera 0 (Left)"
        ;;
    2)
        test_nvgstcapture 1 "Camera 1 (Right)"
        ;;
    3)
        test_nvgstcapture 0 "Camera 0 (Left)"
        test_nvgstcapture 1 "Camera 1 (Right)"
        ;;
    4)
        echo ""
        echo "GStreamer Test Commands:"
        echo "────────────────────────────────────────"
        echo ""
        echo "Camera 0 (1080p @ 30fps):"
        echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \\"
        echo "    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \\"
        echo "    nvvidconv ! xvimagesink"
        echo ""
        echo "Camera 1 (1080p @ 30fps):"
        echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \\"
        echo "    'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \\"
        echo "    nvvidconv ! xvimagesink"
        echo ""
        echo "Capture high-res image from Camera 0:"
        echo "  gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! \\"
        echo "    'video/x-raw(memory:NVMM),width=3280,height=2464' ! \\"
        echo "    nvjpegenc ! filesink location=camera0.jpg"
        echo ""
        ;;
    5)
        echo ""
        echo "Python Test Commands:"
        echo "────────────────────────────────────────"
        echo ""
        echo "Check camera availability:"
        echo "  python3 imx219_camera_test.py check"
        echo ""
        echo "View Camera 0:"
        echo "  python3 imx219_camera_test.py view 0"
        echo ""
        echo "View Camera 1:"
        echo "  python3 imx219_camera_test.py view 1"
        echo ""
        echo "View both cameras (stereo):"
        echo "  python3 imx219_camera_test.py stereo"
        echo ""
        echo "Capture image from Camera 0:"
        echo "  python3 imx219_camera_test.py capture 0"
        echo ""
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac

echo ""
echo "========================================="
echo "Test Complete"
echo "========================================="
echo ""
echo "If cameras are not working:"
echo "  1. Check hardware connections"
echo "  2. Ensure boot config is set to JetsonIO"
echo "  3. Reboot if you haven't already"
echo "  4. Check kernel logs: dmesg | grep -i imx219"
echo ""
echo "If images appear too red:"
echo "  Run: ./install_camera_overrides.sh"
echo ""
