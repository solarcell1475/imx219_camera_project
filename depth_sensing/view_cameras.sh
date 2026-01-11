#!/bin/bash
#
# IMX219 Camera Viewer - Multiple Display Options
#

cd "$(dirname "${BASH_SOURCE[0]}")"

clear
echo "======================================================================"
echo "           IMX219 CAMERA VIEWER - LIVE VIDEO"
echo "======================================================================"
echo ""
echo "Select display mode:"
echo ""
echo "  1) Both cameras side-by-side (1280x720 each)"
echo "  2) Camera 0 only (full screen 1920x1080)"
echo "  3) Camera 1 only (full screen 1920x1080)"
echo "  4) Both cameras side-by-side (high quality 1920x1080 each)"
echo "  5) Exit"
echo ""
echo "======================================================================"
echo ""
read -p "Enter choice [1]: " choice
choice=${choice:-1}

echo ""

case $choice in
    1)
        echo "Starting both cameras side-by-side (1280x720)..."
        echo "Press Ctrl+C to stop"
        echo ""
        gst-launch-1.0 \
          nvarguscamerasrc sensor-id=0 ! \
          'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
          nvvidconv ! 'video/x-raw,width=1280,height=720' ! videobox left=0 ! comp. \
          nvarguscamerasrc sensor-id=1 ! \
          'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
          nvvidconv ! 'video/x-raw,width=1280,height=720' ! videobox left=-1280 ! comp. \
          compositor name=comp ! xvimagesink sync=false
        ;;
    
    2)
        echo "Starting Camera 0 (full screen 1920x1080)..."
        echo "Press Ctrl+C to stop"
        echo ""
        gst-launch-1.0 \
          nvarguscamerasrc sensor-id=0 ! \
          'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
          nvvidconv ! xvimagesink sync=false
        ;;
    
    3)
        echo "Starting Camera 1 (full screen 1920x1080)..."
        echo "Press Ctrl+C to stop"
        echo ""
        gst-launch-1.0 \
          nvarguscamerasrc sensor-id=1 ! \
          'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
          nvvidconv ! xvimagesink sync=false
        ;;
    
    4)
        echo "Starting both cameras side-by-side (1920x1080 each)..."
        echo "Press Ctrl+C to stop"
        echo "Note: This may be slow due to high resolution"
        echo ""
        gst-launch-1.0 \
          nvarguscamerasrc sensor-id=0 ! \
          'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
          nvvidconv ! 'video/x-raw,width=1920,height=1080' ! videobox left=0 ! comp. \
          nvarguscamerasrc sensor-id=1 ! \
          'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
          nvvidconv ! 'video/x-raw,width=1920,height=1080' ! videobox left=-1920 ! comp. \
          compositor name=comp ! xvimagesink sync=false
        ;;
    
    5|q|Q)
        echo "Exiting..."
        exit 0
        ;;
    
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "======================================================================"
echo "Video stopped"
echo "======================================================================"
echo ""
