#!/bin/bash
#
# IMX219 Stereo Camera - Real-Time Video Display
# Shows live video from both cameras side-by-side
#

cd "$(dirname "${BASH_SOURCE[0]}")"

clear
echo "======================================================================"
echo "           IMX219 STEREO CAMERA - LIVE VIDEO"
echo "======================================================================"
echo ""
echo "📹 Real-time video display from both cameras"
echo ""
echo "Left side:  Camera 0"
echo "Right side: Camera 1"
echo ""
echo "Resolution: 1280x720 @ 30fps"
echo "Display:    2560x720 (side-by-side)"
echo ""
echo "Press Ctrl+C in this terminal to stop"
echo ""
echo "======================================================================"
echo ""
read -p "Press ENTER to start live video..."
echo ""
echo "Starting cameras..."
echo ""

# Use GStreamer to display both cameras in real-time
# Higher resolution for better quality
gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! 'video/x-raw,width=1280,height=720' ! videobox left=0 ! comp. \
  nvarguscamerasrc sensor-id=1 ! \
  'video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1' ! \
  nvvidconv ! 'video/x-raw,width=1280,height=720' ! videobox left=-1280 ! comp. \
  compositor name=comp ! xvimagesink sync=false

echo ""
echo "======================================================================"
echo "Video stopped"
echo "======================================================================"
echo ""
