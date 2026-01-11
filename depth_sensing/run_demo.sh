#!/bin/bash
#
# IMX219 Depth Sensing - DEMO MODE (No Calibration)
# ==================================================
# Quick launcher for demo mode
#

# Colors
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

clear
echo -e "${CYAN}======================================================================${NC}"
echo -e "${CYAN}           IMX219 DEPTH SENSING - DEMO MODE${NC}"
echo -e "${CYAN}======================================================================${NC}"
echo ""
echo -e "${RED}⚠️  WARNING: Running without calibration!${NC}"
echo -e "${YELLOW}   Depth values will NOT be accurate.${NC}"
echo -e "${YELLOW}   This is for testing and visualization only.${NC}"
echo ""
echo -e "${CYAN}======================================================================${NC}"
echo ""
echo -e "Controls:"
echo -e "  Q = Quit"
echo -e "  S = Save frame"
echo -e "  C = Change color map"
echo ""
echo -e "${CYAN}======================================================================${NC}"
echo ""
read -p "Press ENTER to start demo mode..."

# Run demo (V2 uses GStreamer subprocess)
python3 3_depth_sensing_demo_v2.py

echo ""
echo -e "${CYAN}======================================================================${NC}"
echo -e "For ACCURATE depth sensing, run proper calibration:"
echo -e "  ./depth_sensing.sh"
echo -e "  Then select Options 1 and 2"
echo -e "${CYAN}======================================================================${NC}"
echo ""
