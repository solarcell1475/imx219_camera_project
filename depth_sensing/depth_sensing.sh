#!/bin/bash
#
# IMX219 Depth Sensing Application Launcher
# ==========================================
# Interactive menu to run depth sensing project steps
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Project directory
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Banner
show_banner() {
    clear
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}           IMX219 STEREO DEPTH SENSING APPLICATION${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
}

# Check if cameras are detected
check_cameras() {
    echo -e "${BLUE}Checking cameras...${NC}"
    if [ -e /dev/video0 ] && [ -e /dev/video1 ]; then
        echo -e "${GREEN}✓ Both cameras detected (/dev/video0, /dev/video1)${NC}"
        return 0
    else
        echo -e "${RED}✗ Cameras not detected!${NC}"
        echo -e "${YELLOW}Run camera activation script first:${NC}"
        echo -e "  cd /home/jetson/Downloads/IMX219_Camera_Project"
        echo -e "  sudo ./activate_imx219_cameras.sh"
        echo -e "  sudo reboot"
        return 1
    fi
}

# Check if calibration exists
check_calibration() {
    if [ -f "stereo_calibration.npz" ]; then
        echo -e "${GREEN}✓ Calibration file found${NC}"
        if [ -f "calibration_report.txt" ]; then
            echo -e "${BLUE}Calibration quality:${NC}"
            grep "Overall Quality:" calibration_report.txt | head -1
        fi
        return 0
    else
        echo -e "${YELLOW}✗ Calibration not found${NC}"
        echo -e "  Run calibration first (Option 1 and 2)"
        return 1
    fi
}

# Menu options
show_menu() {
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo -e "${CYAN}                            MAIN MENU${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${GREEN}CALIBRATION:${NC}"
    echo -e "  ${YELLOW}1${NC}) Capture calibration images"
    echo -e "  ${YELLOW}2${NC}) Calibrate stereo cameras"
    echo -e "  ${YELLOW}3${NC}) View calibration report"
    echo ""
    echo -e "${GREEN}DEPTH SENSING:${NC}"
    echo -e "  ${YELLOW}4${NC}) Run depth sensing (default - 1280x720, SGBM)"
    echo -e "  ${YELLOW}5${NC}) Run depth sensing (high quality - 1920x1080, SGBM)"
    echo -e "  ${YELLOW}6${NC}) Run depth sensing (fast - 640x480, BM)"
    echo -e "  ${YELLOW}7${NC}) Run depth sensing (custom parameters)"
    echo ""
    echo -e "${GREEN}VISUALIZATION:${NC}"
    echo -e "  ${YELLOW}8${NC}) Visualize saved depth map"
    echo ""
    echo -e "${GREEN}INFORMATION:${NC}"
    echo -e "  ${YELLOW}9${NC}) Show camera status"
    echo -e "  ${YELLOW}h${NC}) Show help/documentation"
    echo -e "  ${YELLOW}q${NC}) Quit"
    echo ""
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
}

# Step 1: Capture calibration images
capture_calibration() {
    show_banner
    echo -e "${CYAN}STEP 1: CAPTURE CALIBRATION IMAGES${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${YELLOW}Instructions:${NC}"
    echo -e "  • Hold checkerboard (9x6 internal corners) in front of cameras"
    echo -e "  • Press SPACE when pattern detected (green ✓)"
    echo -e "  • Capture 20-30 images from different angles and distances"
    echo -e "  • Press Q when done"
    echo ""
    read -p "Press ENTER to start capturing..."
    echo ""
    
    python3 1_capture_calibration_images.py
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 2: Calibrate cameras
calibrate_cameras() {
    show_banner
    echo -e "${CYAN}STEP 2: CALIBRATE STEREO CAMERAS${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    if [ ! -d "calibration_images/left" ] || [ ! -d "calibration_images/right" ]; then
        echo -e "${RED}ERROR: No calibration images found!${NC}"
        echo -e "Run ${YELLOW}Option 1${NC} first to capture calibration images."
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo -e "Processing calibration images..."
    echo ""
    
    python3 2_calibrate_stereo_cameras.py
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 3: View calibration report
view_calibration_report() {
    show_banner
    echo -e "${CYAN}CALIBRATION REPORT${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    if [ -f "calibration_report.txt" ]; then
        cat calibration_report.txt
    else
        echo -e "${RED}ERROR: Calibration report not found!${NC}"
        echo -e "Run calibration first (Options 1 and 2)."
    fi
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 4: Run depth sensing (default)
run_depth_sensing_default() {
    show_banner
    echo -e "${CYAN}DEPTH SENSING - DEFAULT MODE${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Resolution: 1280x720"
    echo -e "  Algorithm: SGBM (high quality)"
    echo -e "  FPS: Enabled"
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Q = Quit  |  S = Save  |  D = Display mode  |  C = Color map"
    echo ""
    
    if ! check_calibration; then
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo ""
    read -p "Press ENTER to start depth sensing..."
    echo ""
    
    python3 3_depth_sensing.py --resolution 1280x720 --algorithm sgbm --fps
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 5: Run depth sensing (high quality)
run_depth_sensing_hq() {
    show_banner
    echo -e "${CYAN}DEPTH SENSING - HIGH QUALITY MODE${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Resolution: 1920x1080 (Full HD)"
    echo -e "  Algorithm: SGBM (high quality)"
    echo -e "  FPS: Enabled"
    echo -e "  ${YELLOW}Note: Lower FPS due to high resolution${NC}"
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Q = Quit  |  S = Save  |  D = Display mode  |  C = Color map"
    echo ""
    
    if ! check_calibration; then
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo ""
    read -p "Press ENTER to start depth sensing..."
    echo ""
    
    python3 3_depth_sensing.py --resolution 1920x1080 --algorithm sgbm --fps
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 6: Run depth sensing (fast)
run_depth_sensing_fast() {
    show_banner
    echo -e "${CYAN}DEPTH SENSING - FAST MODE${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Resolution: 640x480"
    echo -e "  Algorithm: BM (fast, lower quality)"
    echo -e "  FPS: Enabled"
    echo -e "  ${YELLOW}Note: Higher FPS but lower quality${NC}"
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Q = Quit  |  S = Save  |  D = Display mode  |  C = Color map"
    echo ""
    
    if ! check_calibration; then
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo ""
    read -p "Press ENTER to start depth sensing..."
    echo ""
    
    python3 3_depth_sensing.py --resolution 640x480 --algorithm bm --fps
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 7: Run depth sensing (custom)
run_depth_sensing_custom() {
    show_banner
    echo -e "${CYAN}DEPTH SENSING - CUSTOM MODE${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    if ! check_calibration; then
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    # Get resolution
    echo -e "${YELLOW}Select resolution:${NC}"
    echo "  1) 640x480   (fast)"
    echo "  2) 1280x720  (balanced)"
    echo "  3) 1920x1080 (high quality)"
    read -p "Choice [2]: " res_choice
    res_choice=${res_choice:-2}
    
    case $res_choice in
        1) resolution="640x480" ;;
        2) resolution="1280x720" ;;
        3) resolution="1920x1080" ;;
        *) resolution="1280x720" ;;
    esac
    
    # Get algorithm
    echo ""
    echo -e "${YELLOW}Select algorithm:${NC}"
    echo "  1) SGBM (high quality, slower)"
    echo "  2) BM   (fast, lower quality)"
    read -p "Choice [1]: " algo_choice
    algo_choice=${algo_choice:-1}
    
    case $algo_choice in
        1) algorithm="sgbm" ;;
        2) algorithm="bm" ;;
        *) algorithm="sgbm" ;;
    esac
    
    # Save video option
    echo ""
    read -p "Save video? (y/N): " save_video
    video_arg=""
    if [[ "$save_video" =~ ^[Yy]$ ]]; then
        timestamp=$(date +%Y%m%d_%H%M%S)
        video_file="depth_output_${timestamp}.mp4"
        video_arg="--save-video $video_file"
        echo -e "${GREEN}Will save to: $video_file${NC}"
    fi
    
    # Show configuration
    echo ""
    echo -e "${BLUE}Configuration:${NC}"
    echo -e "  Resolution: $resolution"
    echo -e "  Algorithm: ${algorithm^^}"
    echo -e "  FPS: Enabled"
    if [ -n "$video_arg" ]; then
        echo -e "  Video: $video_file"
    fi
    
    echo ""
    echo -e "${YELLOW}Controls:${NC}"
    echo -e "  Q = Quit  |  S = Save  |  D = Display mode  |  C = Color map"
    echo ""
    read -p "Press ENTER to start depth sensing..."
    echo ""
    
    python3 3_depth_sensing.py --resolution $resolution --algorithm $algorithm --fps $video_arg
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 8: Visualize saved depth map
visualize_depth_map() {
    show_banner
    echo -e "${CYAN}VISUALIZE SAVED DEPTH MAP${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    # Find depth map files
    depth_files=(depth_map_*.npy)
    
    if [ ! -e "${depth_files[0]}" ]; then
        echo -e "${RED}No saved depth maps found!${NC}"
        echo -e "Save frames during depth sensing (press S) first."
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo -e "${BLUE}Available depth maps:${NC}"
    echo ""
    
    # List files with numbers
    i=1
    for file in "${depth_files[@]}"; do
        if [ -f "$file" ]; then
            size=$(ls -lh "$file" | awk '{print $5}')
            date=$(stat -c %y "$file" | cut -d'.' -f1)
            echo "  $i) $file ($size) - $date"
            ((i++))
        fi
    done
    
    echo ""
    read -p "Select file number [1]: " file_choice
    file_choice=${file_choice:-1}
    
    # Get selected file
    selected_file="${depth_files[$((file_choice-1))]}"
    
    if [ ! -f "$selected_file" ]; then
        echo -e "${RED}Invalid selection!${NC}"
        echo ""
        read -p "Press ENTER to return to menu..."
        return
    fi
    
    echo ""
    echo -e "${BLUE}Visualizing: $selected_file${NC}"
    echo -e "${YELLOW}Press keys 1-5 to change colormap, Q to quit${NC}"
    echo ""
    read -p "Press ENTER to start..."
    echo ""
    
    python3 4_visualize_saved_depth.py "$selected_file"
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Step 9: Show camera status
show_camera_status() {
    show_banner
    echo -e "${CYAN}CAMERA STATUS${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    # Check video devices
    echo -e "${BLUE}Video Devices:${NC}"
    if ls /dev/video* &>/dev/null; then
        ls -l /dev/video*
    else
        echo -e "${RED}No video devices found!${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}V4L2 Devices:${NC}"
    v4l2-ctl --list-devices 2>/dev/null || echo "v4l2-ctl not available"
    
    echo ""
    echo -e "${BLUE}Camera Information:${NC}"
    if [ -e /dev/video0 ]; then
        echo ""
        echo "Camera 0:"
        v4l2-ctl -d /dev/video0 --info 2>/dev/null | head -10 || echo "Cannot get info"
    fi
    
    if [ -e /dev/video1 ]; then
        echo ""
        echo "Camera 1:"
        v4l2-ctl -d /dev/video1 --info 2>/dev/null | head -10 || echo "Cannot get info"
    fi
    
    echo ""
    read -p "Press ENTER to return to menu..."
}

# Show help
show_help() {
    show_banner
    echo -e "${CYAN}HELP & DOCUMENTATION${NC}"
    echo -e "${CYAN}======================================================================${NC}"
    echo ""
    
    echo -e "${GREEN}WORKFLOW:${NC}"
    echo ""
    echo -e "  ${YELLOW}1. Calibration (First Time Setup)${NC}"
    echo -e "     a) Capture calibration images (Option 1)"
    echo -e "     b) Calibrate cameras (Option 2)"
    echo -e "     c) View calibration report (Option 3)"
    echo ""
    echo -e "  ${YELLOW}2. Depth Sensing (After Calibration)${NC}"
    echo -e "     Run depth sensing (Options 4-7)"
    echo ""
    echo -e "  ${YELLOW}3. Analysis (Optional)${NC}"
    echo -e "     Visualize saved depth maps (Option 8)"
    echo ""
    
    echo -e "${GREEN}CALIBRATION TIPS:${NC}"
    echo -e "  • Use 9x6 checkerboard (10x7 squares, 25mm each)"
    echo -e "  • Capture 20-30 images from various angles"
    echo -e "  • Use good lighting"
    echo -e "  • Keep checkerboard flat"
    echo -e "  • Pattern must be visible in both cameras"
    echo ""
    
    echo -e "${GREEN}DEPTH SENSING MODES:${NC}"
    echo -e "  • ${YELLOW}Default${NC}: 1280x720, SGBM - Balanced quality/speed"
    echo -e "  • ${YELLOW}High Quality${NC}: 1920x1080, SGBM - Best quality, slower"
    echo -e "  • ${YELLOW}Fast${NC}: 640x480, BM - Real-time, lower quality"
    echo -e "  • ${YELLOW}Custom${NC}: Your own parameters"
    echo ""
    
    echo -e "${GREEN}DOCUMENTATION:${NC}"
    echo -e "  • README.md - Complete documentation"
    echo -e "  • USAGE_GUIDE.txt - Quick reference"
    echo -e "  • calibration_report.txt - Calibration quality"
    echo ""
    
    echo -e "${GREEN}FILES:${NC}"
    echo -e "  Location: $PROJECT_DIR"
    echo -e "  Scripts: 1_*.py, 2_*.py, 3_*.py, 4_*.py"
    echo -e "  Calibration: stereo_calibration.npz"
    echo -e "  Output: depth_*.jpg, depth_map_*.npy"
    echo ""
    
    read -p "Press ENTER to return to menu..."
}

# Main loop
main() {
    while true; do
        show_banner
        check_cameras
        echo ""
        check_calibration
        show_menu
        
        read -p "Select option: " choice
        
        case $choice in
            1) capture_calibration ;;
            2) calibrate_cameras ;;
            3) view_calibration_report ;;
            4) run_depth_sensing_default ;;
            5) run_depth_sensing_hq ;;
            6) run_depth_sensing_fast ;;
            7) run_depth_sensing_custom ;;
            8) visualize_depth_map ;;
            9) show_camera_status ;;
            h|H) show_help ;;
            q|Q) 
                echo ""
                echo -e "${GREEN}Thank you for using IMX219 Depth Sensing!${NC}"
                echo ""
                exit 0
                ;;
            *)
                echo -e "${RED}Invalid option!${NC}"
                sleep 1
                ;;
        esac
    done
}

# Run main
main
