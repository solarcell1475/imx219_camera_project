# How to Start the YOLO AI Vision System

## Quick Start Guide

### Step 1: Navigate to Project Directory

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project/ai_vision_system
```

### Step 2: Verify Dependencies

Make sure all dependencies are installed:

```bash
pip3 install -r requirements.txt
```

### Step 3: Verify YOLO Setup

Check if YOLO is properly installed:

```bash
python3 ai_engine/yolo_setup/verify_setup.py
```

### Step 4: Set MAXN Power Mode (Recommended)

For best performance, set Jetson to maximum power mode:

```bash
sudo scripts/setup_maxn.sh
```

### Step 5: Start the Program

**Basic usage:**

```bash
python3 main.py
```

**With options:**

```bash
# Use smaller model for faster performance
python3 main.py --model v8n

# Use custom confidence threshold
python3 main.py --confidence 0.3

# Use custom resolution
python3 main.py --resolution 1280x720

# Use configuration file
python3 main.py --config config/settings.yaml
```

## Command Line Options

```bash
python3 main.py [OPTIONS]

Options:
  --config FILE       Configuration file path (YAML)
  --model MODEL       Model version and size (e.g., v8s, v8n, v8m)
  --confidence FLOAT  Confidence threshold (0.0-1.0, default: 0.25)
  --resolution WxH    Resolution (e.g., 1280x720, default: 1280x720)
```

## Keyboard Controls (During Runtime)

- **Q** or **ESC** - Quit the program
- **S** - Save current frames to disk
- **D** - Toggle detections display on/off
- **V** - Change view mode (side-by-side, overlay, split, single)
- **F** - Toggle FPS counter
- **M** - Toggle metrics display
- **+** or **=** - Increase confidence threshold
- **-** or **_ ** - Decrease confidence threshold
- **R** - Reset settings

## Troubleshooting

### Cameras Not Detected

```bash
# Check camera detection
python3 hardware/installation/camera_detection.py

# If cameras not found, activate them
cd ..
sudo ./activate_imx219_cameras.sh
sudo reboot
```

### CUDA Not Available

If you see "CUDA Available: False", you need to install CUDA-enabled PyTorch:

```bash
# See installation guide
cat INSTALL_PYTORCH_JETSON.md
```

### Low Performance

1. Use smaller model: `--model v8n`
2. Reduce resolution: `--resolution 640x480`
3. Ensure MAXN mode: `sudo scripts/setup_maxn.sh`
4. Increase confidence threshold: `--confidence 0.4`

## Example Commands

```bash
# Fast mode (nano model, lower resolution)
python3 main.py --model v8n --resolution 640x480 --confidence 0.3

# High quality mode (small model, full HD)
python3 main.py --model v8s --resolution 1920x1080 --confidence 0.25

# Balanced mode (default)
python3 main.py
```

## Configuration File

Edit `config/settings.yaml` to customize:

- Model version and size
- Camera resolution and FPS
- Confidence and IoU thresholds
- Display view mode
- Performance targets

Then run:

```bash
python3 main.py --config config/settings.yaml
```

## Testing First

Before running with cameras, test the system:

```bash
# Run loopback test (uses test images)
python3 test_loopback.py --iterations 5

# Test with real cameras
python3 test_loopback.py --iterations 5 --cameras
```

## What to Expect

When you start the program:

1. System initializes cameras
2. YOLO model loads (first time downloads model)
3. Real-time video window opens
4. Detections appear as bounding boxes
5. FPS and metrics display on screen
6. Press Q to quit

## Performance Tips

- **First run**: Model download may take time
- **CPU mode**: Slower (~0.5-1 FPS)
- **CUDA mode**: Much faster (20-30+ FPS)
- **Recommended**: YOLOv8s model for balanced performance
