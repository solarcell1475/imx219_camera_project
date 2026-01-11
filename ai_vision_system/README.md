# YOLO AI Vision System for Jetson Orin Nano

Real-time dual-camera object detection system using YOLO (YOLOv8/v9/v10/v11) on Jetson Orin Nano Super 8GB.

## Features

- **Dual Camera Support**: Real-time object detection on both IMX219 cameras
- **Latest YOLO Models**: Support for YOLOv8, YOLOv9, YOLOv10, **YOLO11** (default)
- **YOLO11n** (Default): Optimized for Jetson Orin Nano Super, ~50 FPS with TensorRT
- **Optimized Performance**: Parallel inference, resolution optimization, and efficient processing
- **Real-time Display**: Multiple view modes with FPS and performance metrics
- **Performance Monitoring**: Comprehensive metrics collection and alerting
- **Power Optimization**: MAXN power mode configuration and dynamic balancing
- **Modular Design**: Clean, organized codebase with separate modules

## Performance Optimizations

The system includes several performance optimizations for maximum FPS:

1. **Parallel Inference**: Dual cameras processed in parallel using threading (~2x speedup)
2. **Resolution Optimization**: Inference at 640x640 (square input for YOLO11), display at 1280x720 (3-4x faster inference)
3. **YOLO11n Model**: Default model optimized for Jetson Orin Nano Super
4. **TensorRT Support**: GPU-accelerated inference with TensorRT (2-5x faster than PyTorch)
5. **Direct Array Processing**: No temp file I/O overhead (PIL Image conversion)
6. **Efficient Detection Parsing**: Optimized Results object handling

**Expected Performance:**
- CPU-only (PyTorch): ~1 FPS, 300-400ms per inference
- GPU with PyTorch CUDA: 15-30+ FPS, 30-60ms per inference (10-30x speedup)
- **GPU with TensorRT (YOLO11n)**: **40-60 FPS**, ~20ms per inference (20-40x speedup)
- **Note:** TensorRT optimization provides best performance (see `GPU_ACCELERATION_SUCCESS.md`)

## Project Structure

```
ai_vision_system/
├── hardware/              # Hardware setup and drivers
├── firmware/              # Camera initialization and sync
├── ai_engine/             # YOLO inference engine
├── video_processing/      # Capture, preprocessing, postprocessing, display
├── monitoring/            # Performance monitoring and alerts
├── config/                # Configuration files
├── scripts/               # Utility scripts
└── main.py               # Main application
```

## Quick Start

### 1. Activate Cameras (First Time Setup)

If cameras are not working, activate camera overlays:

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project
sudo ./activate_imx219_cameras.sh
sudo reboot
```

**Note:** Reboot is required to activate camera drivers. This is software configuration only - no BIOS changes needed.

### 2. Set Up GPU Environment (Recommended for Best Performance)

To leverage the 67 TOPS GPU, install PyTorch with CUDA support:

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project/ai_vision_system

# Create and activate GPU environment
conda create -n yolo_gpu python=3.10 -y
conda activate yolo_gpu

# Install PyTorch with CUDA (see INSTALL_PYTORCH_JETSON.md for details)
# For JetPack 6, download PyTorch wheel from NVIDIA forums:
# https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048

# Install dependencies
pip install ultralytics opencv-python numpy pillow

# Verify GPU setup
python verify_gpu.py
```

**See `INSTALL_PYTORCH_JETSON.md` for detailed GPU setup instructions.**

### 3. Install Dependencies (If Not Using GPU Environment)

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project/ai_vision_system
pip3 install -r requirements.txt
```

### 4. Install YOLO

```bash
python3 ai_engine/yolo_setup/install_yolo.py
```

### 5. Verify Setup

```bash
# Verify GPU (if using GPU environment)
conda activate yolo_gpu
python verify_gpu.py

# Or verify general setup
python3 ai_engine/yolo_setup/verify_setup.py
```

### 6. Set MAXN Power Mode (Recommended)

```bash
sudo scripts/setup_maxn.sh
```

### 7. Run the System

**With GPU environment (recommended):**
```bash
conda activate yolo_gpu
python main.py
```

**Or with system Python:**
```bash
python3 main.py
```

**Note:** The system will automatically detect CUDA availability and use GPU if available. If CUDA is not available, it will fall back to CPU mode.

## Usage

### Basic Usage

```bash
# Run with default settings
python3 main.py

# Run with custom model
python3 main.py --model v11n  # YOLO11 Nano (default)
python3 main.py --model v11s  # YOLO11 Small
python3 main.py --model v11b  # YOLO11 Base/Balanced
python3 main.py --model v8s   # YOLOv8 Small

# Run with custom confidence threshold
python3 main.py --confidence 0.3

# Run with custom resolution
python3 main.py --resolution 1280x720
```

### Configuration File

Edit `config/settings.yaml` to customize:

- Model version and size
- Camera resolution and FPS
- Confidence and IoU thresholds
- Display view mode
- Performance targets

## Keyboard Controls

- **Q/ESC**: Quit
- **S**: Save current frames
- **D**: Toggle detections display
- **V**: Change view mode (side-by-side, overlay, split, single)
- **F**: Toggle FPS display
- **M**: Toggle metrics display
- **+/-**: Adjust confidence threshold
- **R**: Reset settings

## View Modes

- **Side-by-Side**: Both cameras with detections side by side
- **Overlay**: Single view with detections from both cameras
- **Split**: Original + detections
- **Single 0/1**: View single camera only

## Model Selection

Recommended models for Jetson Orin Nano Super 8GB:

### YOLO11 Models (Default - Recommended)
- **YOLO11n** (Nano): Fastest, ~50 FPS with TensorRT FP16, optimized for Jetson (Default)
- **YOLO11s** (Small): Balanced, ~35 FPS with TensorRT, good accuracy
- **YOLO11m** (Medium): Higher accuracy, ~20 FPS with TensorRT
- **YOLO11b** (Base/Balanced): Between nano and small, ~30 FPS with TensorRT
- **YOLO11l/x**: Highest accuracy, slower

### YOLOv8 Models (Legacy)
- **YOLOv8n** (Nano): 3-5 FPS CPU / 15-30 FPS GPU, lower accuracy
- **YOLOv8s** (Small): Balanced, 2-3 FPS CPU / 10-20 FPS GPU, good accuracy
- **YOLOv8m** (Medium): Higher accuracy, 1-2 FPS CPU / 8-15 FPS GPU
- **YOLOv8l/x**: Highest accuracy, slower

**Note:** YOLO11 provides significantly better performance than YOLOv8 on Jetson Orin Nano Super. Use TensorRT optimization for best results (see `GPU_ACCELERATION_SUCCESS.md`).

## Performance Targets

- **Display Resolution**: 1280x720 @ 30fps capture
- **Inference Resolution**: 640x640 (square input optimized for YOLO11)
- **Inference Latency**: ~200-400ms per cycle (CPU) or ~20-40ms (GPU with TensorRT)
- **FPS Target**: 3-5 FPS (CPU) or **40-60 FPS (GPU with YOLO11n + TensorRT)**
- **Memory Usage**: < 6GB

## YOLO11 on Jetson Orin Nano Super

The system now defaults to **YOLO11n** which is optimized for Jetson Orin Nano Super:
- **Performance**: ~50 FPS with TensorRT FP16 (vs ~40 FPS for YOLOv8n)
- **Optimization**: Use ONNX → TensorRT conversion for best performance
- **Reference**: [Ultralytics YOLO11 on Jetson Orin Nano Super](https://www.ultralytics.com/blog/ultralytics-yolo11-on-nvidia-jetson-orin-nano-super-fast-and-efficient)

### Converting YOLO11 to TensorRT

For maximum performance, convert YOLO11 to TensorRT:

```bash
# Step 1: Convert YOLO11 to ONNX (CPU-only, one-time)
python scripts/convert_to_onnx.py --model yolov11n.pt --size 640,640

# Step 2: Convert ONNX to TensorRT (GPU-accelerated)
python scripts/onnx_to_tensorrt.py --onnx yolov11n.onnx --precision fp16 --size 640,640

# Step 3: Use TensorRT engine for inference
# (Update settings.yaml to use TensorRT engine path)
```

## Monitoring

The system includes comprehensive performance monitoring:

- CPU/GPU utilization
- Memory usage (RAM and GPU)
- Inference latency
- FPS tracking
- Temperature monitoring
- Performance alerts

View metrics in real-time on the display or check logs in `logs/` directory.

## Troubleshooting

### Cameras Not Detected or Showing "Noise"

**Symptoms:** Display shows green screen, noise, or "Camera Not Available"

**Solution:**
1. Verify cameras are detected:
   ```bash
   python3 test_imx219_cameras.py
   ```

2. If cameras exist but can't read frames, activate camera overlays:
   ```bash
   cd /home/jetson/Downloads/IMX219_Camera_Project
   sudo ./activate_imx219_cameras.sh
   sudo reboot
   ```

3. After reboot, verify cameras work:
   ```bash
   nvgstcapture-1.0 --sensor-id=0 --image-res=3
   nvgstcapture-1.0 --sensor-id=1 --image-res=3
   ```

**Why reboot is needed:** Camera overlay configuration is loaded during boot. No BIOS changes required - this is pure software configuration.

### Detection Boxes Not Showing

**Symptoms:** Video displays but no bounding boxes appear

**Solution:**
1. Check if detections are being found (look for "Detections: Camera 0: X, Camera 1: Y" in console)
2. Lower confidence threshold in `config/settings.yaml` (try 0.15)
3. Ensure objects are in view (person, chair, cup, etc.)
4. Verify YOLO model loaded correctly (check console output)

### Low FPS / GPU Not Being Used

**Current Performance:**
- CPU-only: ~1 FPS, 300-400ms per inference
- With GPU: 15-30+ FPS, 30-60ms per inference (expected)

**If GPU is not being used:**

1. **Verify GPU setup:**
   ```bash
   conda activate yolo_gpu
   python verify_gpu.py
   ```

2. **Check PyTorch CUDA availability:**
   ```bash
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```
   Should show `CUDA available: True`

3. **Install PyTorch with CUDA support:**
   - See `INSTALL_PYTORCH_JETSON.md` for detailed instructions
   - For JetPack 6, download PyTorch wheel from NVIDIA forums
   - Run `bash scripts/setup_pytorch_libs.sh` to configure libraries

4. **Ensure you're using the GPU environment:**
   ```bash
   conda activate yolo_gpu
   python main.py
   ```

5. **Other optimizations (already applied):**
   - YOLOv8n model (default) - smallest and fastest
   - Inference resolution optimized to 640x480
   - Parallel inference enabled
   - MAXN power mode recommended: `sudo scripts/setup_maxn.sh`

### Out of Memory

1. Use YOLOv8n model (smallest)
2. Reduce display resolution in `config/settings.yaml`
3. Close other applications
4. Check memory usage in monitoring dashboard

### System Hangs or Freezes

**Symptoms:** Program stops responding, no updates

**Solution:**
1. Check if inference is completing (look for FPS updates)
2. Verify cameras are streaming (check capture statistics)
3. Reduce inference resolution further if needed
4. Check system logs: `tail -f logs/errors.log`

## Development

### Testing Individual Components

```bash
# Test camera initialization
python3 firmware/camera_init/dual_camera.py

# Test model loading
python3 ai_engine/model_manager/yolo_model_loader.py

# Test hardware
python3 hardware/testing/hardware_test_suite.py
```

### Adding Custom Models

1. Train your YOLO model using Ultralytics
2. Save model as `.pt` file
3. Load using model manager:

```python
from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
manager = YOLOModelManager()
model = manager.load_model(custom_model_path='path/to/model.pt')
```

## License

This project is part of the IMX219 Camera Project.

## Support

For issues or questions:
1. Check logs in `logs/` directory
2. Review hardware setup documentation
3. Verify YOLO installation with `verify_setup.py`
