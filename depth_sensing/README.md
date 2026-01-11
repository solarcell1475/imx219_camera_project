# IMX219 Stereo Depth Sensing Application

A complete depth sensing solution for IMX219 stereo cameras on Jetson platforms.

## 🎯 Overview

This application provides real-time depth sensing using stereo vision techniques. It includes:

- **Camera calibration** tools for accurate depth measurement
- **Real-time depth sensing** with multiple visualization options
- **Depth map export** for further processing
- **Adjustable parameters** for different scenarios

## 📋 Requirements

### Hardware
- Jetson platform (Orin Nano, Xavier NX, etc.)
- Waveshare IMX219-83 Stereo Camera (or similar dual IMX219 setup)
- Both cameras connected and detected

### Software
```bash
# Check if cameras are detected
ls /dev/video*

# Required packages (should already be installed)
sudo apt-get update
sudo apt-get install python3-opencv python3-numpy
```

### Calibration Pattern
- **Checkerboard pattern** with **9x6 internal corners** (10x7 squares)
- **Square size:** 25mm (or adjust in scripts)
- Print on flat, rigid surface
- Download: https://markhedleyjones.com/projects/calibration-checkerboard-collection

## 🚀 Quick Start

### Step 1: Capture Calibration Images (5-10 minutes)

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project/depth_sensing
python3 1_capture_calibration_images.py
```

**Instructions:**
1. Hold checkerboard pattern in front of cameras
2. Press **SPACE** when both cameras detect the pattern (green text)
3. Move checkerboard to different positions and angles
4. Capture **20-30 image pairs**
5. Cover entire field of view
6. Press **Q** when done

**Tips:**
- Use good lighting
- Keep checkerboard flat
- Vary distance (0.3m - 2m)
- Vary angles (tilted, rotated)
- Keep pattern visible in both cameras

### Step 2: Calibrate Cameras (2-3 minutes)

```bash
python3 2_calibrate_stereo_cameras.py
```

This will:
- Process captured images
- Calculate camera intrinsic parameters
- Calculate stereo extrinsic parameters
- Compute rectification maps
- Generate calibration report

**Output files:**
- `stereo_calibration.npz` - Calibration parameters
- `calibration_report.txt` - Quality assessment

**Quality guide:**
- **Excellent:** Reprojection error < 0.5 pixels
- **Good:** Reprojection error < 1.0 pixels
- **Fair:** Reprojection error < 2.0 pixels
- **Poor:** Reprojection error > 2.0 pixels (recalibrate!)

### Step 3: Run Depth Sensing

```bash
python3 3_depth_sensing.py
```

**Controls:**
- **Q** - Quit
- **S** - Save current frame
- **D** - Toggle display mode
- **C** - Cycle color maps
- **Mouse** - Hover to see depth at cursor

**Display modes:**
1. Side-by-side (left camera + depth map)
2. Full-screen depth map

**Advanced options:**
```bash
# High resolution (slower)
python3 3_depth_sensing.py --resolution 1920x1080

# Block Matching algorithm (faster, lower quality)
python3 3_depth_sensing.py --algorithm bm

# Save output to video
python3 3_depth_sensing.py --save-video depth_output.mp4

# Show FPS counter
python3 3_depth_sensing.py --fps
```

### Step 4: Visualize Saved Depth Maps (Optional)

```bash
# Visualize a saved depth map
python3 4_visualize_saved_depth.py depth_map_20260108_153000.npy
```

## 📊 Understanding Depth Maps

### Depth Calculation

Depth is calculated using stereo triangulation:

```
Depth = (Focal Length × Baseline) / Disparity
```

Where:
- **Focal Length**: Camera's focal length (from calibration)
- **Baseline**: Distance between cameras (~60mm for IMX219-83)
- **Disparity**: Pixel difference between left and right images

### Depth Range

Typical depth range for IMX219-83:
- **Minimum distance:** ~0.3m (closer objects may not match)
- **Maximum distance:** ~5-10m (depends on lighting and texture)
- **Optimal range:** 0.5m - 3m

### Factors Affecting Accuracy

✅ **Good conditions:**
- Textured surfaces (patterns, edges)
- Good lighting
- Perpendicular viewing angle
- Proper calibration

❌ **Poor conditions:**
- Smooth, featureless surfaces
- Low light
- Reflective or transparent objects
- Motion blur

## 🎨 Color Maps

Different color maps for depth visualization:

1. **JET** (default) - Blue (far) to red (near)
2. **TURBO** - Similar to JET but more perceptually uniform
3. **HOT** - Black → Red → Yellow → White
4. **RAINBOW** - Full spectrum
5. **VIRIDIS** - Perceptually uniform, colorblind-friendly

Press **C** to cycle through color maps.

## 🔧 Troubleshooting

### Cameras not detected
```bash
# Check camera status
ls /dev/video*
v4l2-ctl --list-devices

# Ensure cameras are activated
cd /home/jetson/Downloads/IMX219_Camera_Project
sudo ./activate_imx219_cameras.sh
sudo reboot
```

### Poor depth quality

**Problem:** Noisy or invalid depth values

**Solutions:**
1. **Improve calibration:**
   - Capture more calibration images
   - Use better lighting
   - Ensure checkerboard is flat
   
2. **Adjust stereo parameters:**
   Edit `3_depth_sensing.py` and modify:
   ```python
   self.num_disp = 128  # Increase for longer range
   self.block_size = 11  # Increase for smoother results
   ```

3. **Use better algorithm:**
   SGBM (Semi-Global Block Matching) is slower but produces better results than BM (Block Matching)

4. **Improve scene:**
   - Add lighting
   - Avoid smooth/reflective surfaces
   - Add texture to scene

### Calibration errors

**Problem:** "Pattern not found" during calibration

**Solutions:**
- Check checkerboard pattern size (9x6 internal corners)
- Improve lighting
- Reduce motion blur (hold steady)
- Ensure pattern is flat
- Move pattern closer to cameras

**Problem:** High reprojection error

**Solutions:**
- Recalibrate with more images
- Ensure cameras are stable (not moving)
- Use better quality checkerboard
- Check camera focus

### Performance issues

**Problem:** Low FPS

**Solutions:**
```bash
# Reduce resolution
python3 3_depth_sensing.py --resolution 640x480

# Use BM algorithm (faster)
python3 3_depth_sensing.py --algorithm bm

# Reduce num_disp in stereo matcher
# Edit 3_depth_sensing.py: self.num_disp = 64
```

## 📁 File Structure

```
depth_sensing/
├── 1_capture_calibration_images.py  # Step 1: Capture calibration images
├── 2_calibrate_stereo_cameras.py     # Step 2: Compute calibration
├── 3_depth_sensing.py                # Step 3: Real-time depth sensing
├── 4_visualize_saved_depth.py        # Step 4: Visualize saved maps
├── README.md                          # This file
├── USAGE_GUIDE.txt                    # Quick reference
├── calibration_images/                # Created by step 1
│   ├── left/                         # Left camera images
│   └── right/                        # Right camera images
├── stereo_calibration.npz            # Created by step 2
├── calibration_report.txt            # Created by step 2
├── depth_*.jpg                       # Saved frames
└── depth_map_*.npy                   # Saved depth data
```

## 🔬 Technical Details

### Stereo Algorithms

**SGBM (Semi-Global Block Matching):**
- Higher quality depth maps
- Better handling of textureless regions
- Slower processing (~10-15 FPS @ 1280x720)
- Default choice

**BM (Block Matching):**
- Faster processing (~20-30 FPS @ 1280x720)
- Lower quality, more noise
- Good for real-time applications

### Output Formats

**Depth Map (.npy):**
- NumPy array with shape (height, width, 3)
- Contains 3D coordinates (X, Y, Z) for each pixel
- Z coordinate = depth in millimeters
- Load with: `depth_map = np.load('depth_map_*.npy')`

**Disparity Image (.jpg):**
- Colorized visualization
- For display purposes only
- Does not contain actual depth values

## 🎓 Applications

### Robotics
- Obstacle detection and avoidance
- Navigation and path planning
- Object manipulation

### Autonomous Vehicles
- Distance measurement
- Collision avoidance
- Lane detection

### Computer Vision
- 3D reconstruction
- Scene understanding
- Object recognition with depth

### SLAM (Simultaneous Localization and Mapping)
- Visual odometry
- Environment mapping
- Robot localization

## 📚 Further Reading

- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)
- [Waveshare IMX219-83 Wiki](https://www.waveshare.net/wiki/IMX219-83_Stereo_Camera)
- [Camera Calibration Theory](https://docs.opencv.org/master/dc/dbb/tutorial_py_calibration.html)
- [Stereo Rectification](https://docs.opencv.org/master/d9/d0c/group__calib3d.html#ga617b1685d4059c6040827800e72ad2b6)

## 🤝 Support

For issues or questions:
1. Check this README
2. Review calibration report
3. Check camera detection: `ls /dev/video*`
4. Review kernel messages: `dmesg | grep imx219`

## 📝 License

This project is provided as-is for educational and research purposes.

---

**Author:** AI Development Assistant  
**Date:** 2026-01-08  
**Version:** 1.0
