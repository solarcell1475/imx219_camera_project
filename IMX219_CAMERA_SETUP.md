# IMX219-83 Stereo Camera Setup for Jetson Orin Nano

## Overview
This guide helps you set up and test the Waveshare IMX219-83 Stereo Camera on your Jetson Orin Nano.

**Product:** Waveshare IMX219-83 Stereo Camera  
**Documentation:** https://www.waveshare.net/wiki/IMX219-83_Stereo_Camera

## Camera Specifications
- **Sensor:** Sony IMX219 (x2)
- **Resolution:** 8MP per camera (3280 x 2464)
- **Field of View:** 83 degrees
- **Interface:** MIPI CSI-2
- **Baseline Distance:** ~60mm (stereo baseline)
- **Frame Rate:** Up to 30fps @ 1080p

## Hardware Connection
1. **Power off** your Jetson Orin Nano completely
2. Connect the stereo camera to the **CSI connector(s)** on your Jetson Orin Nano
   - The dual camera module uses a splitter to connect to both CAM0 and CAM1 ports
3. Ensure the ribbon cable is properly seated with the blue side facing the correct direction
4. Power on your Jetson

## Software Setup

### Step 1: Activate Camera Overlay

Your Jetson Orin Nano already has the IMX219 dual camera device tree overlay installed. You just need to activate it:

```bash
sudo ./activate_imx219_cameras.sh
```

This script will:
- Backup your current boot configuration
- Enable the dual IMX219 camera overlay
- Prompt you to reboot

### Step 2: Reboot

After running the activation script, **reboot your system**:

```bash
sudo reboot
```

### Step 3: Verify Camera Detection

After rebooting, check if cameras are detected:

```bash
./test_imx219_cameras.sh
```

Or manually check:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

You should see video devices like `/dev/video0` and `/dev/video1`.

## Testing Cameras

### Method 1: Using GStreamer (Native Jetson)

**Test Camera 0 (Left):**
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! xvimagesink
```

**Test Camera 1 (Right):**
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! \
  'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! \
  nvvidconv ! xvimagesink
```

**Capture a high-resolution image:**
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! \
  'video/x-raw(memory:NVMM),width=3280,height=2464' ! \
  nvjpegenc ! filesink location=camera0.jpg
```

**Display both cameras side by side:**
```bash
gst-launch-1.0 \
  nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=960,height=540' ! \
  nvvidconv ! 'video/x-raw,width=960,height=540' ! videobox left=0 ! comp. \
  nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=960,height=540' ! \
  nvvidconv ! 'video/x-raw,width=960,height=540' ! videobox left=-960 ! comp. \
  compositor name=comp ! xvimagesink
```

### Method 2: Using Python Script

**Check camera availability:**
```bash
python3 imx219_camera_test.py check
```

**View live feed from Camera 0:**
```bash
python3 imx219_camera_test.py view 0
```

**View live feed from Camera 1:**
```bash
python3 imx219_camera_test.py view 1
```

**View stereo cameras side by side:**
```bash
python3 imx219_camera_test.py stereo
```

**Capture a single image:**
```bash
python3 imx219_camera_test.py capture 0
```

### Method 3: Using OpenCV (Python)

```python
import cv2

# Open camera 0
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if ret:
        cv2.imshow('Camera', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

## Troubleshooting

### Cameras Not Detected

1. **Check boot configuration:**
   ```bash
   cat /boot/extlinux/extlinux.conf | grep DEFAULT
   ```
   Should show `DEFAULT JetsonIO`

2. **Verify overlay is loaded:**
   ```bash
   cat /boot/extlinux/extlinux.conf | grep -A 5 "LABEL JetsonIO"
   ```
   Should show the IMX219 dual overlay

3. **Check kernel messages:**
   ```bash
   dmesg | grep -i imx219
   ```

4. **Check I2C detection:**
   ```bash
   i2cdetect -l
   ```

5. **Verify hardware connection:**
   - Power off completely
   - Reseat the camera ribbon cables
   - Check for any damage to the cables
   - Ensure the blue side of the ribbon faces the correct direction

### Poor Image Quality

- Check if the lens protective film has been removed
- Adjust camera focus if needed (some models have adjustable focus)
- Ensure adequate lighting
- Try different resolution/framerate settings

### Permission Issues

If you get permission errors accessing cameras:
```bash
sudo usermod -a -G video $USER
# Then log out and log back in
```

## Advanced Usage

### Stereo Vision Applications

The IMX219-83 Stereo Camera is ideal for:
- **Depth estimation**
- **3D reconstruction**
- **Obstacle detection**
- **SLAM (Simultaneous Localization and Mapping)**
- **Autonomous navigation**

### Camera Calibration

For stereo vision applications, you'll need to calibrate the cameras:

1. Use OpenCV's calibration tools
2. Capture checkerboard patterns from both cameras
3. Calculate intrinsic and extrinsic parameters
4. Compute stereo rectification maps

Example calibration resources:
- OpenCV: `cv2.stereoCalibrate()`
- ROS: `camera_calibration` package

### Integration with ROS

If using ROS:

```bash
# Install ROS camera drivers
sudo apt-get install ros-<distro>-image-transport-plugins

# Use gscam or v4l2_camera nodes
```

## Camera Modes and Capabilities

### Supported Resolutions

The IMX219 sensor supports various modes:
- 3280 x 2464 @ 21 fps (full resolution)
- 1920 x 1080 @ 30 fps (1080p)
- 1640 x 1232 @ 30 fps
- 1280 x 720 @ 60 fps (720p, high framerate)

### Setting Custom Resolutions

With GStreamer:
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! \
  'video/x-raw(memory:NVMM),width=1640,height=1232,framerate=30/1' ! \
  nvvidconv ! xvimagesink
```

## Scripts Provided

1. **activate_imx219_cameras.sh** - Enables camera overlay in boot configuration
2. **test_imx219_cameras.sh** - Tests camera detection and provides example commands
3. **imx219_camera_test.py** - Python script for various camera tests and live viewing

## Additional Resources

- [Waveshare Wiki](https://www.waveshare.net/wiki/IMX219-83_Stereo_Camera)
- [NVIDIA Jetson Camera Documentation](https://docs.nvidia.com/jetson/archives/r36.4/DeveloperGuide/SD/CameraDevelopment.html)
- [OpenCV Stereo Vision Tutorial](https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html)

## Support

If you encounter issues:
1. Check the Waveshare product wiki
2. Review Jetson developer forums
3. Verify your JetPack version is up to date
4. Check kernel messages: `dmesg | grep -i imx219`

## Notes

- The camera driver is included in the standard Jetson Linux kernel
- No additional kernel modules need to be compiled
- The device tree overlay handles camera configuration
- Make sure your JetPack version supports the IMX219 sensor (R36+ recommended)

---

**Last Updated:** 2026-01-08  
**Tested On:** Jetson Orin Nano, JetPack R36.4.7
