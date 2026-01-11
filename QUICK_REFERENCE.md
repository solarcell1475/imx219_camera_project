# IMX219-83 Stereo Camera - Quick Reference

## 🚀 Quick Start (First Time Setup)

### Step 1: Activate Cameras
```bash
cd /home/jetson
sudo ./activate_imx219_cameras.sh
```

### Step 2: Reboot
```bash
sudo reboot
```

### Step 3: Verify
```bash
cd /home/jetson
./test_imx219_cameras.sh
```

---

## 📹 Quick Test Commands

### Check if Cameras are Detected
```bash
ls /dev/video*
# Should show: /dev/video0 and /dev/video1
```

### GStreamer - View Camera 0
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=0 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink
```

### GStreamer - View Camera 1
```bash
gst-launch-1.0 nvarguscamerasrc sensor-id=1 ! 'video/x-raw(memory:NVMM),width=1920,height=1080,framerate=30/1' ! nvvidconv ! xvimagesink
```

### Python - Check Both Cameras
```bash
python3 imx219_camera_test.py check
```

### Python - View Stereo (Both Cameras)
```bash
python3 imx219_camera_test.py stereo
```

---

## 🔧 Common Commands

### Capture Image (High Resolution)
```bash
# Camera 0
gst-launch-1.0 nvarguscamerasrc sensor-id=0 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=3280,height=2464' ! nvjpegenc ! filesink location=cam0.jpg

# Camera 1
gst-launch-1.0 nvarguscamerasrc sensor-id=1 num-buffers=1 ! 'video/x-raw(memory:NVMM),width=3280,height=2464' ! nvjpegenc ! filesink location=cam1.jpg
```

### List Camera Information
```bash
v4l2-ctl --list-devices
v4l2-ctl --device=/dev/video0 --all
```

### Check Kernel Logs
```bash
dmesg | grep -i imx219
```

---

## 🐛 Troubleshooting

### No Video Devices Found
```bash
# Check boot config
cat /boot/extlinux/extlinux.conf | grep DEFAULT
# Should show: DEFAULT JetsonIO

# If not, run activation script again
sudo ./activate_imx219_cameras.sh
sudo reboot
```

### Camera Not Working
```bash
# Check kernel messages
dmesg | grep -i imx219
dmesg | grep -i csi

# Check I2C
i2cdetect -l

# Verify overlay loaded
cat /boot/extlinux/extlinux.conf | grep -A 5 JetsonIO
```

### Permission Denied
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Then logout and login again
```

---

## 📊 Supported Resolutions

| Resolution | Max FPS | Use Case |
|------------|---------|----------|
| 3280x2464 | 21 fps | Full resolution, photos |
| 1920x1080 | 30 fps | HD video |
| 1640x1232 | 30 fps | Good balance |
| 1280x720  | 60 fps | High framerate |

---

## 📁 File Locations

- **Activation Script:** `/home/jetson/activate_imx219_cameras.sh`
- **Test Script:** `/home/jetson/test_imx219_cameras.sh`
- **Python Script:** `/home/jetson/imx219_camera_test.py`
- **Full Documentation:** `/home/jetson/IMX219_CAMERA_SETUP.md`
- **Development Log:** `/home/jetson/Downloads/IMX219_Camera_Project/DEVELOPMENT_LOG.md`

---

## 💡 Useful Keyboard Shortcuts

When viewing camera feed:
- **'q'** - Quit viewing
- **'s'** - Save snapshot (Python script)
- **Ctrl+C** - Stop GStreamer pipeline

---

## 📚 Full Documentation

For detailed information, see:
- `IMX219_CAMERA_SETUP.md` - Complete setup guide
- `DEVELOPMENT_LOG.md` - Development progress and errors
- Waveshare Wiki: https://www.waveshare.net/wiki/IMX219-83_Stereo_Camera

---

**Quick Help:** `python3 imx219_camera_test.py` (shows usage)
