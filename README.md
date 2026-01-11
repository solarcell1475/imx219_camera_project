# IMX219-83 Stereo Camera Project

**Project Folder:** `/home/jetson/Downloads/IMX219_Camera_Project/`  
**Created:** 2026-01-08  
**Camera:** Waveshare IMX219-83 Stereo Camera  
**Platform:** Jetson Orin Nano

---

## 📂 Project Structure

```
IMX219_Camera_Project/
├── README.md                       # This file - Project overview
├── DEVELOPMENT_LOG.md              # Detailed development progress and errors
├── QUICK_REFERENCE.md              # Quick commands and shortcuts
├── IMX219_CAMERA_SETUP.md          # Complete setup documentation
├── activate_imx219_cameras.sh      # Script to enable cameras (run with sudo)
├── test_imx219_cameras.sh          # Camera detection and testing script
└── imx219_camera_test.py           # Python camera testing utility
```

---

## 🎯 Purpose

This project documents the setup, configuration, and testing of the Waveshare IMX219-83 Stereo Camera on the Jetson Orin Nano platform. The goal is to activate both cameras for stereo vision applications.

---

## 🚀 Getting Started

### First Time Setup (3 Steps)

1. **Activate the cameras:**
   ```bash
   cd /home/jetson
   sudo ./activate_imx219_cameras.sh
   ```

2. **Reboot your Jetson:**
   ```bash
   sudo reboot
   ```

3. **Test the cameras:**
   ```bash
   cd /home/jetson
   ./test_imx219_cameras.sh
   ```

That's it! Your cameras should now be working.

---

## 📖 Documentation Guide

### For Quick Tasks
→ Read **QUICK_REFERENCE.md**
- Fast command lookup
- Common operations
- Quick troubleshooting

### For Setup and Usage
→ Read **IMX219_CAMERA_SETUP.md**
- Complete hardware setup guide
- Software installation
- Testing procedures
- Troubleshooting guide
- Advanced usage examples

### For Development History
→ Read **DEVELOPMENT_LOG.md**
- Detailed timeline of development
- All errors encountered
- Solutions applied
- Testing results
- Technical notes
- Future development tasks

---

## 🛠️ Scripts Overview

### activate_imx219_cameras.sh
**Purpose:** Enable IMX219 dual camera overlay in boot configuration  
**Usage:** `sudo ./activate_imx219_cameras.sh`  
**When to use:** First time setup or if cameras stop working  
**Requirements:** sudo privileges

### test_imx219_cameras.sh
**Purpose:** Verify camera detection and provide test commands  
**Usage:** `./test_imx219_cameras.sh`  
**When to use:** After reboot to verify cameras are working  
**Requirements:** Cameras must be activated first

### imx219_camera_test.py
**Purpose:** Python-based camera testing with live preview  
**Usage:**
```bash
python3 imx219_camera_test.py check        # Check availability
python3 imx219_camera_test.py view 0       # View camera 0
python3 imx219_camera_test.py stereo       # View both cameras
python3 imx219_camera_test.py capture 0    # Capture image
```
**When to use:** For interactive testing and development  
**Requirements:** opencv-python, numpy

---

## ✅ Current Status

### Completed
- ✅ System analysis and hardware detection
- ✅ IMX219 dual camera overlay identified
- ✅ Activation scripts created
- ✅ Testing scripts created
- ✅ Python testing utility created
- ✅ Documentation created
- ✅ Development log established

### Pending (Requires User Action)
- ⏳ Run activation script with sudo
- ⏳ Reboot system
- ⏳ Verify camera detection
- ⏳ Test camera functionality
- ⏳ Update development log with results

### Future Development
- ⏳ Camera calibration
- ⏳ Stereo vision implementation
- ⏳ Depth map generation
- ⏳ Performance optimization
- ⏳ Application integration

---

## 🎯 Use Cases

This stereo camera setup is suitable for:

- **3D Vision:** Depth perception and 3D reconstruction
- **Robotics:** Obstacle detection and navigation
- **SLAM:** Simultaneous Localization and Mapping
- **Autonomous Vehicles:** Distance measurement
- **Computer Vision Research:** Stereo matching algorithms
- **AR/VR:** Spatial mapping

---

## 🔗 Important Links

- **Waveshare Product Page:** https://www.waveshare.net/wiki/IMX219-83_Stereo_Camera
- **NVIDIA Jetson Docs:** https://docs.nvidia.com/jetson/
- **OpenCV Stereo Tutorial:** https://docs.opencv.org/master/dd/d53/tutorial_py_depthmap.html

---

## 📊 Hardware Specifications

### Camera Module
- **Sensor:** Sony IMX219 (x2)
- **Resolution:** 3280 x 2464 pixels (8MP per camera)
- **Field of View:** 83 degrees
- **Baseline Distance:** ~60mm (distance between cameras)
- **Interface:** MIPI CSI-2
- **Max Frame Rate:** 30fps @ 1080p, 21fps @ full resolution

### Jetson Platform
- **Model:** Jetson Orin Nano
- **Board:** tegra234-p3768-0000+p3767-0005
- **OS:** Linux 5.15.148-tegra
- **JetPack:** R36 (Release 36.4.7)
- **CSI Ports:** Dual CSI-2 interfaces

---

## 🆘 Quick Troubleshooting

### Problem: No video devices found
**Solution:** Run activation script and reboot
```bash
sudo ./activate_imx219_cameras.sh
sudo reboot
```

### Problem: Permission denied
**Solution:** Add user to video group
```bash
sudo usermod -a -G video $USER
# Logout and login
```

### Problem: Camera not streaming
**Solution:** Check kernel messages
```bash
dmesg | grep -i imx219
```

For more troubleshooting, see **IMX219_CAMERA_SETUP.md** → Troubleshooting section.

---

## 📝 Notes

- All scripts are also available in `/home/jetson/` for easy access
- The activation script creates automatic backups of boot configuration
- Cameras require proper hardware connection before software activation
- For development history and detailed logs, always refer to `DEVELOPMENT_LOG.md`

---

## 🤝 Contributing to Development Log

When updating this project:

1. **After successful tests:** Update `DEVELOPMENT_LOG.md` → Testing Results
2. **If errors occur:** Document in `DEVELOPMENT_LOG.md` → Errors Encountered
3. **When making progress:** Update `DEVELOPMENT_LOG.md` → Development Timeline
4. **For new scripts:** Add to this README and project structure

---

## 📞 Support

If you encounter issues:

1. Check `QUICK_REFERENCE.md` for common solutions
2. Review `DEVELOPMENT_LOG.md` for similar errors
3. Consult `IMX219_CAMERA_SETUP.md` troubleshooting section
4. Check Waveshare wiki for hardware-specific issues
5. Review NVIDIA Jetson forums for platform issues

---

**Project Maintainer:** AI Development Assistant  
**Last Updated:** 2026-01-08  
**Version:** 1.0 - Initial Setup
