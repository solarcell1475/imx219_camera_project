# IMX219-83 Stereo Camera Installation Guide Translation

**Original File:** `installation_guideline.txt` (Chinese)  
**Translated for:** Jetson Orin Nano  
**Date:** 2026-01-08

---

## Key Instructions from Waveshare Guide

### Hardware Connection (硬件连接)

1. **Connect Camera Ribbons:**
   - Insert the two camera ribbon cables into the camera ports on Jetson Nano
   - **Important:** Metal contacts should face toward the heatsink
   - Ensure cables are firmly seated

2. **Power On:**
   - Start your Jetson Nano after connecting cameras

---

### Testing Cameras (测试摄像头)

#### Step 1: Check Device Detection
Open terminal and run:
```bash
ls /dev/video*
```

**Expected Result:** Should see `video0` and `video1`

#### Step 2: Test Camera 0 (Left Camera)
```bash
DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0
```

#### Step 3: Test Camera 1 (Right Camera)
```bash
DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=1
```

---

### Fix Reddish Images (如果摄像头拍摄效果偏红)

If camera images appear too red, install the camera override ISP file:

#### Step 1: Download and Extract
```bash
cd ~/Downloads
wget http://www.waveshare.net/w/upload/e/eb/Camera_overrides.tar.gz
tar zxvf Camera_overrides.tar.gz
sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
```

#### Step 2: Set Permissions
```bash
sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
```

---

### Important Notes from Guide

⚠️ **NV12 Format:** The "12" in NV12 is a number, not the letter "L"

⚠️ **Display Required:** Test output goes to HDMI or DP screen, so connect a monitor before testing

---

## IMU Sensor Usage (IMU传感器使用)

The camera module includes an ICM20948 9-axis sensor. To use it:

1. **Connect IMU:**
   - Connect SDA and SCL pins from camera to Jetson Nano pins 3 and 5
   - Only SDA and SCL pins are needed for basic operation

2. **Download and Test:**
```bash
wget http://www.waveshare.net/w/upload/a/a4/D219-9dof.tar.gz
tar zxvf D219-9dof.tar.gz
cd D219-9dof/07-icm20948-demo
make
./ICM20948-Demo
```

3. **Test:** Rotate the camera and observe the value changes

---

## Additional Information

The guide also contains extensive information about:
- Raspberry Pi usage (not applicable for Jetson)
- libcamera and rpicam commands (Raspberry Pi specific)
- Various camera parameters and settings

For Jetson Nano/Orin, the relevant parts are:
1. Hardware connection
2. Device detection check (`ls /dev/video*`)
3. Testing with nvgstcapture
4. Optional camera_overrides.isp for color correction

---

## Current Status on Your System

### ✅ What's Ready:
- nvgstcapture-1.0 is installed
- IMX219 dual camera overlay is available
- Activation scripts are prepared

### ⏳ What's Needed:
- Run activation script to enable cameras
- Reboot system
- Test with nvgstcapture-1.0
- Install camera_overrides.isp if colors are off

---

## Next Steps (Based on Guide)

1. **Activate Cameras:**
   ```bash
   cd /home/jetson
   sudo ./activate_imx219_cameras.sh
   ```

2. **Reboot:**
   ```bash
   sudo reboot
   ```

3. **After Reboot - Check Detection:**
   ```bash
   ls /dev/video*
   ```
   Should see: `/dev/video0` and `/dev/video1`

4. **Test Camera 0:**
   ```bash
   DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0
   ```

5. **Test Camera 1:**
   ```bash
   DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=1
   ```

6. **If Images Too Red:**
   ```bash
   cd ~/Downloads
   wget http://www.waveshare.net/w/upload/e/eb/Camera_overrides.tar.gz
   tar zxvf Camera_overrides.tar.gz
   sudo cp camera_overrides.isp /var/nvidia/nvcam/settings/
   sudo chmod 664 /var/nvidia/nvcam/settings/camera_overrides.isp
   sudo chown root:root /var/nvidia/nvcam/settings/camera_overrides.isp
   ```

---

## Comparison: Our Method vs Waveshare Guide

### Our Approach:
- Uses device tree overlay (more modern)
- Permanent configuration in boot config
- Multiple testing methods (GStreamer, Python, nvgstcapture)

### Waveshare Guide:
- Focuses on nvgstcapture for testing
- Includes ISP override for color correction
- Mentions IMU sensor usage

### Best Practice:
Combine both approaches:
1. Use our activation script for permanent setup
2. Use nvgstcapture for initial testing (per Waveshare)
3. Apply ISP override if needed (per Waveshare)
4. Use our Python scripts for development

---

**Translation Date:** 2026-01-08  
**Guide Source:** Waveshare Wiki  
**Applicable To:** Jetson Nano, Jetson Orin Nano
