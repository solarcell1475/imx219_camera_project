#!/usr/bin/env python3
"""
Comprehensive Camera Detection and Verification
================================================
Detects and verifies dual IMX219 cameras on Jetson Orin Nano.
"""

import subprocess
import cv2
import sys
from pathlib import Path


class CameraDetection:
    """Camera detection and verification system"""
    
    def __init__(self):
        self.cameras = []
        
    def detect_video_devices(self):
        """Detect all video devices"""
        devices = []
        for i in range(10):  # Check up to /dev/video9
            device_path = f"/dev/video{i}"
            if Path(device_path).exists():
                devices.append(device_path)
        return devices
    
    def test_gstreamer_pipeline(self, sensor_id, width=1280, height=720):
        """Test GStreamer pipeline for a camera"""
        gst_cmd = [
            'gst-launch-1.0',
            '-q',
            f'nvarguscamerasrc',
            f'sensor-id={sensor_id}',
            f'!', f'video/x-raw(memory:NVMM),width={width},height={height},framerate=30/1',
            f'!', 'nvvidconv',
            f'!', 'video/x-raw,format=BGRx',
            f'!', 'videoconvert',
            f'!', 'video/x-raw,format=BGR',
            f'!', 'appsink', 'max-buffers=1', 'drop=true'
        ]
        
        try:
            # Try to open with OpenCV
            gst_str = (
                f"nvarguscamerasrc sensor-id={sensor_id} ! "
                f"video/x-raw(memory:NVMM), width={width}, height={height}, "
                f"format=NV12, framerate=30/1 ! "
                f"nvvidconv ! video/x-raw, format=BGRx ! "
                f"videoconvert ! video/x-raw, format=BGR ! appsink"
            )
            
            cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()
                return ret and frame is not None
            return False
        except Exception as e:
            return False
    
    def verify_camera(self, sensor_id):
        """Verify a camera is working"""
        result = {
            'sensor_id': sensor_id,
            'device_path': f'/dev/video{sensor_id}',
            'device_exists': Path(f'/dev/video{sensor_id}').exists(),
            'gstreamer_works': False,
            'opencv_works': False
        }
        
        # Test GStreamer
        result['gstreamer_works'] = self.test_gstreamer_pipeline(sensor_id)
        
        # Test OpenCV direct access
        try:
            cap = cv2.VideoCapture(sensor_id)
            if cap.isOpened():
                ret, frame = cap.read()
                result['opencv_works'] = ret and frame is not None
                cap.release()
        except:
            pass
        
        return result
    
    def detect_all_cameras(self):
        """Detect and verify all cameras"""
        results = []
        
        # Check for both cameras (sensor-id 0 and 1)
        for sensor_id in [0, 1]:
            result = self.verify_camera(sensor_id)
            results.append(result)
            if result['gstreamer_works'] or result['opencv_works']:
                self.cameras.append(result)
        
        return results
    
    def generate_report(self):
        """Generate installation verification report"""
        results = self.detect_all_cameras()
        
        report = []
        report.append("=" * 70)
        report.append("Camera Installation Verification Report")
        report.append("=" * 70)
        report.append("")
        
        for result in results:
            sensor_id = result['sensor_id']
            report.append(f"Camera {sensor_id} ({result['device_path']}):")
            
            if result['device_exists']:
                report.append("  ✓ Device file exists")
            else:
                report.append("  ✗ Device file not found")
            
            if result['gstreamer_works']:
                report.append("  ✓ GStreamer pipeline works")
            else:
                report.append("  ✗ GStreamer pipeline failed")
            
            if result['opencv_works']:
                report.append("  ✓ OpenCV access works")
            else:
                report.append("  ✗ OpenCV access failed")
            
            report.append("")
        
        report.append("=" * 70)
        report.append(f"Summary: {len(self.cameras)} camera(s) detected and working")
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Main function"""
    detector = CameraDetection()
    report = detector.generate_report()
    print(report)
    
    if len(detector.cameras) >= 2:
        print("\n✓ Dual camera setup verified!")
        return 0
    elif len(detector.cameras) == 1:
        print("\n⚠ Only one camera detected")
        return 1
    else:
        print("\n✗ No cameras detected")
        return 2


if __name__ == "__main__":
    sys.exit(main())
