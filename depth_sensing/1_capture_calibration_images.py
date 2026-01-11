#!/usr/bin/env python3
"""
IMX219 Stereo Camera Calibration Image Capture
===============================================
This script captures synchronized images from both cameras for calibration.
Use a checkerboard pattern (recommended: 9x6 internal corners, 25mm square size).

Usage:
    python3 1_capture_calibration_images.py

Controls:
    SPACE - Capture image pair
    Q     - Quit
    
Requirements:
    - Both cameras connected and working
    - Checkerboard calibration pattern
    - Good lighting conditions
"""

import cv2
import numpy as np
import os
from datetime import datetime
import sys

class StereoCalibrationCapture:
    def __init__(self):
        # Camera indices
        self.camera0_id = 0  # Left camera
        self.camera1_id = 1  # Right camera
        
        # Output directory
        self.output_dir = "calibration_images"
        self.left_dir = os.path.join(self.output_dir, "left")
        self.right_dir = os.path.join(self.output_dir, "right")
        
        # Create directories
        os.makedirs(self.left_dir, exist_ok=True)
        os.makedirs(self.right_dir, exist_ok=True)
        
        # Image counter
        self.image_count = 0
        
        # Camera resolution
        self.width = 1280
        self.height = 720
        
        # Checkerboard pattern (internal corners)
        self.pattern_size = (9, 6)  # 9x6 internal corners
        self.square_size = 25  # mm
        
        print("=" * 70)
        print("IMX219 Stereo Camera Calibration - Image Capture")
        print("=" * 70)
        print(f"\nCheckerboard Pattern: {self.pattern_size[0]}x{self.pattern_size[1]} internal corners")
        print(f"Square Size: {self.square_size}mm")
        print(f"Output Directory: {self.output_dir}/")
        print("\nControls:")
        print("  SPACE - Capture image pair")
        print("  Q     - Quit")
        print("\nTips for good calibration:")
        print("  • Use good lighting")
        print("  • Capture 20-30 image pairs")
        print("  • Vary checkerboard position and angle")
        print("  • Cover entire field of view")
        print("  • Keep checkerboard flat and visible in both cameras")
        print("=" * 70)
        
    def initialize_cameras(self):
        """Initialize both cameras with GStreamer pipeline"""
        print("\nInitializing cameras...")
        
        # GStreamer pipeline for camera 0 (left)
        gst_str_0 = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        # GStreamer pipeline for camera 1 (right)
        gst_str_1 = (
            f"nvarguscamerasrc sensor-id=1 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        self.cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
        self.cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)
        
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            print("ERROR: Failed to open cameras!")
            sys.exit(1)
        
        print("✓ Both cameras initialized successfully")
        
    def detect_checkerboard(self, image):
        """Detect checkerboard pattern in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(
            gray, 
            self.pattern_size,
            cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )
        return ret, corners, gray
    
    def capture_images(self):
        """Main capture loop"""
        print("\nStarting capture... Press SPACE to capture, Q to quit")
        
        while True:
            # Read frames from both cameras
            ret0, frame0 = self.cap0.read()
            ret1, frame1 = self.cap1.read()
            
            if not ret0 or not ret1:
                print("ERROR: Failed to read from cameras")
                break
            
            # Detect checkerboard in both images
            found0, corners0, gray0 = self.detect_checkerboard(frame0)
            found1, corners1, gray1 = self.detect_checkerboard(frame1)
            
            # Draw checkerboard corners if found
            display0 = frame0.copy()
            display1 = frame1.copy()
            
            if found0:
                cv2.drawChessboardCorners(display0, self.pattern_size, corners0, found0)
            if found1:
                cv2.drawChessboardCorners(display1, self.pattern_size, corners1, found1)
            
            # Add status text
            status0 = "✓ Pattern Found" if found0 else "✗ Pattern Not Found"
            status1 = "✓ Pattern Found" if found1 else "✗ Pattern Not Found"
            
            color0 = (0, 255, 0) if found0 else (0, 0, 255)
            color1 = (0, 255, 0) if found1 else (0, 0, 255)
            
            cv2.putText(display0, f"Left: {status0}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color0, 2)
            cv2.putText(display1, f"Right: {status1}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color1, 2)
            
            cv2.putText(display0, f"Images: {self.image_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display1, f"Images: {self.image_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Combine views side by side
            combined = np.hstack((display0, display1))
            
            # Resize for display if too large
            display_height = 540
            display_width = int(combined.shape[1] * display_height / combined.shape[0])
            combined_resized = cv2.resize(combined, (display_width, display_height))
            
            cv2.imshow('Stereo Calibration Capture (SPACE=Capture, Q=Quit)', combined_resized)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord(' '):  # Space - capture
                if found0 and found1:
                    # Save images
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    left_path = os.path.join(self.left_dir, f"left_{self.image_count:03d}_{timestamp}.jpg")
                    right_path = os.path.join(self.right_dir, f"right_{self.image_count:03d}_{timestamp}.jpg")
                    
                    cv2.imwrite(left_path, frame0)
                    cv2.imwrite(right_path, frame1)
                    
                    self.image_count += 1
                    print(f"✓ Captured image pair {self.image_count}")
                    
                    if self.image_count >= 20:
                        print(f"\n✓ Good! You have {self.image_count} image pairs.")
                        print("  You can continue or press Q to finish.")
                else:
                    print("✗ Checkerboard not detected in both cameras! Position it correctly.")
                    
            elif key == ord('q') or key == ord('Q'):  # Quit
                break
        
        # Cleanup
        self.cap0.release()
        self.cap1.release()
        cv2.destroyAllWindows()
        
        print("\n" + "=" * 70)
        print(f"Capture Complete!")
        print(f"Total images captured: {self.image_count} pairs")
        print(f"Left images: {self.left_dir}/")
        print(f"Right images: {self.right_dir}/")
        print("\nNext step: Run calibration script")
        print("  python3 2_calibrate_stereo_cameras.py")
        print("=" * 70)

def main():
    try:
        capturer = StereoCalibrationCapture()
        capturer.initialize_cameras()
        capturer.capture_images()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()
