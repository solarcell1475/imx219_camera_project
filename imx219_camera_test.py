#!/usr/bin/env python3

"""
IMX219-83 Stereo Camera Test Script
This script provides various tests and examples for the dual IMX219 cameras
"""

import cv2
import numpy as np
import sys
import time

def test_camera_availability():
    """Test if cameras are available"""
    print("=" * 50)
    print("Testing Camera Availability")
    print("=" * 50)
    
    # Test camera 0
    cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    if cap0.isOpened():
        print("✓ Camera 0 (Left) is available")
        ret, frame = cap0.read()
        if ret:
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap0.release()
    else:
        print("✗ Camera 0 (Left) is NOT available")
    
    # Test camera 1
    cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
    if cap1.isOpened():
        print("✓ Camera 1 (Right) is available")
        ret, frame = cap1.read()
        if ret:
            print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        cap1.release()
    else:
        print("✗ Camera 1 (Right) is NOT available")
    print()

def capture_single_image(camera_id=0, filename=None):
    """Capture a single image from specified camera"""
    if filename is None:
        filename = f"camera_{camera_id}_capture.jpg"
    
    print(f"Capturing image from camera {camera_id}...")
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        return False
    
    # Wait for camera to warm up
    time.sleep(0.5)
    
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(filename, frame)
        print(f"✓ Image saved as {filename}")
        print(f"  Resolution: {frame.shape[1]}x{frame.shape[0]}")
        return True
    else:
        print(f"✗ Failed to capture image from camera {camera_id}")
        return False

def display_single_camera(camera_id=0):
    """Display live feed from a single camera"""
    print(f"Opening camera {camera_id} live view...")
    print("Press 'q' to quit, 's' to save a snapshot")
    
    cap = cv2.VideoCapture(camera_id, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print(f"✗ Failed to open camera {camera_id}")
        return
    
    # Set resolution (optional)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    snapshot_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to read frame")
            break
        
        # Add camera info to frame
        cv2.putText(frame, f"Camera {camera_id}", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit, 's' to save", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
        
        cv2.imshow(f"Camera {camera_id}", frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"snapshot_cam{camera_id}_{snapshot_count}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Snapshot saved: {filename}")
            snapshot_count += 1
    
    cap.release()
    cv2.destroyAllWindows()

def display_stereo_cameras():
    """Display both cameras side by side"""
    print("Opening stereo camera view...")
    print("Press 'q' to quit, 's' to save synchronized snapshots")
    
    cap0 = cv2.VideoCapture(0, cv2.CAP_V4L2)
    cap1 = cv2.VideoCapture(1, cv2.CAP_V4L2)
    
    if not cap0.isOpened() or not cap1.isOpened():
        print("✗ Failed to open both cameras")
        if cap0.isOpened():
            cap0.release()
        if cap1.isOpened():
            cap1.release()
        return
    
    # Set resolution
    for cap in [cap0, cap1]:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    snapshot_count = 0
    
    while True:
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if not ret0 or not ret1:
            print("Failed to read frames")
            break
        
        # Add labels
        cv2.putText(frame0, "Left Camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame1, "Right Camera", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Combine frames side by side
        combined = np.hstack((frame0, frame1))
        
        # Add instructions
        cv2.putText(combined, "Press 'q' to quit, 's' to save", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Stereo Cameras", combined)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.imwrite(f"stereo_left_{snapshot_count}.jpg", frame0)
            cv2.imwrite(f"stereo_right_{snapshot_count}.jpg", frame1)
            print(f"Stereo pair saved: stereo_left_{snapshot_count}.jpg, stereo_right_{snapshot_count}.jpg")
            snapshot_count += 1
    
    cap0.release()
    cap1.release()
    cv2.destroyAllWindows()

def main():
    print("=" * 50)
    print("IMX219-83 Stereo Camera Test")
    print("=" * 50)
    print()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 imx219_camera_test.py check         - Check camera availability")
        print("  python3 imx219_camera_test.py capture [0|1] - Capture single image")
        print("  python3 imx219_camera_test.py view [0|1]    - Display single camera")
        print("  python3 imx219_camera_test.py stereo        - Display both cameras")
        print()
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "check":
        test_camera_availability()
    
    elif command == "capture":
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        capture_single_image(camera_id)
    
    elif command == "view":
        camera_id = int(sys.argv[2]) if len(sys.argv) > 2 else 0
        display_single_camera(camera_id)
    
    elif command == "stereo":
        display_stereo_cameras()
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()
