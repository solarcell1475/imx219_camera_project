#!/usr/bin/env python3
"""
IMX219 Camera Test Script
==========================
Comprehensive test of IMX219 dual cameras before running YOLO system.
Tests camera detection, access, and basic functionality.
"""

import sys
import time
import subprocess
from pathlib import Path
import cv2
import numpy as np

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup display environment
try:
    from utils.display_fix import setup_display_environment
    setup_display_environment()
except:
    pass


def print_header(text):
    """Print formatted header"""
    print("\n" + "=" * 70)
    print(f" {text}")
    print("=" * 70)


def print_result(success, message):
    """Print test result"""
    symbol = "✓" if success else "✗"
    status = "PASS" if success else "FAIL"
    print(f"  {symbol} [{status}] {message}")


def test_video_devices():
    """Test 1: Check if video devices exist"""
    print_header("TEST 1: Video Device Detection")
    
    try:
        # Check if devices exist directly
        from pathlib import Path
        video0 = Path('/dev/video0')
        video1 = Path('/dev/video1')
        
        devices = []
        if video0.exists():
            devices.append('/dev/video0')
        if video1.exists():
            devices.append('/dev/video1')
        
        # Also check for any other video devices
        try:
            result = subprocess.run(['ls', '/dev/video*'], 
                                  capture_output=True, text=True, 
                                  timeout=2, stderr=subprocess.DEVNULL)
            if result.stdout.strip():
                all_devices = [d for d in result.stdout.strip().split('\n') 
                             if d and '/dev/video' in d and d not in devices]
                devices.extend(all_devices)
        except:
            pass
        
        if devices:
            print_result(True, f"Found {len(devices)} video device(s)")
            for dev in devices:
                print(f"    {dev}")
            
            # Check if we have at least video0 and video1
            has_video0 = '/dev/video0' in devices
            has_video1 = '/dev/video1' in devices
            
            if has_video0 and has_video1:
                print_result(True, "Both video0 and video1 found")
                return True
            elif has_video0 or has_video1:
                print_result(False, f"Only {'video0' if has_video0 else 'video1'} found (need both)")
                return False
            else:
                print_result(False, "video0 and video1 not found")
                return False
        else:
            print_result(False, "No video devices found")
            return False
    except Exception as e:
        print_result(False, f"Error: {e}")
        return False


def test_opencv_capture():
    """Test 2: Test OpenCV camera access"""
    print_header("TEST 2: OpenCV Camera Access")
    
    results = {'camera0': False, 'camera1': False}
    
    # Test Camera 0
    print("  Testing Camera 0 (/dev/video0)...")
    try:
        cap0 = cv2.VideoCapture(0)
        if cap0.isOpened():
            print_result(True, "Camera 0 opened successfully")
            
            # Set properties
            cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Try to read a frame
            ret, frame = cap0.read()
            if ret and frame is not None:
                print_result(True, f"Camera 0 frame read: {frame.shape[1]}x{frame.shape[0]}")
                results['camera0'] = True
                
                # Check if frame is valid (not all zeros or noise)
                if frame.size > 0 and np.any(frame):
                    mean_val = np.mean(frame)
                    std_val = np.std(frame)
                    print(f"    Frame stats: mean={mean_val:.2f}, std={std_val:.2f}")
                    if mean_val > 5 and std_val > 5:
                        print_result(True, "Camera 0 frame looks valid (not pure noise)")
                    else:
                        print_result(False, "Camera 0 frame may be noise/zeros")
                else:
                    print_result(False, "Camera 0 frame is empty")
            else:
                print_result(False, "Camera 0 cannot read frames")
            cap0.release()
        else:
            print_result(False, "Camera 0 failed to open")
    except Exception as e:
        print_result(False, f"Camera 0 error: {e}")
    
    # Test Camera 1
    print("\n  Testing Camera 1 (/dev/video1)...")
    try:
        cap1 = cv2.VideoCapture(1)
        if cap1.isOpened():
            print_result(True, "Camera 1 opened successfully")
            
            # Set properties
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            
            # Try to read a frame
            ret, frame = cap1.read()
            if ret and frame is not None:
                print_result(True, f"Camera 1 frame read: {frame.shape[1]}x{frame.shape[0]}")
                results['camera1'] = True
                
                # Check if frame is valid
                if frame.size > 0 and np.any(frame):
                    mean_val = np.mean(frame)
                    std_val = np.std(frame)
                    print(f"    Frame stats: mean={mean_val:.2f}, std={std_val:.2f}")
                    if mean_val > 5 and std_val > 5:
                        print_result(True, "Camera 1 frame looks valid (not pure noise)")
                    else:
                        print_result(False, "Camera 1 frame may be noise/zeros")
                else:
                    print_result(False, "Camera 1 frame is empty")
            else:
                print_result(False, "Camera 1 cannot read frames")
            cap1.release()
        else:
            print_result(False, "Camera 1 failed to open")
    except Exception as e:
        print_result(False, f"Camera 1 error: {e}")
    
    return results


def test_gstreamer_pipeline():
    """Test 3: Test GStreamer pipeline (used by our system)"""
    print_header("TEST 3: GStreamer Pipeline Test")
    
    # GStreamer pipeline for IMX219 on Jetson
    pipeline0 = (
        "nvarguscamerasrc sensor-id=0 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=1280, height=720, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    pipeline1 = (
        "nvarguscamerasrc sensor-id=1 ! "
        "video/x-raw(memory:NVMM), width=1280, height=720, format=NV12, framerate=30/1 ! "
        "nvvidconv flip-method=0 ! "
        "video/x-raw, width=1280, height=720, format=BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=BGR ! "
        "appsink"
    )
    
    results = {'camera0': False, 'camera1': False}
    
    # Test Camera 0 GStreamer
    print("  Testing Camera 0 GStreamer pipeline...")
    try:
        cap0 = cv2.VideoCapture(pipeline0, cv2.CAP_GSTREAMER)
        if cap0.isOpened():
            print_result(True, "Camera 0 GStreamer pipeline opened")
            
            # Try to read a frame (with timeout)
            start_time = time.time()
            ret, frame = cap0.read()
            elapsed = time.time() - start_time
            
            if ret and frame is not None:
                print_result(True, f"Camera 0 GStreamer frame read: {frame.shape[1]}x{frame.shape[0]} ({elapsed:.2f}s)")
                results['camera0'] = True
            else:
                print_result(False, f"Camera 0 GStreamer cannot read frames (timeout: {elapsed:.2f}s)")
            cap0.release()
        else:
            print_result(False, "Camera 0 GStreamer pipeline failed to open")
    except Exception as e:
        print_result(False, f"Camera 0 GStreamer error: {e}")
    
    # Test Camera 1 GStreamer
    print("\n  Testing Camera 1 GStreamer pipeline...")
    try:
        cap1 = cv2.VideoCapture(pipeline1, cv2.CAP_GSTREAMER)
        if cap1.isOpened():
            print_result(True, "Camera 1 GStreamer pipeline opened")
            
            start_time = time.time()
            ret, frame = cap1.read()
            elapsed = time.time() - start_time
            
            if ret and frame is not None:
                print_result(True, f"Camera 1 GStreamer frame read: {frame.shape[1]}x{frame.shape[0]} ({elapsed:.2f}s)")
                results['camera1'] = True
            else:
                print_result(False, f"Camera 1 GStreamer cannot read frames (timeout: {elapsed:.2f}s)")
            cap1.release()
        else:
            print_result(False, "Camera 1 GStreamer pipeline failed to open")
    except Exception as e:
        print_result(False, f"Camera 1 GStreamer error: {e}")
    
    return results


def test_nvgstcapture():
    """Test 4: Test with nvgstcapture (vendor's recommended method)"""
    print_header("TEST 4: nvgstcapture Test (Vendor Method)")
    
    try:
        result = subprocess.run(['which', 'nvgstcapture-1.0'], 
                              capture_output=True, text=True, timeout=2)
        if result.returncode == 0:
            print_result(True, f"nvgstcapture-1.0 found: {result.stdout.strip()}")
            print("\n  NOTE: nvgstcapture requires HDMI/DP display and outputs to screen.")
            print("  Manual test commands:")
            print("    DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0")
            print("    DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=1")
            return True
        else:
            print_result(False, "nvgstcapture-1.0 not found")
            return False
    except Exception as e:
        print_result(False, f"Error checking nvgstcapture: {e}")
        return False


def test_display_capture():
    """Test 5: Display captured frames"""
    print_header("TEST 5: Display Test (If cameras work)")
    
    print("  Attempting to capture and display frames from both cameras...")
    
    # Try OpenCV first
    cap0 = cv2.VideoCapture(0)
    cap1 = cv2.VideoCapture(1)
    
    if not (cap0.isOpened() and cap1.isOpened()):
        print_result(False, "Cannot open cameras for display test")
        return False
    
    # Set properties
    cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("  Reading frames (this may take a few seconds)...")
    
    frames_read = 0
    max_attempts = 30
    
    for i in range(max_attempts):
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()
        
        if ret0 and ret1 and frame0 is not None and frame1 is not None:
            frames_read += 1
            if frames_read >= 3:  # Successfully read 3 frames
                break
    
    cap0.release()
    cap1.release()
    
    if frames_read >= 3:
        print_result(True, f"Successfully read {frames_read} frame pairs")
        
        # Try to display
        print("\n  Displaying frames in window (press 'q' to quit)...")
        try:
            cap0 = cv2.VideoCapture(0)
            cap1 = cv2.VideoCapture(1)
            cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            window_name0 = "IMX219 Camera 0 Test"
            window_name1 = "IMX219 Camera 1 Test"
            cv2.namedWindow(window_name0, cv2.WINDOW_NORMAL)
            cv2.namedWindow(window_name1, cv2.WINDOW_NORMAL)
            
            print("  Showing frames for 5 seconds...")
            start_time = time.time()
            
            while (time.time() - start_time) < 5:
                ret0, frame0 = cap0.read()
                ret1, frame1 = cap1.read()
                
                if ret0 and frame0 is not None:
                    cv2.putText(frame0, "Camera 0", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name0, frame0)
                
                if ret1 and frame1 is not None:
                    cv2.putText(frame1, "Camera 1", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imshow(window_name1, frame1)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            
            cv2.destroyAllWindows()
            cap0.release()
            cap1.release()
            
            print_result(True, "Display test completed")
            return True
        except Exception as e:
            print_result(False, f"Display error: {e}")
            return False
    else:
        print_result(False, f"Only read {frames_read} frame pairs (expected 3+)")
        return False


def print_summary(test_results):
    """Print test summary"""
    print_header("TEST SUMMARY")
    
    total_tests = len(test_results)
    passed_tests = sum(1 for r in test_results.values() if r)
    
    print(f"\n  Tests Passed: {passed_tests}/{total_tests}\n")
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    print("\n" + "=" * 70)
    
    if passed_tests == total_tests:
        print("  ✓ ALL TESTS PASSED - Cameras are ready for YOLO system!")
    elif passed_tests > 0:
        print("  ⚠ SOME TESTS FAILED - Review errors above")
        print("  → Cameras may work but need configuration")
    else:
        print("  ✗ ALL TESTS FAILED - Cameras are not working")
        print("\n  Troubleshooting:")
        print("    1. Check hardware connections")
        print("    2. Run: sudo ./activate_imx219_cameras.sh")
        print("    3. Reboot: sudo reboot")
        print("    4. Test with: DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0")
    
    print("=" * 70)


def main():
    """Main test function"""
    print("\n" + "=" * 70)
    print(" IMX219 DUAL CAMERA TEST")
    print("=" * 70)
    print("\nThis script tests IMX219 cameras before running YOLO system.")
    print("Testing camera detection, access, and functionality...")
    
    test_results = {}
    
    # Run all tests
    test_results['Video Devices'] = test_video_devices()
    
    opencv_results = test_opencv_capture()
    test_results['OpenCV Camera 0'] = opencv_results.get('camera0', False)
    test_results['OpenCV Camera 1'] = opencv_results.get('camera1', False)
    
    gstreamer_results = test_gstreamer_pipeline()
    test_results['GStreamer Camera 0'] = gstreamer_results.get('camera0', False)
    test_results['GStreamer Camera 1'] = gstreamer_results.get('camera1', False)
    
    test_results['nvgstcapture Available'] = test_nvgstcapture()
    
    # Only test display if cameras are working
    if opencv_results.get('camera0') or opencv_results.get('camera1'):
        test_results['Display Test'] = test_display_capture()
    else:
        print_header("TEST 5: Display Test (SKIPPED)")
        print("  Skipped: Cameras not working, cannot test display")
        test_results['Display Test'] = False
    
    # Print summary
    print_summary(test_results)
    
    return 0 if all(test_results.values()) else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        cv2.destroyAllWindows()
        sys.exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        cv2.destroyAllWindows()
        sys.exit(1)
