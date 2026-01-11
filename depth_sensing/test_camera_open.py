#!/usr/bin/env python3
"""
Test camera opening with different methods
"""

import cv2
import sys

print("Testing camera access methods...")
print("=" * 70)

# Method 1: Simple index
print("\nMethod 1: Simple camera index")
cap0 = cv2.VideoCapture(0)
cap1 = cv2.VideoCapture(1)

if cap0.isOpened():
    print("✓ Camera 0 opened with simple index")
    ret, frame = cap0.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap0.release()
else:
    print("✗ Camera 0 failed with simple index")

if cap1.isOpened():
    print("✓ Camera 1 opened with simple index")
    ret, frame = cap1.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap1.release()
else:
    print("✗ Camera 1 failed with simple index")

# Method 2: GStreamer with v4l2src
print("\nMethod 2: GStreamer v4l2src")
gst_str_0 = (
    "v4l2src device=/dev/video0 ! "
    "video/x-raw, width=1280, height=720 ! "
    "videoconvert ! appsink"
)

gst_str_1 = (
    "v4l2src device=/dev/video1 ! "
    "video/x-raw, width=1280, height=720 ! "
    "videoconvert ! appsink"
)

cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)

if cap0.isOpened():
    print("✓ Camera 0 opened with v4l2src")
    ret, frame = cap0.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap0.release()
else:
    print("✗ Camera 0 failed with v4l2src")

if cap1.isOpened():
    print("✓ Camera 1 opened with v4l2src")
    ret, frame = cap1.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap1.release()
else:
    print("✗ Camera 1 failed with v4l2src")

# Method 3: nvarguscamerasrc (original)
print("\nMethod 3: nvarguscamerasrc")
gst_str_0 = (
    "nvarguscamerasrc sensor-id=0 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

gst_str_1 = (
    "nvarguscamerasrc sensor-id=1 ! "
    "video/x-raw(memory:NVMM), width=1280, height=720, "
    "format=NV12, framerate=30/1 ! "
    "nvvidconv ! video/x-raw, format=BGRx ! "
    "videoconvert ! video/x-raw, format=BGR ! appsink"
)

cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)

if cap0.isOpened():
    print("✓ Camera 0 opened with nvarguscamerasrc")
    ret, frame = cap0.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap0.release()
else:
    print("✗ Camera 0 failed with nvarguscamerasrc")

if cap1.isOpened():
    print("✓ Camera 1 opened with nvarguscamerasrc")
    ret, frame = cap1.read()
    if ret:
        print(f"  Frame size: {frame.shape}")
    cap1.release()
else:
    print("✗ Camera 1 failed with nvarguscamerasrc")

print("\n" + "=" * 70)
print("Test complete")
