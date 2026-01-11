#!/usr/bin/env python3
"""
Simple camera display test
Tests if cameras can capture and display frames correctly
"""

import cv2
import numpy as np
import time

print("=" * 70)
print("Camera Display Test")
print("=" * 70)
print()

# Test Camera 0
print("Testing Camera 0...")
cap0 = cv2.VideoCapture(0)
cap0.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap0.isOpened():
    print("✗ Camera 0 failed to open")
else:
    print("✓ Camera 0 opened")
    ret, frame = cap0.read()
    if ret and frame is not None:
        print(f"✓ Frame captured: {frame.shape}")
        print(f"  Mean: {np.mean(frame):.2f}, Std: {np.std(frame):.2f}")
        
        # Resize if too large
        if frame.shape[1] > 1280 or frame.shape[0] > 720:
            frame = cv2.resize(frame, (1280, 720))
        
        # Display
        cv2.namedWindow("Camera 0 Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera 0 Test", frame)
        print("  Displaying frame for 3 seconds...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        print("✗ Failed to read frame from Camera 0")
    cap0.release()

print()

# Test Camera 1
print("Testing Camera 1...")
cap1 = cv2.VideoCapture(1)
cap1.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap1.isOpened():
    print("✗ Camera 1 failed to open")
else:
    print("✓ Camera 1 opened")
    ret, frame = cap1.read()
    if ret and frame is not None:
        print(f"✓ Frame captured: {frame.shape}")
        print(f"  Mean: {np.mean(frame):.2f}, Std: {np.std(frame):.2f}")
        
        # Resize if too large
        if frame.shape[1] > 1280 or frame.shape[0] > 720:
            frame = cv2.resize(frame, (1280, 720))
        
        # Display
        cv2.namedWindow("Camera 1 Test", cv2.WINDOW_NORMAL)
        cv2.imshow("Camera 1 Test", frame)
        print("  Displaying frame for 3 seconds...")
        cv2.waitKey(3000)
        cv2.destroyAllWindows()
    else:
        print("✗ Failed to read frame from Camera 1")
    cap1.release()

print()
print("=" * 70)
print("Test Complete")
print("=" * 70)
