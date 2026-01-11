#!/usr/bin/env python3
"""
Simple test of capturing from GStreamer subprocess
"""

import cv2
import numpy as np
import subprocess
import sys

print("Testing GStreamer subprocess capture...")

width, height = 640, 480

# Simple GStreamer pipeline
gst_cmd = [
    'gst-launch-1.0', '-q',
    'nvarguscamerasrc', 'sensor-id=0', 'num-buffers=10',
    '!', f'video/x-raw(memory:NVMM),width={width},height={height},framerate=30/1',
    '!', 'nvvidconv',
    '!', 'video/x-raw,format=BGR',
    '!', 'fdsink'
]

print(f"Pipeline: {' '.join(gst_cmd)}")
print(f"Expected frame size: {width}x{height}x3 = {width*height*3} bytes")

try:
    proc = subprocess.Popen(
        gst_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=10**8
    )
    
    frame_size = width * height * 3
    
    for i in range(5):
        print(f"\nReading frame {i+1}...")
        raw_data = proc.stdout.read(frame_size)
        
        print(f"  Received {len(raw_data)} bytes")
        
        if len(raw_data) == frame_size:
            frame = np.frombuffer(raw_data, dtype=np.uint8).reshape((height, width, 3))
            print(f"  Frame shape: {frame.shape}")
            cv2.imwrite(f"test_frame_{i}.jpg", frame)
            print(f"  Saved test_frame_{i}.jpg")
        else:
            print(f"  ERROR: Expected {frame_size}, got {len(raw_data)}")
            break
    
    proc.terminate()
    proc.wait()
    
    print("\n✓ Test successful!")
    
except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
