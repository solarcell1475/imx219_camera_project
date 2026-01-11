#!/usr/bin/env python3
"""
GStreamer Dual Camera Capture
==============================
Thread-safe dual camera capture using GStreamer with frame buffers.
"""

import cv2
import threading
import queue
import time
from typing import Optional, Tuple
import numpy as np


class GStreamerDualCapture:
    """Thread-safe dual camera capture using GStreamer"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize dual camera capture
        
        Args:
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self.cap0: Optional[cv2.VideoCapture] = None
        self.cap1: Optional[cv2.VideoCapture] = None
        
        # Frame buffers
        self.frame_queue0 = queue.Queue(maxsize=2)
        self.frame_queue1 = queue.Queue(maxsize=2)
        
        # Capture threads
        self.capture_thread0: Optional[threading.Thread] = None
        self.capture_thread1: Optional[threading.Thread] = None
        
        self.running = False
        
        # Statistics
        self.stats = {
            'frames_captured_0': 0,
            'frames_captured_1': 0,
            'frames_dropped_0': 0,
            'frames_dropped_1': 0,
            'last_frame_time_0': 0.0,
            'last_frame_time_1': 0.0
        }
    
    def _create_gstreamer_pipeline(self, sensor_id: int) -> str:
        """Create GStreamer pipeline string - matches working nvgstcapture approach"""
        # Use the exact format that works with nvgstcapture
        # Add max-buffers and drop to prevent blocking
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! "
            f"appsink max-buffers=1 drop=true"
        )
    
    def _capture_loop(self, cap: cv2.VideoCapture, frame_queue: queue.Queue, camera_id: int):
        """Capture loop for a single camera"""
        frames_key = f'frames_captured_{camera_id}'
        dropped_key = f'frames_dropped_{camera_id}'
        time_key = f'last_frame_time_{camera_id}'
        
        while self.running:
            ret, frame = cap.read()
            if ret and frame is not None:
                # Resize frame to configured resolution if needed
                if frame.shape[1] != self.width or frame.shape[0] != self.height:
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
                
                # Validate frame (check if it's not all zeros or uniform)
                if frame.size > 0 and np.std(frame) > 5:  # Frame has some variance
                    # Try to put frame in queue (drop if full)
                    try:
                        frame_queue.put_nowait(frame)
                        self.stats[frames_key] += 1
                        self.stats[time_key] = time.time()
                    except queue.Full:
                        # Drop oldest frame
                        try:
                            frame_queue.get_nowait()
                            frame_queue.put_nowait(frame)
                            self.stats[dropped_key] += 1
                            self.stats[frames_key] += 1
                            self.stats[time_key] = time.time()
                        except queue.Empty:
                            pass
                else:
                    # Invalid frame (too uniform or empty)
                    time.sleep(0.01)
            else:
                time.sleep(0.001)  # Small delay on error
    
    def start(self) -> bool:
        """Start capture threads"""
        if self.running:
            return True
        
        # Try GStreamer first - this is the preferred method for Jetson
        gst_str_0 = self._create_gstreamer_pipeline(0)
        gst_str_1 = self._create_gstreamer_pipeline(1)
        
        print(f"Attempting GStreamer pipeline for Camera 0...")
        self.cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
        
        print(f"Attempting GStreamer pipeline for Camera 1...")
        self.cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)
        
        # Wait a bit for GStreamer to initialize
        time.sleep(0.5)
        
        # Test if we can actually read frames (not just isOpened)
        gstreamer_works = False
        if self.cap0.isOpened() and self.cap1.isOpened():
            # Try reading a test frame from both cameras
            ret0, test_frame0 = self.cap0.read()
            ret1, test_frame1 = self.cap1.read()
            
            if ret0 and ret1 and test_frame0 is not None and test_frame1 is not None:
                gstreamer_works = True
                print("✓ GStreamer pipelines working - frames can be read")
            else:
                print(f"⚠ GStreamer opened but cannot read frames (cam0: {ret0}, cam1: {ret1})")
                self.release()
        else:
            print("⚠ GStreamer pipelines failed to open")
            self.release()
        
        # If GStreamer doesn't work, we MUST NOT fallback to V4L2 (it times out)
        # Instead, we should fix GStreamer or fail gracefully
        if not gstreamer_works:
            print("ERROR: GStreamer pipelines are required for Jetson cameras")
            print("  V4L2 direct access does not work reliably on Jetson")
            print("  Please ensure:")
            print("    1. Cameras are properly initialized (reboot if needed)")
            print("    2. GStreamer plugins are installed")
            print("    3. Try running: DISPLAY=:0.0 nvgstcapture-1.0 --sensor-id=0")
            self.release()
            return False
        
        # Start capture threads
        self.running = True
        self.capture_thread0 = threading.Thread(
            target=self._capture_loop,
            args=(self.cap0, self.frame_queue0, 0),
            daemon=True
        )
        self.capture_thread1 = threading.Thread(
            target=self._capture_loop,
            args=(self.cap1, self.frame_queue1, 1),
            daemon=True
        )
        
        self.capture_thread0.start()
        self.capture_thread1.start()
        
        # Wait a bit for frames to start coming
        time.sleep(0.1)
        
        return True
    
    def stop(self):
        """Stop capture threads"""
        self.running = False
        
        if self.capture_thread0:
            self.capture_thread0.join(timeout=1.0)
        if self.capture_thread1:
            self.capture_thread1.join(timeout=1.0)
        
        self.release()
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Read synchronized frames from both cameras
        
        Args:
            timeout: Timeout in seconds
        
        Returns:
            (success, frame0, frame1)
        """
        if not self.running:
            return False, None, None
        
        try:
            frame0 = self.frame_queue0.get(timeout=timeout)
            frame1 = self.frame_queue1.get(timeout=timeout)
            return True, frame0, frame1
        except queue.Empty:
            return False, None, None
    
    def get_statistics(self) -> dict:
        """Get capture statistics"""
        return self.stats.copy()
    
    def release(self):
        """Release camera resources"""
        if self.cap0:
            self.cap0.release()
            self.cap0 = None
        if self.cap1:
            self.cap1.release()
            self.cap1 = None
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()


class SimpleDualCapture:
    """Simpler synchronous capture (for testing)"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        self.width = width
        self.height = height
        self.fps = fps
        
        gst_str_0 = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        gst_str_1 = (
            f"nvarguscamerasrc sensor-id=1 ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate={fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        self.cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
        self.cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)
    
    def read(self) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read frames synchronously"""
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        
        if ret0 and ret1:
            return True, frame0, frame1
        else:
            return False, frame0, frame1
    
    def release(self):
        """Release cameras"""
        if self.cap0:
            self.cap0.release()
        if self.cap1:
            self.cap1.release()
