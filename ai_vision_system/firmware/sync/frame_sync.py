#!/usr/bin/env python3
"""
Frame Synchronization Module
============================
Hardware timestamp-based synchronization for dual camera frames with
frame buffer management and sync quality monitoring.
"""

import cv2
import time
import threading
import queue
from typing import Optional, Tuple, Dict
from collections import deque
import numpy as np


class SynchronizedDualCapture:
    """Synchronized dual-camera capture with timestamp matching"""
    
    def __init__(self, camera0, camera1, max_buffer_size=5, sync_threshold_ms=33.0):
        """
        Initialize synchronized capture
        
        Args:
            camera0: First camera (cv2.VideoCapture)
            camera1: Second camera (cv2.VideoCapture)
            max_buffer_size: Maximum frames to buffer per camera
            sync_threshold_ms: Maximum time difference for sync (ms)
        """
        self.camera0 = camera0
        self.camera1 = camera1
        self.max_buffer_size = max_buffer_size
        self.sync_threshold_ms = sync_threshold_ms
        
        # Frame buffers with timestamps
        self.buffer0 = deque(maxlen=max_buffer_size)
        self.buffer1 = deque(maxlen=max_buffer_size)
        
        # Statistics
        self.stats = {
            'frames_captured_0': 0,
            'frames_captured_1': 0,
            'frames_synced': 0,
            'frames_dropped': 0,
            'avg_sync_error_ms': 0.0,
            'sync_errors': deque(maxlen=100)
        }
        
        self.running = False
        self.capture_thread0 = None
        self.capture_thread1 = None
        self.lock = threading.Lock()
        
    def _capture_loop(self, camera, buffer_id):
        """Capture loop for a single camera"""
        buffer = self.buffer0 if buffer_id == 0 else self.buffer1
        stats_key = 'frames_captured_0' if buffer_id == 0 else 'frames_captured_1'
        
        while self.running:
            ret, frame = camera.read()
            if ret:
                timestamp = time.time()
                with self.lock:
                    buffer.append((timestamp, frame))
                    self.stats[stats_key] += 1
            else:
                time.sleep(0.001)  # Small delay on error
    
    def start(self):
        """Start synchronized capture threads"""
        if self.running:
            return
        
        self.running = True
        self.capture_thread0 = threading.Thread(
            target=self._capture_loop,
            args=(self.camera0, 0),
            daemon=True
        )
        self.capture_thread1 = threading.Thread(
            target=self._capture_loop,
            args=(self.camera1, 1),
            daemon=True
        )
        
        self.capture_thread0.start()
        self.capture_thread1.start()
    
    def stop(self):
        """Stop capture threads"""
        self.running = False
        if self.capture_thread0:
            self.capture_thread0.join(timeout=1.0)
        if self.capture_thread1:
            self.capture_thread1.join(timeout=1.0)
    
    def read_synchronized(self) -> Tuple[bool, Optional[cv2.Mat], Optional[cv2.Mat], float]:
        """
        Read synchronized frame pair
        
        Returns:
            (success, frame0, frame1, sync_error_ms)
        """
        with self.lock:
            if len(self.buffer0) == 0 or len(self.buffer1) == 0:
                return False, None, None, 0.0
            
            # Get most recent frames
            ts0, frame0 = self.buffer0[-1]
            ts1, frame1 = self.buffer1[-1]
            
            # Calculate sync error
            sync_error_ms = abs(ts0 - ts1) * 1000.0
            
            # Check if frames are synchronized
            if sync_error_ms <= self.sync_threshold_ms:
                # Remove used frames
                self.buffer0.clear()
                self.buffer1.clear()
                
                # Update statistics
                self.stats['frames_synced'] += 1
                self.stats['sync_errors'].append(sync_error_ms)
                self.stats['avg_sync_error_ms'] = np.mean(self.stats['sync_errors'])
                
                return True, frame0, frame1, sync_error_ms
            else:
                # Frames not synchronized, try to find better match
                best_match = self._find_best_match()
                if best_match:
                    frame0, frame1, sync_error_ms = best_match
                    self.stats['frames_synced'] += 1
                    self.stats['sync_errors'].append(sync_error_ms)
                    self.stats['avg_sync_error_ms'] = np.mean(self.stats['sync_errors'])
                    return True, frame0, frame1, sync_error_ms
                else:
                    self.stats['frames_dropped'] += 1
                    return False, None, None, sync_error_ms
    
    def _find_best_match(self) -> Optional[Tuple]:
        """Find best matching frame pair from buffers"""
        if len(self.buffer0) == 0 or len(self.buffer1) == 0:
            return None
        
        best_error = float('inf')
        best_pair = None
        
        for ts0, frame0 in self.buffer0:
            for ts1, frame1 in self.buffer1:
                error = abs(ts0 - ts1) * 1000.0
                if error < best_error and error <= self.sync_threshold_ms * 2:
                    best_error = error
                    best_pair = (frame0, frame1, error)
        
        if best_pair:
            # Clear buffers after finding match
            self.buffer0.clear()
            self.buffer1.clear()
        
        return best_pair
    
    def get_sync_quality(self) -> Dict:
        """Get synchronization quality metrics"""
        with self.lock:
            total_frames = self.stats['frames_captured_0'] + self.stats['frames_captured_1']
            sync_rate = (self.stats['frames_synced'] / max(total_frames / 2, 1)) * 100
            
            return {
                'sync_rate_percent': sync_rate,
                'frames_synced': self.stats['frames_synced'],
                'frames_dropped': self.stats['frames_dropped'],
                'avg_sync_error_ms': self.stats['avg_sync_error_ms'],
                'current_buffer_size_0': len(self.buffer0),
                'current_buffer_size_1': len(self.buffer1)
            }
    
    def reset_stats(self):
        """Reset statistics"""
        with self.lock:
            self.stats = {
                'frames_captured_0': 0,
                'frames_captured_1': 0,
                'frames_synced': 0,
                'frames_dropped': 0,
                'avg_sync_error_ms': 0.0,
                'sync_errors': deque(maxlen=100)
            }


class SimpleSynchronizedCapture:
    """Simpler synchronized capture without threading (for testing)"""
    
    def __init__(self, camera0, camera1):
        self.camera0 = camera0
        self.camera1 = camera1
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat], Optional[cv2.Mat]]:
        """Read synchronized frames (simple version)"""
        ret0, frame0 = self.camera0.read()
        ret1, frame1 = self.camera1.read()
        
        if ret0 and ret1:
            return True, frame0, frame1
        else:
            return False, frame0, frame1
