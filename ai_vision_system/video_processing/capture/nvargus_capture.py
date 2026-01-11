#!/usr/bin/env python3
"""
NVIDIA ARGUS Camera Capture (Based on nvgstcapture approach)
============================================================
Uses GStreamer subprocess to capture frames, matching the working nvgstcapture method.
"""

import cv2
import threading
import queue
import time
import subprocess
import numpy as np
from typing import Optional, Tuple
import os


class NVArgusCapture:
    """Camera capture using GStreamer subprocess (like nvgstcapture)"""
    
    def __init__(self, width: int = 1280, height: int = 720, fps: int = 30):
        """
        Initialize camera capture
        
        Args:
            width: Frame width
            height: Frame height
            fps: Target FPS
        """
        self.width = width
        self.height = height
        self.fps = fps
        
        self.process0: Optional[subprocess.Popen] = None
        self.process1: Optional[subprocess.Popen] = None
        
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
    
    def _create_gstreamer_command(self, sensor_id: int) -> list:
        """Create GStreamer command list (like nvgstcapture uses)"""
        # Use gst-launch-1.0 with the same pipeline format that works
        return [
            'gst-launch-1.0',
            '-q',  # Quiet mode
            'nvarguscamerasrc',
            f'sensor-id={sensor_id}',
            f'!', f'video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1',
            '!', 'nvvidconv',
            '!', 'video/x-raw,format=BGRx',
            '!', 'videoconvert',
            '!', 'video/x-raw,format=BGR',
            '!', 'fdsink', 'fd=1'  # Write to stdout
        ]
    
    def _capture_loop(self, process: subprocess.Popen, frame_queue: queue.Queue, camera_id: int):
        """Capture loop reading from subprocess stdout"""
        frames_key = f'frames_captured_{camera_id}'
        dropped_key = f'frames_dropped_{camera_id}'
        time_key = f'last_frame_time_{camera_id}'
        
        frame_size = self.width * self.height * 3  # BGR format
        buffer = bytearray()  # Accumulate partial reads
        
        while self.running and process.poll() is None:
            try:
                # Read available data (non-blocking)
                try:
                    chunk = process.stdout.read(65536)  # Read in chunks
                    if chunk is None:
                        time.sleep(0.01)
                        continue
                except BlockingIOError:
                    # No data available yet
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    time.sleep(0.01)
                    continue
                
                if len(chunk) == 0:
                    # Process ended
                    break
                
                # Accumulate data
                buffer.extend(chunk)
                
                # Process complete frames
                while len(buffer) >= frame_size:
                    # Extract one frame
                    frame_data = bytes(buffer[:frame_size])
                    buffer = buffer[frame_size:]
                    
                    # Convert to numpy array
                    frame = np.frombuffer(frame_data, dtype=np.uint8).reshape((self.height, self.width, 3))
                    
                    # Validate frame
                    if frame.size > 0 and np.std(frame) > 5:
                        try:
                            frame_queue.put_nowait(frame)
                            self.stats[frames_key] += 1
                            self.stats[time_key] = time.time()
                        except queue.Full:
                            try:
                                frame_queue.get_nowait()
                                frame_queue.put_nowait(frame)
                                self.stats[dropped_key] += 1
                                self.stats[frames_key] += 1
                                self.stats[time_key] = time.time()
                            except queue.Empty:
                                pass
                
            except Exception as e:
                time.sleep(0.01)
    
    def start(self) -> bool:
        """Start capture subprocesses"""
        if self.running:
            return True
        
        try:
            # Start GStreamer subprocesses
            print(f"Starting GStreamer capture for Camera 0...")
            cmd0 = self._create_gstreamer_command(0)
            self.process0 = subprocess.Popen(
                cmd0,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time
                stdin=subprocess.DEVNULL  # Prevent stdin blocking
            )
            # Make stdout non-blocking
            import fcntl
            flags = fcntl.fcntl(self.process0.stdout.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(self.process0.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            print(f"Starting GStreamer capture for Camera 1...")
            cmd1 = self._create_gstreamer_command(1)
            self.process1 = subprocess.Popen(
                cmd1,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=0,  # Unbuffered for real-time
                stdin=subprocess.DEVNULL  # Prevent stdin blocking
            )
            # Make stdout non-blocking
            flags = fcntl.fcntl(self.process1.stdout.fileno(), fcntl.F_GETFL)
            fcntl.fcntl(self.process1.stdout.fileno(), fcntl.F_SETFL, flags | os.O_NONBLOCK)
            
            # Wait a bit for processes to start
            time.sleep(1.0)
            
            # Check if processes are running
            if self.process0.poll() is not None or self.process1.poll() is not None:
                err0 = self.process0.stderr.read().decode() if self.process0.stderr else ""
                err1 = self.process1.stderr.read().decode() if self.process1.stderr else ""
                print(f"ERROR: GStreamer processes failed to start")
                print(f"Camera 0 error: {err0[:200]}")
                print(f"Camera 1 error: {err1[:200]}")
                self.stop()
                return False
            
            # Start capture threads
            self.running = True
            self.capture_thread0 = threading.Thread(
                target=self._capture_loop,
                args=(self.process0, self.frame_queue0, 0),
                daemon=True
            )
            self.capture_thread1 = threading.Thread(
                target=self._capture_loop,
                args=(self.process1, self.frame_queue1, 1),
                daemon=True
            )
            
            self.capture_thread0.start()
            self.capture_thread1.start()
            
            # Wait for first frames
            time.sleep(0.5)
            
            print("✓ GStreamer subprocess capture started")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to start capture: {e}")
            self.stop()
            return False
    
    def stop(self):
        """Stop capture subprocesses"""
        self.running = False
        
        if self.capture_thread0:
            self.capture_thread0.join(timeout=1.0)
        if self.capture_thread1:
            self.capture_thread1.join(timeout=1.0)
        
        if self.process0:
            self.process0.terminate()
            self.process0.wait(timeout=2.0)
            self.process0 = None
        
        if self.process1:
            self.process1.terminate()
            self.process1.wait(timeout=2.0)
            self.process1 = None
    
    def read(self, timeout: float = 1.0) -> Tuple[bool, Optional[np.ndarray], Optional[np.ndarray]]:
        """Read synchronized frames"""
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
        """Release resources"""
        self.stop()
