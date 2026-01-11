#!/usr/bin/env python3
"""
IMX219 Depth Sensing - DEMO MODE V2 (Using GStreamer subprocess)
================================================================
This version uses GStreamer as subprocess to capture from IMX219 cameras.

Usage:
    python3 3_depth_sensing_demo_v2.py
"""

import cv2
import numpy as np
import subprocess
import threading
import queue
import sys
from datetime import datetime

class GStreamerCamera:
    """Camera capture using GStreamer subprocess"""
    
    def __init__(self, sensor_id, width, height, fps=30):
        self.sensor_id = sensor_id
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_queue = queue.Queue(maxsize=2)
        self.running = False
        self.process = None
        self.thread = None
        
    def start(self):
        """Start capture thread"""
        # GStreamer pipeline
        gst_cmd = [
            'gst-launch-1.0',
            '-q',  # Quiet mode
            f'nvarguscamerasrc sensor-id={self.sensor_id}',
            '!', f'video/x-raw(memory:NVMM),width={self.width},height={self.height},framerate={self.fps}/1',
            '!', 'nvvidconv',
            '!', 'video/x-raw,format=BGRx',
            '!', 'videoconvert',
            '!', 'video/x-raw,format=BGR',
            '!', 'fdsink'
        ]
        
        print(f"  Starting camera {self.sensor_id}...")
        
        try:
            self.process = subprocess.Popen(
                gst_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=self.width * self.height * 3
            )
            
            self.running = True
            self.thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.thread.start()
            
            return True
        except Exception as e:
            print(f"  ERROR starting camera {self.sensor_id}: {e}")
            return False
    
    def _capture_loop(self):
        """Capture loop running in thread"""
        frame_size = self.width * self.height * 3
        
        while self.running:
            try:
                # Read frame data
                raw_frame = self.process.stdout.read(frame_size)
                
                if len(raw_frame) != frame_size:
                    break
                
                # Convert to numpy array
                frame = np.frombuffer(raw_frame, dtype=np.uint8)
                frame = frame.reshape((self.height, self.width, 3))
                
                # Put in queue (drop old frames if full)
                if self.frame_queue.full():
                    try:
                        self.frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                
                self.frame_queue.put(frame)
                
            except Exception as e:
                if self.running:
                    print(f"Capture error camera {self.sensor_id}: {e}")
                break
    
    def read(self):
        """Read a frame"""
        try:
            frame = self.frame_queue.get(timeout=1.0)
            return True, frame
        except queue.Empty:
            return False, None
    
    def release(self):
        """Stop capture"""
        self.running = False
        if self.process:
            self.process.terminate()
            self.process.wait()
        if self.thread:
            self.thread.join(timeout=2.0)
    
    def isOpened(self):
        """Check if camera is running"""
        return self.running and self.process and self.process.poll() is None

class DepthSensingDemo:
    def __init__(self, resolution='1280x720', algorithm='sgbm'):
        self.resolution = resolution
        self.algorithm_name = algorithm
        
        # Parse resolution
        self.width, self.height = map(int, resolution.split('x'))
        
        # Color maps
        self.color_maps = [
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_TURBO, "TURBO"),
            (cv2.COLORMAP_HOT, "HOT"),
            (cv2.COLORMAP_RAINBOW, "RAINBOW"),
            (cv2.COLORMAP_VIRIDIS, "VIRIDIS")
        ]
        self.color_map_idx = 0
        self.color_map = self.color_maps[0][0]
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = None
        
        print("=" * 70)
        print("IMX219 Depth Sensing - DEMO MODE V2")
        print("=" * 70)
        print("⚠️  WARNING: Running without calibration!")
        print("   Depth values will NOT be accurate.")
        print("=" * 70)
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Algorithm: {algorithm.upper()}")
        print("=" * 70)
        
    def initialize_stereo_matcher(self):
        """Initialize stereo matching algorithm"""
        print(f"\nInitializing {self.algorithm_name.upper()} stereo matcher...")
        
        if self.algorithm_name == 'sgbm':
            self.min_disp = 0
            self.num_disp = 128
            self.block_size = 11
            
            self.stereo = cv2.StereoSGBM_create(
                minDisparity=self.min_disp,
                numDisparities=self.num_disp,
                blockSize=self.block_size,
                P1=8 * 3 * self.block_size ** 2,
                P2=32 * 3 * self.block_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                preFilterCap=63,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            self.min_disp = 0
            self.num_disp = 128
            self.block_size = 15
            
            self.stereo = cv2.StereoBM_create(
                numDisparities=self.num_disp,
                blockSize=self.block_size
            )
        
        print("✓ Stereo matcher initialized")
        
    def initialize_cameras(self):
        """Initialize both cameras using GStreamer"""
        print("\nInitializing cameras with GStreamer...")
        
        self.cap0 = GStreamerCamera(0, self.width, self.height)
        self.cap1 = GStreamerCamera(1, self.width, self.height)
        
        if not self.cap0.start():
            print("ERROR: Failed to start camera 0!")
            sys.exit(1)
        
        if not self.cap1.start():
            print("ERROR: Failed to start camera 1!")
            sys.exit(1)
        
        # Wait for cameras to start
        import time
        time.sleep(2)
        
        print("✓ Cameras initialized")
        
    def compute_disparity(self, left, right):
        """Compute disparity map"""
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        disparity[disparity <= self.min_disp] = 0
        disparity[disparity > self.num_disp] = 0
        
        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        disparity_color = cv2.applyColorMap(disparity_normalized, self.color_map)
        
        return disparity, disparity_color
    
    def update_fps(self):
        """Update FPS counter"""
        if self.fps_start_time is None:
            self.fps_start_time = cv2.getTickCount()
        
        self.frame_count += 1
        
        if self.frame_count >= 30:
            time_elapsed = (cv2.getTickCount() - self.fps_start_time) / cv2.getTickFrequency()
            self.fps = self.frame_count / time_elapsed
            self.frame_count = 0
            self.fps_start_time = cv2.getTickCount()
    
    def draw_info(self, image):
        """Draw information overlay"""
        cv2.putText(image, "DEMO MODE - Not Calibrated", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        colormap_name = self.color_maps[self.color_map_idx][1]
        cv2.putText(image, f"Colormap: {colormap_name}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.putText(image, "Q:Quit S:Save C:Colormap", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def save_frame(self, left, disparity_color):
        """Save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"demo_left_{timestamp}.jpg", left)
        cv2.imwrite(f"demo_disparity_{timestamp}.jpg", disparity_color)
        print(f"✓ Saved frame: demo_*_{timestamp}.jpg")
    
    def run(self):
        """Main processing loop"""
        try:
            self.initialize_stereo_matcher()
            self.initialize_cameras()
            
            print("\n" + "=" * 70)
            print("Starting depth sensing demo...")
            print("Controls: Q=Quit, S=Save, C=Colormap")
            print("=" * 70)
            print()
            
            frame_skip = 0
            
            while True:
                # Capture frames
                ret0, frame_left = self.cap0.read()
                ret1, frame_right = self.cap1.read()
                
                if not ret0 or not ret1:
                    # Skip first few frames as cameras warm up
                    frame_skip += 1
                    if frame_skip > 10:
                        print("ERROR: Failed to read from cameras")
                        break
                    continue
                
                frame_skip = 0
                
                # Compute disparity
                disparity, disparity_color = self.compute_disparity(frame_left, frame_right)
                
                # Create display
                left_display = self.draw_info(frame_left.copy())
                display = np.hstack((left_display, disparity_color))
                
                # Update FPS
                self.update_fps()
                
                # Show display
                cv2.imshow('Depth Sensing Demo', display)
                
                # Handle keyboard
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_frame(frame_left, disparity_color)
                elif key == ord('c') or key == ord('C'):
                    self.color_map_idx = (self.color_map_idx + 1) % len(self.color_maps)
                    self.color_map = self.color_maps[self.color_map_idx][0]
            
            # Cleanup
            self.cap0.release()
            self.cap1.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 70)
            print("Depth sensing demo stopped")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            if hasattr(self, 'cap0'):
                self.cap0.release()
            if hasattr(self, 'cap1'):
                self.cap1.release()
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()

def main():
    app = DepthSensingDemo(resolution='1280x720', algorithm='sgbm')
    app.run()

if __name__ == "__main__":
    main()
