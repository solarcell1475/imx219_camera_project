#!/usr/bin/env python3
"""
IMX219 Depth Sensing - DEMO MODE (No Calibration Required)
===========================================================
This demo mode allows testing depth sensing WITHOUT calibration.

WARNING: Depth values will NOT be accurate without proper calibration!
This is for testing and visualization only.

Usage:
    python3 3_depth_sensing_demo.py [options]
    
Options:
    --resolution WxH      Resolution (default: 1280x720)
    --algorithm ALGO      Stereo algorithm: sgbm, bm (default: sgbm)
    --fps                 Show FPS counter
    
Controls:
    Q     - Quit
    S     - Save current frame
    C     - Cycle through color maps
"""

import cv2
import numpy as np
import argparse
import sys
from datetime import datetime

class DepthSensingDemo:
    def __init__(self, resolution, algorithm='sgbm', show_fps=False):
        self.resolution = resolution
        self.algorithm_name = algorithm
        self.show_fps = show_fps
        
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
        print("IMX219 Depth Sensing - DEMO MODE (No Calibration)")
        print("=" * 70)
        print("⚠️  WARNING: Running without calibration!")
        print("   Depth values will NOT be accurate.")
        print("   For accurate depth, run proper calibration first.")
        print("=" * 70)
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Algorithm: {algorithm.upper()}")
        print("=" * 70)
        
    def initialize_stereo_matcher(self):
        """Initialize stereo matching algorithm"""
        print(f"\nInitializing {self.algorithm_name.upper()} stereo matcher...")
        
        if self.algorithm_name == 'sgbm':
            # Semi-Global Block Matching
            self.min_disp = 0
            self.num_disp = 128  # Must be divisible by 16
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
            # Block Matching
            self.min_disp = 0
            self.num_disp = 128
            self.block_size = 15
            
            self.stereo = cv2.StereoBM_create(
                numDisparities=self.num_disp,
                blockSize=self.block_size
            )
            self.stereo.setPreFilterType(cv2.STEREO_BM_PREFILTER_NORMALIZED_RESPONSE)
            self.stereo.setPreFilterSize(5)
            self.stereo.setPreFilterCap(61)
            self.stereo.setTextureThreshold(10)
            self.stereo.setUniquenessRatio(15)
            self.stereo.setSpeckleRange(32)
            self.stereo.setSpeckleWindowSize(100)
        
        print("✓ Stereo matcher initialized")
        
    def initialize_cameras(self):
        """Initialize both cameras"""
        print("\nInitializing cameras...")
        
        # Use simple camera index (works on Jetson)
        self.cap0 = cv2.VideoCapture(0)
        self.cap1 = cv2.VideoCapture(1)
        
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            print("ERROR: Failed to open cameras!")
            sys.exit(1)
        
        # Set resolution
        self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print("✓ Cameras initialized")
        print(f"  Camera 0: {int(self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        print(f"  Camera 1: {int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
        
    def compute_disparity(self, left, right):
        """Compute disparity map"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Filter invalid disparities
        disparity[disparity <= self.min_disp] = 0
        disparity[disparity > self.num_disp] = 0
        
        # Normalize for visualization
        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Apply color map
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
        # Demo warning
        cv2.putText(image, "DEMO MODE - Not Calibrated", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # FPS
        if self.show_fps:
            cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Color map
        colormap_name = self.color_maps[self.color_map_idx][1]
        cv2.putText(image, f"Colormap: {colormap_name}", (10, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Controls
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
            print("⚠️  Remember: Depth values are NOT accurate without calibration!")
            print("=" * 70)
            
            while True:
                # Capture frames
                ret0, frame_left = self.cap0.read()
                ret1, frame_right = self.cap1.read()
                
                if not ret0 or not ret1:
                    print("ERROR: Failed to read from cameras")
                    break
                
                # Compute disparity (without rectification in demo mode)
                disparity, disparity_color = self.compute_disparity(frame_left, frame_right)
                
                # Create display
                left_display = self.draw_info(frame_left.copy())
                display = np.hstack((left_display, disparity_color))
                
                # Update FPS
                if self.show_fps:
                    self.update_fps()
                
                # Show display
                cv2.imshow('Depth Sensing Demo (Not Calibrated)', display)
                
                # Handle keyboard input
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
            print("\nFor ACCURATE depth sensing:")
            print("  1. python3 1_capture_calibration_images.py")
            print("  2. python3 2_calibrate_stereo_cameras.py")
            print("  3. python3 3_depth_sensing.py")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='IMX219 Depth Sensing Demo (No Calibration)')
    parser.add_argument('--resolution', default='1280x720',
                       help='Resolution WxH (default: 1280x720)')
    parser.add_argument('--algorithm', choices=['sgbm', 'bm'], default='sgbm',
                       help='Stereo matching algorithm (default: sgbm)')
    parser.add_argument('--fps', action='store_true',
                       help='Show FPS counter')
    
    args = parser.parse_args()
    
    app = DepthSensingDemo(
        args.resolution,
        args.algorithm,
        args.fps
    )
    app.run()

if __name__ == "__main__":
    main()
