#!/usr/bin/env python3
"""
IMX219 Real-Time Depth Sensing Application
===========================================
This application provides real-time depth sensing using stereo cameras.

Usage:
    python3 3_depth_sensing.py [options]
    
Options:
    --calibration FILE    Calibration file (default: stereo_calibration.npz)
    --resolution WxH      Resolution (default: 1280x720)
    --algorithm ALGO      Stereo algorithm: sgbm, bm (default: sgbm)
    --save-video FILE     Save output to video file
    --fps                 Show FPS counter
    
Controls:
    Q     - Quit
    S     - Save current frame
    D     - Toggle depth map display
    C     - Cycle through color maps
    +/-   - Adjust disparity parameters
    R     - Reset parameters
"""

import cv2
import numpy as np
import argparse
import sys
import os
from datetime import datetime

class DepthSensingApp:
    def __init__(self, calibration_file, resolution, algorithm='sgbm', save_video=None, show_fps=False):
        self.calibration_file = calibration_file
        self.resolution = resolution
        self.algorithm_name = algorithm
        self.save_video = save_video
        self.show_fps = show_fps
        
        # Parse resolution
        self.width, self.height = map(int, resolution.split('x'))
        
        # Display modes
        self.display_mode = 0  # 0=stereo+depth, 1=depth only, 2=point cloud
        self.color_map = cv2.COLORMAP_JET
        self.color_maps = [
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_TURBO, "TURBO"),
            (cv2.COLORMAP_HOT, "HOT"),
            (cv2.COLORMAP_RAINBOW, "RAINBOW"),
            (cv2.COLORMAP_VIRIDIS, "VIRIDIS")
        ]
        self.color_map_idx = 0
        
        # FPS tracking
        self.fps = 0
        self.frame_count = 0
        self.fps_start_time = None
        
        # Video writer
        self.video_writer = None
        
        print("=" * 70)
        print("IMX219 Real-Time Depth Sensing")
        print("=" * 70)
        print(f"Resolution: {self.width}x{self.height}")
        print(f"Algorithm: {algorithm.upper()}")
        if save_video:
            print(f"Saving to: {save_video}")
        print("=" * 70)
        
    def load_calibration(self):
        """Load stereo calibration parameters"""
        print(f"\nLoading calibration from {self.calibration_file}...")
        
        if not os.path.exists(self.calibration_file):
            print(f"ERROR: Calibration file not found: {self.calibration_file}")
            print("Run calibration first:")
            print("  1. python3 1_capture_calibration_images.py")
            print("  2. python3 2_calibrate_stereo_cameras.py")
            sys.exit(1)
        
        # Load calibration data
        calib_data = np.load(self.calibration_file)
        
        self.camera_matrix_left = calib_data['camera_matrix_left']
        self.dist_left = calib_data['dist_left']
        self.camera_matrix_right = calib_data['camera_matrix_right']
        self.dist_right = calib_data['dist_right']
        self.R = calib_data['R']
        self.T = calib_data['T']
        self.R1 = calib_data['R1']
        self.R2 = calib_data['R2']
        self.P1 = calib_data['P1']
        self.P2 = calib_data['P2']
        self.Q = calib_data['Q']
        
        # Calculate baseline
        self.baseline = np.linalg.norm(self.T)
        
        print(f"✓ Calibration loaded")
        print(f"  Baseline: {self.baseline:.2f} mm")
        
        # Compute rectification maps
        print("Computing rectification maps...")
        img_size = (self.width, self.height)
        
        self.map_left_x, self.map_left_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_left, self.dist_left, self.R1, self.P1,
            img_size, cv2.CV_32FC1
        )
        
        self.map_right_x, self.map_right_y = cv2.initUndistortRectifyMap(
            self.camera_matrix_right, self.dist_right, self.R2, self.P2,
            img_size, cv2.CV_32FC1
        )
        
        print("✓ Rectification maps ready")
        
    def initialize_stereo_matcher(self):
        """Initialize stereo matching algorithm"""
        print(f"\nInitializing {self.algorithm_name.upper()} stereo matcher...")
        
        if self.algorithm_name == 'sgbm':
            # Semi-Global Block Matching (better quality, slower)
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
        elif self.algorithm_name == 'bm':
            # Block Matching (faster, lower quality)
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
        else:
            print(f"ERROR: Unknown algorithm: {self.algorithm_name}")
            sys.exit(1)
        
        print("✓ Stereo matcher initialized")
        print(f"  Min disparity: {self.min_disp}")
        print(f"  Num disparities: {self.num_disp}")
        print(f"  Block size: {self.block_size}")
        
    def initialize_cameras(self):
        """Initialize both cameras"""
        print("\nInitializing cameras...")
        
        # GStreamer pipeline for camera 0 (left)
        gst_str_0 = (
            f"nvarguscamerasrc sensor-id=0 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        # GStreamer pipeline for camera 1 (right)
        gst_str_1 = (
            f"nvarguscamerasrc sensor-id=1 ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        self.cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
        self.cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)
        
        if not self.cap0.isOpened() or not self.cap1.isOpened():
            print("ERROR: Failed to open cameras!")
            sys.exit(1)
        
        print("✓ Cameras initialized")
        
        # Initialize video writer if requested
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(
                self.save_video, fourcc, 30.0, 
                (self.width * 2, self.height)
            )
            print(f"✓ Video writer initialized: {self.save_video}")
        
    def compute_depth(self, left_rect, right_rect):
        """Compute disparity and depth map"""
        # Convert to grayscale for stereo matching
        left_gray = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY)
        
        # Compute disparity
        disparity = self.stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
        
        # Filter invalid disparities
        disparity[disparity <= self.min_disp] = 0
        disparity[disparity > self.num_disp] = 0
        
        # Normalize disparity for visualization
        disparity_normalized = cv2.normalize(
            disparity, None, alpha=0, beta=255,
            norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Apply color map
        disparity_color = cv2.applyColorMap(disparity_normalized, self.color_map)
        
        # Calculate actual depth (in mm)
        # Depth = (focal_length * baseline) / disparity
        # Using Q matrix for better accuracy
        depth_map = cv2.reprojectImageTo3D(disparity, self.Q)
        
        return disparity, disparity_color, depth_map
    
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
    
    def draw_info(self, image, disparity, depth_map, mouse_x=None, mouse_y=None):
        """Draw information overlay on image"""
        # FPS
        if self.show_fps:
            cv2.putText(image, f"FPS: {self.fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Color map name
        colormap_name = self.color_maps[self.color_map_idx][1]
        cv2.putText(image, f"Colormap: {colormap_name}", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Depth at cursor (if provided)
        if mouse_x is not None and mouse_y is not None:
            if 0 <= mouse_y < depth_map.shape[0] and 0 <= mouse_x < depth_map.shape[1]:
                depth_value = depth_map[mouse_y, mouse_x, 2]  # Z coordinate
                if not np.isnan(depth_value) and not np.isinf(depth_value) and depth_value > 0:
                    depth_m = abs(depth_value) / 1000.0  # Convert mm to meters
                    cv2.putText(image, f"Depth: {depth_m:.2f}m", (10, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # Draw crosshair
                    cv2.drawMarker(image, (mouse_x, mouse_y), (0, 255, 255),
                                 cv2.MARKER_CROSS, 20, 2)
        
        # Controls
        cv2.putText(image, "Q:Quit S:Save D:Mode C:Colormap", (10, self.height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return image
    
    def save_frame(self, left_rect, disparity_color, depth_map):
        """Save current frame"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save images
        cv2.imwrite(f"depth_left_{timestamp}.jpg", left_rect)
        cv2.imwrite(f"depth_disparity_{timestamp}.jpg", disparity_color)
        
        # Save depth map as numpy array
        np.save(f"depth_map_{timestamp}.npy", depth_map)
        
        print(f"✓ Saved frame: depth_*_{timestamp}.*")
    
    def run(self):
        """Main processing loop"""
        try:
            self.load_calibration()
            self.initialize_stereo_matcher()
            self.initialize_cameras()
            
            print("\n" + "=" * 70)
            print("Starting depth sensing...")
            print("Controls: Q=Quit, S=Save, D=Mode, C=Colormap")
            print("=" * 70)
            
            # Mouse tracking for depth at cursor
            mouse_x, mouse_y = None, None
            
            def mouse_callback(event, x, y, flags, param):
                nonlocal mouse_x, mouse_y
                if event == cv2.EVENT_MOUSEMOVE:
                    # Adjust for left image in side-by-side display
                    if x < self.width:
                        mouse_x, mouse_y = x, y
                    else:
                        mouse_x, mouse_y = x - self.width, y
            
            cv2.namedWindow('Depth Sensing')
            cv2.setMouseCallback('Depth Sensing', mouse_callback)
            
            while True:
                # Capture frames
                ret0, frame_left = self.cap0.read()
                ret1, frame_right = self.cap1.read()
                
                if not ret0 or not ret1:
                    print("ERROR: Failed to read from cameras")
                    break
                
                # Rectify images
                left_rect = cv2.remap(frame_left, self.map_left_x, self.map_left_y, cv2.INTER_LINEAR)
                right_rect = cv2.remap(frame_right, self.map_right_x, self.map_right_y, cv2.INTER_LINEAR)
                
                # Compute depth
                disparity, disparity_color, depth_map = self.compute_depth(left_rect, right_rect)
                
                # Create display
                if self.display_mode == 0:
                    # Side by side: stereo + depth
                    left_display = self.draw_info(left_rect.copy(), disparity, depth_map, mouse_x, mouse_y)
                    display = np.hstack((left_display, disparity_color))
                elif self.display_mode == 1:
                    # Full depth map
                    display = cv2.resize(disparity_color, (self.width * 2, self.height))
                
                # Update FPS
                if self.show_fps:
                    self.update_fps()
                
                # Save to video if requested
                if self.video_writer:
                    self.video_writer.write(display)
                
                # Show display
                cv2.imshow('Depth Sensing', display)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('s') or key == ord('S'):
                    self.save_frame(left_rect, disparity_color, depth_map)
                elif key == ord('d') or key == ord('D'):
                    self.display_mode = (self.display_mode + 1) % 2
                elif key == ord('c') or key == ord('C'):
                    self.color_map_idx = (self.color_map_idx + 1) % len(self.color_maps)
                    self.color_map = self.color_maps[self.color_map_idx][0]
            
            # Cleanup
            self.cap0.release()
            self.cap1.release()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            
            print("\n" + "=" * 70)
            print("Depth sensing stopped")
            print("=" * 70)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='IMX219 Real-Time Depth Sensing')
    parser.add_argument('--calibration', default='stereo_calibration.npz',
                       help='Calibration file (default: stereo_calibration.npz)')
    parser.add_argument('--resolution', default='1280x720',
                       help='Resolution WxH (default: 1280x720)')
    parser.add_argument('--algorithm', choices=['sgbm', 'bm'], default='sgbm',
                       help='Stereo matching algorithm (default: sgbm)')
    parser.add_argument('--save-video', help='Save output to video file')
    parser.add_argument('--fps', action='store_true',
                       help='Show FPS counter')
    
    args = parser.parse_args()
    
    app = DepthSensingApp(
        args.calibration,
        args.resolution,
        args.algorithm,
        args.save_video,
        args.fps
    )
    app.run()

if __name__ == "__main__":
    main()
