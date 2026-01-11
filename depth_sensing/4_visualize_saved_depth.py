#!/usr/bin/env python3
"""
Depth Map Visualization Tool
=============================
Visualize and analyze saved depth maps.

Usage:
    python3 4_visualize_saved_depth.py <depth_map.npy>
"""

import cv2
import numpy as np
import sys
import argparse

def visualize_depth_map(depth_file):
    """Visualize a saved depth map"""
    print(f"Loading depth map: {depth_file}")
    
    # Load depth map
    depth_map = np.load(depth_file)
    
    print(f"Depth map shape: {depth_map.shape}")
    print(f"Depth map dtype: {depth_map.dtype}")
    
    # Extract Z (depth) values
    depth_values = depth_map[:, :, 2]
    
    # Filter invalid values
    valid_mask = ~(np.isnan(depth_values) | np.isinf(depth_values))
    valid_depths = depth_values[valid_mask]
    
    if len(valid_depths) > 0:
        min_depth = np.min(valid_depths)
        max_depth = np.max(valid_depths)
        mean_depth = np.mean(valid_depths)
        
        print(f"\nDepth Statistics:")
        print(f"  Min depth: {abs(min_depth)/1000:.2f} m")
        print(f"  Max depth: {abs(max_depth)/1000:.2f} m")
        print(f"  Mean depth: {abs(mean_depth)/1000:.2f} m")
        print(f"  Valid pixels: {np.sum(valid_mask)} / {depth_values.size} ({100*np.sum(valid_mask)/depth_values.size:.1f}%)")
        
        # Normalize for visualization
        depth_normalized = np.zeros_like(depth_values, dtype=np.uint8)
        depth_normalized[valid_mask] = cv2.normalize(
            valid_depths, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U
        )
        
        # Apply colormap
        colormaps = [
            (cv2.COLORMAP_JET, "JET"),
            (cv2.COLORMAP_TURBO, "TURBO"),
            (cv2.COLORMAP_HOT, "HOT"),
            (cv2.COLORMAP_RAINBOW, "RAINBOW"),
            (cv2.COLORMAP_VIRIDIS, "VIRIDIS")
        ]
        
        print("\nPress number keys 1-5 to change colormap, Q to quit")
        
        colormap_idx = 0
        
        while True:
            depth_color = cv2.applyColorMap(depth_normalized, colormaps[colormap_idx][0])
            
            # Add text
            cv2.putText(depth_color, f"Colormap: {colormaps[colormap_idx][1]}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(depth_color, f"Depth: {abs(min_depth)/1000:.2f}m - {abs(max_depth)/1000:.2f}m", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Depth Map Visualization', depth_color)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif ord('1') <= key <= ord('5'):
                colormap_idx = key - ord('1')
        
        cv2.destroyAllWindows()
    else:
        print("ERROR: No valid depth values found!")

def main():
    parser = argparse.ArgumentParser(description='Visualize saved depth maps')
    parser.add_argument('depth_file', help='Depth map file (.npy)')
    args = parser.parse_args()
    
    visualize_depth_map(args.depth_file)

if __name__ == "__main__":
    main()
