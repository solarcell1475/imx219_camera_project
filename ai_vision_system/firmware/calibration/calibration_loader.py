#!/usr/bin/env python3
"""
Calibration Integration Module
================================
Loads and manages stereo camera calibration parameters for optional
rectification in AI processing.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple


class CalibrationLoader:
    """Load and manage camera calibration parameters"""
    
    def __init__(self, calibration_file: Optional[str] = None):
        """
        Initialize calibration loader
        
        Args:
            calibration_file: Path to calibration .npz file
        """
        self.calibration_file = calibration_file
        self.calibration_data = None
        self.rectification_maps = None
        self.loaded = False
        
        # Calibration parameters
        self.camera_matrix_left = None
        self.camera_matrix_right = None
        self.dist_coeffs_left = None
        self.dist_coeffs_right = None
        self.R = None  # Rotation matrix
        self.T = None  # Translation vector
        self.E = None  # Essential matrix
        self.F = None  # Fundamental matrix
        
        # Rectification parameters
        self.R1 = None
        self.R2 = None
        self.P1 = None
        self.P2 = None
        self.Q = None  # Disparity-to-depth mapping matrix
        
        if calibration_file:
            self.load(calibration_file)
    
    def load(self, calibration_file: str) -> bool:
        """Load calibration from file"""
        file_path = Path(calibration_file)
        
        # Try default location if not found
        if not file_path.exists():
            default_path = Path("/home/jetson/Downloads/IMX219_Camera_Project/depth_sensing/stereo_calibration.npz")
            if default_path.exists():
                file_path = default_path
            else:
                print(f"Warning: Calibration file not found: {calibration_file}")
                return False
        
        try:
            self.calibration_data = np.load(str(file_path), allow_pickle=True)
            
            # Load camera matrices
            self.camera_matrix_left = self.calibration_data.get('camera_matrix_left')
            self.camera_matrix_right = self.calibration_data.get('camera_matrix_right')
            self.dist_coeffs_left = self.calibration_data.get('dist_coeffs_left')
            self.dist_coeffs_right = self.calibration_data.get('dist_coeffs_right')
            
            # Load stereo parameters
            self.R = self.calibration_data.get('R')
            self.T = self.calibration_data.get('T')
            self.E = self.calibration_data.get('E')
            self.F = self.calibration_data.get('F')
            
            # Load rectification parameters
            self.R1 = self.calibration_data.get('R1')
            self.R2 = self.calibration_data.get('R2')
            self.P1 = self.calibration_data.get('P1')
            self.P2 = self.calibration_data.get('P2')
            self.Q = self.calibration_data.get('Q')
            
            self.calibration_file = str(file_path)
            self.loaded = True
            
            print(f"✓ Calibration loaded from: {file_path}")
            return True
            
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def compute_rectification_maps(self, image_size: Tuple[int, int]) -> bool:
        """
        Compute rectification maps for given image size
        
        Args:
            image_size: (width, height) tuple
        """
        if not self.loaded:
            print("Error: Calibration not loaded")
            return False
        
        if self.R1 is None or self.R2 is None or self.P1 is None or self.P2 is None:
            print("Error: Rectification parameters not available in calibration file")
            return False
        
        width, height = image_size
        
        # Compute rectification maps
        map1_left, map2_left = cv2.initUndistortRectifyMap(
            self.camera_matrix_left,
            self.dist_coeffs_left,
            self.R1,
            self.P1,
            (width, height),
            cv2.CV_32FC1
        )
        
        map1_right, map2_right = cv2.initUndistortRectifyMap(
            self.camera_matrix_right,
            self.dist_coeffs_right,
            self.R2,
            self.P2,
            (width, height),
            cv2.CV_32FC1
        )
        
        self.rectification_maps = {
            'left': (map1_left, map2_left),
            'right': (map1_right, map2_right)
        }
        
        return True
    
    def rectify_images(self, left_image: np.ndarray, right_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Rectify stereo image pair
        
        Args:
            left_image: Left camera image
            right_image: Right camera image
        
        Returns:
            (rectified_left, rectified_right)
        """
        if self.rectification_maps is None:
            # Compute maps if not already computed
            h, w = left_image.shape[:2]
            if not self.compute_rectification_maps((w, h)):
                return left_image, right_image
        
        map1_left, map2_left = self.rectification_maps['left']
        map1_right, map2_right = self.rectification_maps['right']
        
        rectified_left = cv2.remap(
            left_image, map1_left, map2_left,
            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
        )
        
        rectified_right = cv2.remap(
            right_image, map1_right, map2_right,
            cv2.INTER_LINEAR, cv2.BORDER_CONSTANT
        )
        
        return rectified_left, rectified_right
    
    def get_calibration_info(self) -> Dict:
        """Get calibration information"""
        if not self.loaded:
            return {'loaded': False}
        
        info = {
            'loaded': True,
            'calibration_file': self.calibration_file,
            'has_rectification': self.R1 is not None and self.R2 is not None,
            'has_rectification_maps': self.rectification_maps is not None
        }
        
        if self.camera_matrix_left is not None:
            info['focal_length_left'] = (
                float(self.camera_matrix_left[0, 0]),
                float(self.camera_matrix_left[1, 1])
            )
            info['principal_point_left'] = (
                float(self.camera_matrix_left[0, 2]),
                float(self.camera_matrix_left[1, 2])
            )
        
        if self.T is not None:
            baseline = np.linalg.norm(self.T)
            info['baseline_mm'] = float(baseline)
        
        return info
    
    def is_available(self) -> bool:
        """Check if calibration is available"""
        return self.loaded


def main():
    """Test function"""
    print("Testing Calibration Loader...")
    
    # Try to load calibration
    loader = CalibrationLoader()
    
    # Try default location
    default_path = "/home/jetson/Downloads/IMX219_Camera_Project/depth_sensing/stereo_calibration.npz"
    if Path(default_path).exists():
        if loader.load(default_path):
            info = loader.get_calibration_info()
            print(f"\nCalibration Info:")
            print(f"  Loaded: {info['loaded']}")
            print(f"  Has rectification: {info.get('has_rectification', False)}")
            if 'baseline_mm' in info:
                print(f"  Baseline: {info['baseline_mm']:.2f} mm")
            return 0
        else:
            print("Failed to load calibration")
            return 1
    else:
        print(f"Calibration file not found: {default_path}")
        print("Run calibration first if needed")
        return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
