#!/usr/bin/env python3
"""
IMX219 Stereo Camera Calibration
=================================
This script computes stereo camera calibration parameters from captured images.

Usage:
    python3 2_calibrate_stereo_cameras.py

Output:
    - stereo_calibration.npz (calibration parameters)
    - calibration_report.txt (calibration quality report)
"""

import cv2
import numpy as np
import os
import glob
import sys
from datetime import datetime

class StereoCalibrator:
    def __init__(self):
        # Directories
        self.left_dir = "calibration_images/left"
        self.right_dir = "calibration_images/right"
        self.output_file = "stereo_calibration.npz"
        self.report_file = "calibration_report.txt"
        
        # Checkerboard pattern (must match capture script)
        self.pattern_size = (9, 6)  # 9x6 internal corners
        self.square_size = 25.0  # mm
        
        # Termination criteria for corner refinement
        self.criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        
        # 3D points of checkerboard corners in real world coordinates
        self.objp = np.zeros((self.pattern_size[0] * self.pattern_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.pattern_size[0], 0:self.pattern_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size
        
        print("=" * 70)
        print("IMX219 Stereo Camera Calibration")
        print("=" * 70)
        print(f"Checkerboard: {self.pattern_size[0]}x{self.pattern_size[1]} corners")
        print(f"Square size: {self.square_size}mm")
        print("=" * 70)
        
    def load_images(self):
        """Load calibration images"""
        print("\nLoading calibration images...")
        
        if not os.path.exists(self.left_dir) or not os.path.exists(self.right_dir):
            print(f"ERROR: Calibration image directories not found!")
            print(f"Expected: {self.left_dir}/ and {self.right_dir}/")
            print("Run 1_capture_calibration_images.py first!")
            sys.exit(1)
        
        left_images = sorted(glob.glob(os.path.join(self.left_dir, "*.jpg")))
        right_images = sorted(glob.glob(os.path.join(self.right_dir, "*.jpg")))
        
        if len(left_images) == 0 or len(right_images) == 0:
            print("ERROR: No calibration images found!")
            print("Run 1_capture_calibration_images.py first!")
            sys.exit(1)
        
        if len(left_images) != len(right_images):
            print(f"WARNING: Image count mismatch!")
            print(f"  Left: {len(left_images)}, Right: {len(right_images)}")
        
        print(f"✓ Found {len(left_images)} left images")
        print(f"✓ Found {len(right_images)} right images")
        
        return left_images, right_images
    
    def find_corners(self, images, camera_name):
        """Find checkerboard corners in images"""
        print(f"\nProcessing {camera_name} camera images...")
        
        objpoints = []  # 3D points in real world
        imgpoints = []  # 2D points in image plane
        
        valid_count = 0
        
        for i, image_path in enumerate(images):
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret, corners = cv2.findChessboardCorners(
                gray,
                self.pattern_size,
                cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
            )
            
            if ret:
                # Refine corner positions to sub-pixel accuracy
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                
                objpoints.append(self.objp)
                imgpoints.append(corners_refined)
                valid_count += 1
                print(f"  {i+1}/{len(images)} ✓")
            else:
                print(f"  {i+1}/{len(images)} ✗ (pattern not found)")
        
        print(f"✓ {camera_name}: {valid_count}/{len(images)} valid images")
        
        # Get image size from first image
        img_size = (gray.shape[1], gray.shape[0])
        
        return objpoints, imgpoints, img_size, valid_count
    
    def calibrate_camera(self, objpoints, imgpoints, img_size, camera_name):
        """Calibrate individual camera"""
        print(f"\nCalibrating {camera_name} camera...")
        
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        
        # Calculate reprojection error
        mean_error = 0
        for i in range(len(objpoints)):
            imgpoints_projected, _ = cv2.projectPoints(
                objpoints[i], rvecs[i], tvecs[i], camera_matrix, dist_coeffs
            )
            error = cv2.norm(imgpoints[i], imgpoints_projected, cv2.NORM_L2) / len(imgpoints_projected)
            mean_error += error
        mean_error /= len(objpoints)
        
        print(f"✓ {camera_name} calibration complete")
        print(f"  Reprojection error: {mean_error:.4f} pixels")
        
        return camera_matrix, dist_coeffs, mean_error
    
    def stereo_calibrate(self, objpoints, imgpoints_left, imgpoints_right, 
                         camera_matrix_left, dist_left, camera_matrix_right, dist_right, img_size):
        """Perform stereo calibration"""
        print("\nPerforming stereo calibration...")
        
        # Stereo calibration flags
        flags = cv2.CALIB_FIX_INTRINSIC
        
        # Stereo calibration
        ret, camera_matrix_left, dist_left, camera_matrix_right, dist_right, R, T, E, F = cv2.stereoCalibrate(
            objpoints,
            imgpoints_left,
            imgpoints_right,
            camera_matrix_left,
            dist_left,
            camera_matrix_right,
            dist_right,
            img_size,
            criteria=self.criteria,
            flags=flags
        )
        
        print(f"✓ Stereo calibration complete")
        print(f"  Stereo reprojection error: {ret:.4f}")
        
        # Stereo rectification
        print("\nComputing rectification parameters...")
        R1, R2, P1, P2, Q, roi_left, roi_right = cv2.stereoRectify(
            camera_matrix_left,
            dist_left,
            camera_matrix_right,
            dist_right,
            img_size,
            R,
            T,
            alpha=0  # 0=crop to valid pixels, 1=keep all pixels
        )
        
        print("✓ Rectification complete")
        
        return R, T, E, F, R1, R2, P1, P2, Q, roi_left, roi_right, ret
    
    def save_calibration(self, camera_matrix_left, dist_left, camera_matrix_right, dist_right,
                         R, T, E, F, R1, R2, P1, P2, Q, roi_left, roi_right, img_size):
        """Save calibration parameters"""
        print(f"\nSaving calibration to {self.output_file}...")
        
        np.savez(
            self.output_file,
            camera_matrix_left=camera_matrix_left,
            dist_left=dist_left,
            camera_matrix_right=camera_matrix_right,
            dist_right=dist_right,
            R=R,
            T=T,
            E=E,
            F=F,
            R1=R1,
            R2=R2,
            P1=P1,
            P2=P2,
            Q=Q,
            roi_left=roi_left,
            roi_right=roi_right,
            img_size=img_size
        )
        
        print(f"✓ Calibration saved")
    
    def generate_report(self, error_left, error_right, stereo_error, T, baseline_mm):
        """Generate calibration quality report"""
        print(f"\nGenerating calibration report...")
        
        report = []
        report.append("=" * 70)
        report.append("IMX219 Stereo Camera Calibration Report")
        report.append("=" * 70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        report.append("Calibration Quality:")
        report.append(f"  Left camera reprojection error:   {error_left:.4f} pixels")
        report.append(f"  Right camera reprojection error:  {error_right:.4f} pixels")
        report.append(f"  Stereo reprojection error:        {stereo_error:.4f}")
        report.append("")
        report.append("Stereo Geometry:")
        report.append(f"  Baseline (distance between cameras): {baseline_mm:.2f} mm")
        report.append(f"  Translation vector (mm):")
        report.append(f"    X: {T[0][0]:.2f}")
        report.append(f"    Y: {T[1][0]:.2f}")
        report.append(f"    Z: {T[2][0]:.2f}")
        report.append("")
        report.append("Quality Assessment:")
        
        # Quality assessment
        if error_left < 0.5 and error_right < 0.5 and stereo_error < 0.5:
            quality = "EXCELLENT"
            recommendation = "Ready for high-precision depth sensing"
        elif error_left < 1.0 and error_right < 1.0 and stereo_error < 1.0:
            quality = "GOOD"
            recommendation = "Suitable for most depth sensing applications"
        elif error_left < 2.0 and error_right < 2.0 and stereo_error < 2.0:
            quality = "FAIR"
            recommendation = "May work but consider recalibration for better accuracy"
        else:
            quality = "POOR"
            recommendation = "Recalibration recommended - capture more/better images"
        
        report.append(f"  Overall Quality: {quality}")
        report.append(f"  Recommendation: {recommendation}")
        report.append("")
        report.append("Next Steps:")
        report.append("  1. Run depth sensing application:")
        report.append("     python3 3_depth_sensing.py")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file
        with open(self.report_file, 'w') as f:
            f.write(report_text)
        
        # Print to console
        print("\n" + report_text)
        
        print(f"\n✓ Report saved to {self.report_file}")
    
    def run(self):
        """Main calibration process"""
        try:
            # Load images
            left_images, right_images = self.load_images()
            
            # Find corners in left images
            objpoints_left, imgpoints_left, img_size, valid_left = self.find_corners(
                left_images, "Left"
            )
            
            # Find corners in right images
            objpoints_right, imgpoints_right, img_size_right, valid_right = self.find_corners(
                right_images, "Right"
            )
            
            # Ensure we have matching pairs
            min_valid = min(valid_left, valid_right)
            if min_valid < 10:
                print(f"\nERROR: Not enough valid image pairs ({min_valid})")
                print("Capture at least 10 valid calibration images!")
                sys.exit(1)
            
            # Use same number of points for both cameras
            objpoints_left = objpoints_left[:min_valid]
            imgpoints_left = imgpoints_left[:min_valid]
            objpoints_right = objpoints_right[:min_valid]
            imgpoints_right = imgpoints_right[:min_valid]
            
            # Calibrate left camera
            camera_matrix_left, dist_left, error_left = self.calibrate_camera(
                objpoints_left, imgpoints_left, img_size, "Left"
            )
            
            # Calibrate right camera
            camera_matrix_right, dist_right, error_right = self.calibrate_camera(
                objpoints_right, imgpoints_right, img_size, "Right"
            )
            
            # Stereo calibration
            R, T, E, F, R1, R2, P1, P2, Q, roi_left, roi_right, stereo_error = self.stereo_calibrate(
                objpoints_left,
                imgpoints_left,
                imgpoints_right,
                camera_matrix_left,
                dist_left,
                camera_matrix_right,
                dist_right,
                img_size
            )
            
            # Calculate baseline (distance between cameras)
            baseline_mm = np.linalg.norm(T)
            
            # Save calibration
            self.save_calibration(
                camera_matrix_left, dist_left,
                camera_matrix_right, dist_right,
                R, T, E, F, R1, R2, P1, P2, Q,
                roi_left, roi_right, img_size
            )
            
            # Generate report
            self.generate_report(error_left, error_right, stereo_error, T, baseline_mm)
            
            print("\n✓ Calibration complete!")
            print(f"\nCalibration file: {self.output_file}")
            print(f"Report file: {self.report_file}")
            
        except Exception as e:
            print(f"\nERROR: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)

def main():
    calibrator = StereoCalibrator()
    calibrator.run()

if __name__ == "__main__":
    main()
