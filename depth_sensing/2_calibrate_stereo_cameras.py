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
from pathlib import Path

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
        
        def pair_key(path, prefix):
            name = Path(path).stem
            expected_prefix = f"{prefix}_"
            return name[len(expected_prefix):] if name.startswith(expected_prefix) else name

        left_by_key = {pair_key(path, "left"): path for path in left_images}
        right_by_key = {pair_key(path, "right"): path for path in right_images}
        matched_keys = sorted(left_by_key.keys() & right_by_key.keys())
        unmatched_left = sorted(left_by_key.keys() - right_by_key.keys())
        unmatched_right = sorted(right_by_key.keys() - left_by_key.keys())

        print(f"✓ Found {len(left_images)} left images")
        print(f"✓ Found {len(right_images)} right images")
        print(f"✓ Matched {len(matched_keys)} stereo pairs by filename")

        if unmatched_left or unmatched_right:
            print("WARNING: Ignoring unpaired calibration images:")
            for key in unmatched_left:
                print(f"  Left only:  {left_by_key[key]}")
            for key in unmatched_right:
                print(f"  Right only: {right_by_key[key]}")

        if not matched_keys:
            raise RuntimeError("No matching left/right calibration image pairs found")

        return [(left_by_key[key], right_by_key[key]) for key in matched_keys]

    def find_stereo_corners(self, image_pairs):
        """Find checkerboard corners only in valid, matching stereo pairs."""
        print("\nProcessing matched stereo image pairs...")

        objpoints = []
        imgpoints_left = []
        imgpoints_right = []
        img_size = None
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE

        for index, (left_path, right_path) in enumerate(image_pairs, start=1):
            left = cv2.imread(left_path)
            right = cv2.imread(right_path)
            if left is None or right is None:
                print(f"  {index}/{len(image_pairs)} ✗ (unreadable image)")
                continue
            if left.shape[:2] != right.shape[:2]:
                print(f"  {index}/{len(image_pairs)} ✗ (left/right size mismatch)")
                continue

            pair_size = (left.shape[1], left.shape[0])
            if img_size is None:
                img_size = pair_size
            elif pair_size != img_size:
                print(f"  {index}/{len(image_pairs)} ✗ (inconsistent image size {pair_size})")
                continue

            gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
            gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
            found_left, corners_left = cv2.findChessboardCorners(
                gray_left, self.pattern_size, flags
            )
            found_right, corners_right = cv2.findChessboardCorners(
                gray_right, self.pattern_size, flags
            )

            if not (found_left and found_right):
                missing = []
                if not found_left:
                    missing.append("left")
                if not found_right:
                    missing.append("right")
                print(f"  {index}/{len(image_pairs)} ✗ (pattern missing: {', '.join(missing)})")
                continue

            refined_left = cv2.cornerSubPix(
                gray_left, corners_left, (11, 11), (-1, -1), self.criteria
            )
            refined_right = cv2.cornerSubPix(
                gray_right, corners_right, (11, 11), (-1, -1), self.criteria
            )
            objpoints.append(self.objp.copy())
            imgpoints_left.append(refined_left)
            imgpoints_right.append(refined_right)
            print(f"  {index}/{len(image_pairs)} ✓")

        print(f"✓ Valid stereo pairs: {len(objpoints)}/{len(image_pairs)}")
        if img_size is None:
            raise RuntimeError("No readable, same-size stereo image pairs found")

        return objpoints, imgpoints_left, imgpoints_right, img_size
    
    def calibrate_camera(self, objpoints, imgpoints, img_size, camera_name):
        """Calibrate individual camera"""
        print(f"\nCalibrating {camera_name} camera...")
        
        rms_error, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, img_size, None, None
        )
        
        print(f"✓ {camera_name} calibration complete")
        print(f"  RMS reprojection error: {rms_error:.4f} pixels")
        
        return camera_matrix, dist_coeffs, rms_error
    
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
            dist_coeffs_left=dist_left,
            camera_matrix_right=camera_matrix_right,
            dist_right=dist_right,
            dist_coeffs_right=dist_right,
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
            img_size=img_size,
            pattern_size=self.pattern_size,
            square_size_mm=self.square_size,
            length_unit="mm",
            calibration_timestamp=datetime.now().isoformat(timespec="seconds")
        )
        
        print(f"✓ Calibration saved")
    
    def generate_report(self, error_left, error_right, stereo_error, T, baseline_mm,
                        valid_pairs, total_pairs, img_size):
        """Generate calibration quality report"""
        print(f"\nGenerating calibration report...")
        
        report = []
        report.append("=" * 70)
        report.append("IMX219 Stereo Camera Calibration Report")
        report.append("=" * 70)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Image size: {img_size[0]}x{img_size[1]}")
        report.append(f"Valid stereo pairs: {valid_pairs}/{total_pairs}")
        report.append(f"Checkerboard: {self.pattern_size[0]}x{self.pattern_size[1]} internal corners")
        report.append(f"Square size: {self.square_size:.2f} mm")
        report.append("")
        report.append("Calibration Quality:")
        report.append(f"  Left camera RMS reprojection error:  {error_left:.4f} pixels")
        report.append(f"  Right camera RMS reprojection error: {error_right:.4f} pixels")
        report.append(f"  Stereo RMS reprojection error:       {stereo_error:.4f} pixels")
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
            image_pairs = self.load_images()
            objpoints, imgpoints_left, imgpoints_right, img_size = self.find_stereo_corners(
                image_pairs
            )

            if len(objpoints) < 10:
                print(f"\nERROR: Not enough valid stereo pairs ({len(objpoints)})")
                print("Capture at least 10 valid calibration images!")
                sys.exit(1)
            
            # Calibrate left camera
            camera_matrix_left, dist_left, error_left = self.calibrate_camera(
                objpoints, imgpoints_left, img_size, "Left"
            )
            
            # Calibrate right camera
            camera_matrix_right, dist_right, error_right = self.calibrate_camera(
                objpoints, imgpoints_right, img_size, "Right"
            )
            
            # Stereo calibration
            R, T, E, F, R1, R2, P1, P2, Q, roi_left, roi_right, stereo_error = self.stereo_calibrate(
                objpoints,
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
            self.generate_report(
                error_left, error_right, stereo_error, T, baseline_mm,
                len(objpoints), len(image_pairs), img_size
            )
            
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
