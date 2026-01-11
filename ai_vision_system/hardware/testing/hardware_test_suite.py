#!/usr/bin/env python3
"""
Hardware Testing Suite
======================
Comprehensive testing for dual IMX219 cameras including frame rates,
latency, image quality, and stress testing.
"""

import cv2
import time
import numpy as np
import sys
from datetime import datetime
from pathlib import Path


class HardwareTestSuite:
    """Comprehensive hardware testing"""
    
    def __init__(self):
        self.results = {}
        
    def test_frame_rate(self, sensor_id, width=1280, height=720, duration=5):
        """Test frame rate for a camera"""
        gst_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return None
        
        frame_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
        
        cap.release()
        elapsed = time.time() - start_time
        fps = frame_count / elapsed if elapsed > 0 else 0
        
        return {
            'fps': fps,
            'frames': frame_count,
            'duration': elapsed,
            'resolution': f"{width}x{height}"
        }
    
    def test_latency(self, sensor_id, width=1280, height=720, samples=30):
        """Measure latency between frame capture and processing"""
        gst_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return None
        
        latencies = []
        
        for _ in range(samples):
            t0 = time.time()
            ret, frame = cap.read()
            if ret:
                # Simulate processing
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                latency = (time.time() - t0) * 1000  # Convert to ms
                latencies.append(latency)
        
        cap.release()
        
        if latencies:
            return {
                'mean': np.mean(latencies),
                'std': np.std(latencies),
                'min': np.min(latencies),
                'max': np.max(latencies),
                'samples': len(latencies)
            }
        return None
    
    def test_image_quality(self, sensor_id, width=1280, height=720):
        """Assess image quality metrics"""
        gst_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return None
        
        # Capture a few frames and analyze
        frames = []
        for _ in range(5):
            ret, frame = cap.read()
            if ret:
                frames.append(frame)
        
        cap.release()
        
        if not frames:
            return None
        
        # Analyze quality
        frame = frames[-1]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics
        mean_brightness = np.mean(gray)
        std_brightness = np.std(gray)  # Contrast indicator
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()  # Sharpness
        
        return {
            'mean_brightness': float(mean_brightness),
            'contrast': float(std_brightness),
            'sharpness': float(laplacian_var),
            'resolution': f"{width}x{height}",
            'actual_size': frame.shape[:2]
        }
    
    def stress_test(self, sensor_id, width=1280, height=720, duration=60):
        """Stress test: continuous capture for extended period"""
        gst_str = (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={width}, height={height}, "
            f"format=NV12, framerate=30/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
        
        cap = cv2.VideoCapture(gst_str, cv2.CAP_GSTREAMER)
        if not cap.isOpened():
            return None
        
        frame_count = 0
        error_count = 0
        start_time = time.time()
        end_time = start_time + duration
        
        while time.time() < end_time:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
            else:
                error_count += 1
        
        cap.release()
        elapsed = time.time() - start_time
        
        return {
            'frames_captured': frame_count,
            'errors': error_count,
            'duration': elapsed,
            'success_rate': (frame_count / (frame_count + error_count) * 100) if (frame_count + error_count) > 0 else 0
        }
    
    def run_all_tests(self):
        """Run complete test suite"""
        print("Running Hardware Test Suite...")
        print("=" * 70)
        
        results = {}
        
        for sensor_id in [0, 1]:
            print(f"\nTesting Camera {sensor_id}...")
            camera_results = {}
            
            # Frame rate test
            print(f"  Testing frame rate (1280x720)...")
            fps_result = self.test_frame_rate(sensor_id, 1280, 720, 5)
            camera_results['frame_rate'] = fps_result
            if fps_result:
                print(f"    FPS: {fps_result['fps']:.2f}")
            
            # Latency test
            print(f"  Testing latency...")
            latency_result = self.test_latency(sensor_id, 1280, 720, 30)
            camera_results['latency'] = latency_result
            if latency_result:
                print(f"    Mean latency: {latency_result['mean']:.2f} ms")
            
            # Image quality test
            print(f"  Testing image quality...")
            quality_result = self.test_image_quality(sensor_id, 1280, 720)
            camera_results['image_quality'] = quality_result
            if quality_result:
                print(f"    Sharpness: {quality_result['sharpness']:.2f}")
            
            results[f'camera_{sensor_id}'] = camera_results
        
        # Stress test (only on camera 0 to save time)
        print(f"\nRunning stress test on Camera 0 (60 seconds)...")
        stress_result = self.stress_test(0, 1280, 720, 60)
        results['stress_test'] = stress_result
        if stress_result:
            print(f"  Frames captured: {stress_result['frames_captured']}")
            print(f"  Success rate: {stress_result['success_rate']:.2f}%")
        
        self.results = results
        return results
    
    def save_report(self, filename=None):
        """Save test results to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"hardware_test_report_{timestamp}.txt"
        
        report = []
        report.append("=" * 70)
        report.append("Hardware Test Suite Report")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        report.append("")
        
        for camera_key, camera_results in self.results.items():
            if camera_key == 'stress_test':
                continue
            report.append(f"{camera_key.upper()}:")
            report.append("-" * 70)
            
            if 'frame_rate' in camera_results and camera_results['frame_rate']:
                fr = camera_results['frame_rate']
                report.append(f"  Frame Rate: {fr['fps']:.2f} FPS ({fr['resolution']})")
            
            if 'latency' in camera_results and camera_results['latency']:
                lat = camera_results['latency']
                report.append(f"  Latency: {lat['mean']:.2f} ms (std: {lat['std']:.2f} ms)")
            
            if 'image_quality' in camera_results and camera_results['image_quality']:
                iq = camera_results['image_quality']
                report.append(f"  Image Quality:")
                report.append(f"    Brightness: {iq['mean_brightness']:.2f}")
                report.append(f"    Contrast: {iq['contrast']:.2f}")
                report.append(f"    Sharpness: {iq['sharpness']:.2f}")
            
            report.append("")
        
        if 'stress_test' in self.results:
            st = self.results['stress_test']
            report.append("STRESS TEST:")
            report.append("-" * 70)
            report.append(f"  Duration: {st['duration']:.2f} seconds")
            report.append(f"  Frames captured: {st['frames_captured']}")
            report.append(f"  Errors: {st['errors']}")
            report.append(f"  Success rate: {st['success_rate']:.2f}%")
        
        report.append("")
        report.append("=" * 70)
        
        report_text = "\n".join(report)
        
        # Save to file
        output_path = Path("/home/jetson/Downloads/IMX219_Camera_Project/ai_vision_system/hardware/testing") / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(report_text)
        
        return str(output_path)


def main():
    """Main function"""
    suite = HardwareTestSuite()
    results = suite.run_all_tests()
    
    print("\n" + "=" * 70)
    print("Test suite completed!")
    
    report_path = suite.save_report()
    print(f"Report saved to: {report_path}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
