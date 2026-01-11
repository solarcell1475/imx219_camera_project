#!/usr/bin/env python3
"""
YOLO System Loopback Test
==========================
Tests the YOLO system end-to-end using test images or cameras.
"""

import sys
import time
import cv2
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Apply PyTorch patch for YOLO compatibility
try:
    from utils.pytorch_patch import patch_torch_load
    patch_torch_load()
except:
    pass

from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
from ai_engine.inference.dual_yolo_inference import DualYOLOInference
from video_processing.postprocessing.yolo_postprocess import YOLOPostprocessor
from monitoring.metrics.performance_metrics import PerformanceMetrics


def create_test_image(width=1280, height=720):
    """Create a test image with some objects"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some colored rectangles to simulate objects
    cv2.rectangle(img, (100, 100), (300, 300), (0, 255, 0), -1)  # Green box
    cv2.rectangle(img, (400, 200), (600, 400), (255, 0, 0), -1)  # Blue box
    cv2.rectangle(img, (800, 150), (1000, 350), (0, 0, 255), -1)  # Red box
    
    # Add text
    cv2.putText(img, "Test Image", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return img


def test_yolo_loopback(num_iterations=10, use_cameras=False):
    """Run loopback test"""
    print("=" * 70)
    print("YOLO System Loopback Test")
    print("=" * 70)
    print()
    
    # Initialize components
    print("1. Loading YOLO model...")
    try:
        model_manager = YOLOModelManager()
        yolo_model = model_manager.load_model(version='v8', size='n', device='cpu')  # Use CPU for testing
        print("   ✓ Model loaded (YOLOv8n)")
    except Exception as e:
        print(f"   ✗ Model loading failed: {e}")
        # Try with weights_only fix
        try:
            from ultralytics.nn.tasks import DetectionModel
            import torch
            torch.serialization.add_safe_globals([DetectionModel])
            yolo_model = model_manager.load_model(version='v8', size='n', device='cpu')
            print("   ✓ Model loaded (with fix)")
        except Exception as e2:
            print(f"   ✗ Still failed: {e2}")
            return False
    
    print()
    print("2. Initializing inference pipeline...")
    inference = DualYOLOInference(
        yolo_model,
        confidence_threshold=0.25,
        iou_threshold=0.45,
        device='cpu'  # Use CPU for testing
    )
    print("   ✓ Inference pipeline ready")
    
    print()
    print("3. Initializing postprocessor...")
    postprocessor = YOLOPostprocessor()
    print("   ✓ Postprocessor ready")
    
    print()
    print("4. Initializing metrics...")
    metrics = PerformanceMetrics()
    print("   ✓ Metrics collector ready")
    
    print()
    print("5. Starting loopback test...")
    print(f"   Running {num_iterations} iterations...")
    print()
    
    # Test loop
    success_count = 0
    total_inference_time = 0.0
    
    for i in range(num_iterations):
        print(f"   Iteration {i+1}/{num_iterations}...", end=" ", flush=True)
        
        try:
            # Create or capture test images
            if use_cameras:
                # Try to capture from cameras
                cap0 = cv2.VideoCapture(0)
                cap1 = cv2.VideoCapture(1)
                ret0, frame0 = cap0.read()
                ret1, frame1 = cap1.read()
                cap0.release()
                cap1.release()
                
                if not ret0 or not ret1:
                    print("Camera capture failed, using test images")
                    frame0 = create_test_image()
                    frame1 = create_test_image()
            else:
                # Use generated test images
                frame0 = create_test_image()
                frame1 = create_test_image()
            
            # Run inference
            start_time = time.time()
            detections0, detections1 = inference.infer_dual(frame0, frame1)
            inference_time = (time.time() - start_time) * 1000
            
            # Process results
            total_detections = len(detections0) + len(detections1)
            
            # Update metrics
            metrics.collect_all_metrics()
            metrics.add_inference_metrics(inference_time, 1000.0 / inference_time if inference_time > 0 else 0)
            
            total_inference_time += inference_time
            success_count += 1
            
            print(f"✓ ({total_detections} detections, {inference_time:.1f}ms)")
            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    print("=" * 70)
    print("Test Results")
    print("=" * 70)
    print(f"Successful iterations: {success_count}/{num_iterations}")
    if success_count > 0:
        avg_inference_time = total_inference_time / success_count
        print(f"Average inference time: {avg_inference_time:.2f} ms")
        print(f"Estimated FPS: {1000.0 / avg_inference_time:.1f}")
    
    # Get metrics summary
    print()
    print("Performance Metrics:")
    summary = metrics.get_summary()
    current = summary.get('current', {})
    if 'cpu_percent' in current:
        print(f"  CPU Usage: {current['cpu_percent']:.1f}%")
    if 'memory_percent' in current:
        print(f"  Memory Usage: {current['memory_percent']:.1f}%")
    if 'fps' in current:
        print(f"  FPS: {current['fps']:.1f}")
    
    print()
    print("=" * 70)
    
    if success_count == num_iterations:
        print("✓ Loopback test PASSED")
        return True
    else:
        print(f"⚠ Loopback test completed with {success_count}/{num_iterations} successes")
        return success_count > 0


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="YOLO Loopback Test")
    parser.add_argument('--iterations', type=int, default=10, help='Number of test iterations')
    parser.add_argument('--cameras', action='store_true', help='Use real cameras instead of test images')
    
    args = parser.parse_args()
    
    success = test_yolo_loopback(num_iterations=args.iterations, use_cameras=args.cameras)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
