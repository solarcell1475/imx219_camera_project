#!/usr/bin/env python3
"""
Closed-Loop Test for YOLO AI Vision System
===========================================
Tests inference performance with dummy images for 10 seconds.
Reports FPS and object detection statistics.
"""

import sys
import time
import numpy as np
import cv2
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Apply PyTorch patch
try:
    from utils.pytorch_patch import patch_torch_load
    patch_torch_load()
except:
    pass

from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
from ai_engine.inference.dual_yolo_inference import DualYOLOInference
from monitoring.detection_stats import DetectionStatistics


def create_test_image(width=640, height=640):
    """Create a test image with some patterns"""
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some geometric shapes to potentially trigger detections
    cv2.rectangle(img, (100, 100), (300, 300), (255, 0, 0), -1)  # Blue rectangle
    cv2.circle(img, (500, 200), 80, (0, 255, 0), -1)  # Green circle
    cv2.ellipse(img, (200, 500), (100, 50), 45, 0, 360, (0, 0, 255), -1)  # Red ellipse
    
    # Add some text
    cv2.putText(img, "TEST", (400, 400), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
    
    return img


def run_closed_loop_test(duration_seconds=10):
    """Run closed-loop inference test"""
    print("=" * 70)
    print("Closed-Loop YOLO Inference Test")
    print("=" * 70)
    print(f"Test Duration: {duration_seconds} seconds")
    print("=" * 70)
    print()
    
    try:
        # Initialize model manager
        print("1. Loading YOLO model...")
        model_manager = YOLOModelManager()
        
        # Try YOLO11n first, fallback to YOLOv8n if not available
        try:
            yolo_model = model_manager.load_model(version='v11', size='n', device='cpu')
            model_name = "YOLO11n"
        except Exception as e:
            print(f"   ⚠ YOLO11n not available, trying YOLOv8n...")
            yolo_model = model_manager.load_model(version='v8', size='n', device='cpu')
            model_name = "YOLOv8n"
        
        print(f"   ✓ Model loaded: {model_name}")
        print()
        
        # Initialize inference
        print("2. Initializing inference engine...")
        inference = DualYOLOInference(
            yolo_model,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device='cpu'
        )
        inference_size = (640, 640)  # Square input for YOLO11
        print(f"   ✓ Inference engine ready (resolution: {inference_size[0]}x{inference_size[1]})")
        print()
        
        # Initialize detection statistics
        print("3. Initializing detection statistics...")
        detection_stats = DetectionStatistics(history_size=1000)
        print("   ✓ Statistics tracker ready")
        print()
        
        # Create test image
        print("4. Generating test images...")
        test_image = create_test_image(640, 640)
        print("   ✓ Test image generated (640x640)")
        print()
        
        # Run inference loop
        print("5. Starting inference loop...")
        print(f"   Running for {duration_seconds} seconds...")
        print()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        frame_count = 0
        total_inference_time = 0.0
        detections_list = []
        
        print("Progress: [", end="", flush=True)
        progress_interval = duration_seconds / 20
        last_progress = 0
        
        while time.time() < end_time:
            # Run inference
            inference_start = time.time()
            detections = inference.infer_single(test_image, inference_size=inference_size)
            inference_time = (time.time() - inference_start) * 1000  # ms
            
            # Update statistics
            detection_stats.update(detections)
            
            # Track metrics
            frame_count += 1
            total_inference_time += inference_time
            detections_list.append(len(detections))
            
            # Progress indicator
            elapsed = time.time() - start_time
            progress_marks = int(elapsed / progress_interval)
            if progress_marks > last_progress:
                print("." * (progress_marks - last_progress), end="", flush=True)
                last_progress = progress_marks
            
            # Small delay to avoid 100% CPU
            time.sleep(0.01)
        
        print("]")
        print()
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        avg_inference_time = total_inference_time / frame_count
        
        # Print results
        print("=" * 70)
        print("Test Results")
        print("=" * 70)
        print()
        
        print(f"Model: {model_name}")
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f} FPS")
        print(f"Average Inference Time: {avg_inference_time:.2f} ms")
        print(f"Total Detections: {sum(detections_list)}")
        print(f"Average Detections per Frame: {sum(detections_list) / frame_count:.2f}")
        print()
        
        # Detection statistics
        print("=" * 70)
        print("Object Detection Statistics")
        print("=" * 70)
        print()
        
        stats_data = detection_stats.get_stats_sorted_by_total()
        
        if stats_data:
            print(f"{'Class':<20} {'Total':<10} {'Rate':<12} {'Avg Conf':<12}")
            print("-" * 70)
            
            for stat in stats_data[:10]:  # Top 10
                print(f"{stat.class_name:<20} {stat.total_detections:<10} "
                      f"{stat.appearance_rate:>10.1%} {stat.avg_confidence:>11.2f}")
            
            if len(stats_data) > 10:
                print(f"\n... and {len(stats_data) - 10} more classes")
        else:
            print("No detections found during test period")
            print("(This is normal if test images don't contain recognizable objects)")
        
        print()
        print("=" * 70)
        print("Performance Summary")
        print("=" * 70)
        print()
        
        if avg_fps >= 30:
            performance_level = "Excellent"
        elif avg_fps >= 20:
            performance_level = "Good"
        elif avg_fps >= 10:
            performance_level = "Fair"
        else:
            performance_level = "Slow (consider GPU acceleration)"
        
        print(f"Performance Level: {performance_level}")
        print(f"FPS: {avg_fps:.2f}")
        print()
        
        if avg_inference_time < 50:
            speed_level = "Fast"
        elif avg_inference_time < 100:
            speed_level = "Moderate"
        else:
            speed_level = "Slow"
        
        print(f"Inference Speed: {speed_level}")
        print(f"Latency: {avg_inference_time:.2f} ms per frame")
        print()
        
        print("=" * 70)
        print("✓ Test Complete")
        print("=" * 70)
        
        return {
            'model': model_name,
            'fps': avg_fps,
            'inference_time_ms': avg_inference_time,
            'total_frames': frame_count,
            'total_detections': sum(detections_list),
            'classes_detected': len(stats_data),
            'performance_level': performance_level
        }
        
    except Exception as e:
        print()
        print("=" * 70)
        print("✗ Test Failed")
        print("=" * 70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    print()
    result = run_closed_loop_test(duration_seconds=10)
    print()
    
    if result:
        print(f"Test Summary:")
        print(f"  Model: {result['model']}")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Latency: {result['inference_time_ms']:.2f} ms")
        print(f"  Classes Detected: {result['classes_detected']}")
        print(f"  Performance: {result['performance_level']}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
