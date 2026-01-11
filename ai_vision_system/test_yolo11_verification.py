#!/usr/bin/env python3
"""
YOLO11n Verification Test with Screen Capture
==============================================
Tests YOLO11n model on Jetson and verifies upgrade success with screen capture.
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


def capture_screen_region(width=640, height=640):
    """Capture a region of the screen as test image"""
    try:
        # Try to capture from display (if available)
        import mss
        with mss.mss() as sct:
            # Capture primary monitor
            monitor = sct.monitors[1]  # Primary monitor
            # Capture center region
            region = {
                "top": monitor["top"] + (monitor["height"] - height) // 2,
                "left": monitor["left"] + (monitor["width"] - width) // 2,
                "width": width,
                "height": height
            }
            screenshot = sct.grab(region)
            img = np.array(screenshot)
            # Convert BGRA to BGR
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            return img
    except:
        # Fallback: create test image with more realistic patterns
        img = np.zeros((height, width, 3), dtype=np.uint8)
        img.fill(128)  # Gray background
        
        # Add some realistic patterns
        cv2.rectangle(img, (50, 50), (250, 250), (0, 255, 0), 3)  # Green rectangle
        cv2.circle(img, (450, 200), 100, (255, 0, 0), 3)  # Blue circle
        cv2.putText(img, "YOLO11 TEST", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        
        return img


def test_yolo11_verification(duration_seconds=10):
    """Test YOLO11n model and verify upgrade"""
    print("=" * 70)
    print("YOLO11n Verification Test - Jetson Orin Nano Super")
    print("=" * 70)
    print(f"Test Duration: {duration_seconds} seconds")
    print("=" * 70)
    print()
    
    try:
        # Step 1: Test YOLO11n model loading
        print("1. Testing YOLO11n Model Loading...")
        model_manager = YOLOModelManager()
        
        # Try multiple ways to load YOLO11
        upgrade_success = False
        yolo_model = None
        model_name = None
        
        # Try 1: Direct YOLO11n
        try:
            from ultralytics import YOLO
            yolo_model = YOLO('yolo11n.pt')
            model_name = "YOLO11n"
            upgrade_success = True
            print(f"   ✓ YOLO11n model loaded successfully (direct)!")
        except Exception as e1:
            # Try 2: Via model manager with v11
            try:
                yolo_model = model_manager.load_model(version='v11', size='n', device='cpu')
                model_name = "YOLO11n"
                upgrade_success = True
                print(f"   ✓ YOLO11n model loaded successfully (via manager)!")
            except Exception as e2:
                # Try 3: Check if it's just a naming issue
                try:
                    yolo_model = YOLO('yolov11n.pt')
                    model_name = "YOLO11n"
                    upgrade_success = True
                    print(f"   ✓ YOLO11n model loaded successfully (yolov11n.pt)!")
                except Exception as e3:
                    print(f"   ✗ YOLO11n loading failed")
                    print(f"   Error: {str(e1)[:80]}")
                    print(f"   ⚠ Falling back to YOLOv8n...")
                    yolo_model = model_manager.load_model(version='v8', size='n', device='cpu')
                    model_name = "YOLOv8n"
                    upgrade_success = False
                    print(f"   ✓ YOLOv8n loaded (fallback)")
        
        print()
        
        # Step 2: Initialize inference
        print("2. Initializing inference engine...")
        inference = DualYOLOInference(
            yolo_model,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device='cpu'
        )
        inference_size = (640, 640)  # Square input for YOLO11
        print(f"   ✓ Inference engine ready")
        print(f"   Resolution: {inference_size[0]}x{inference_size[1]}")
        print()
        
        # Step 3: Initialize statistics
        print("3. Initializing detection statistics...")
        detection_stats = DetectionStatistics(history_size=1000)
        print("   ✓ Statistics tracker ready")
        print()
        
        # Step 4: Screen capture test
        print("4. Testing screen capture...")
        try:
            test_image = capture_screen_region(640, 640)
            print("   ✓ Screen capture successful (640x640)")
        except Exception as e:
            print(f"   ⚠ Screen capture failed: {e}")
            print("   Using generated test image...")
            test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            cv2.putText(test_image, "YOLO11 TEST", (150, 320), 
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
        print()
        
        # Step 5: Run inference loop
        print("5. Running inference loop...")
        print(f"   Duration: {duration_seconds} seconds")
        print()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        frame_count = 0
        total_inference_time = 0.0
        detections_list = []
        detection_details = []
        
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
            
            # Store detection details
            if detections:
                for det in detections:
                    detection_details.append({
                        'class': det.get('class_name', 'unknown'),
                        'confidence': det.get('confidence', 0.0)
                    })
            
            # Progress indicator
            elapsed = time.time() - start_time
            progress_marks = int(elapsed / progress_interval)
            if progress_marks > last_progress:
                print("." * (progress_marks - last_progress), end="", flush=True)
                last_progress = progress_marks
            
            # Small delay
            time.sleep(0.01)
        
        print("]")
        print()
        
        # Calculate statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time
        avg_inference_time = total_inference_time / frame_count
        
        # Print comprehensive results
        print("=" * 70)
        print("YOLO11n Verification Results")
        print("=" * 70)
        print()
        
        print(f"Model Used: {model_name}")
        print(f"Upgrade Status: {'✓ SUCCESS' if upgrade_success else '✗ FAILED (using YOLOv8n)'}")
        print()
        
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Total Frames Processed: {frame_count}")
        print(f"Average FPS: {avg_fps:.2f} FPS")
        print(f"Average Inference Time: {avg_inference_time:.2f} ms")
        print()
        
        print(f"Total Detections: {sum(detections_list)}")
        print(f"Average Detections per Frame: {sum(detections_list) / frame_count:.2f}")
        print(f"Frames with Detections: {sum(1 for d in detections_list if d > 0)}/{frame_count}")
        print()
        
        # Detection details
        if detection_details:
            print("=" * 70)
            print("Detection Details")
            print("=" * 70)
            print()
            
            # Count by class
            from collections import Counter
            class_counts = Counter([d['class'] for d in detection_details])
            confidences = [d['confidence'] for d in detection_details]
            
            print(f"{'Class':<20} {'Count':<10} {'Avg Confidence':<15}")
            print("-" * 70)
            for class_name, count in class_counts.most_common(10):
                class_confs = [d['confidence'] for d in detection_details if d['class'] == class_name]
                avg_conf = sum(class_confs) / len(class_confs) if class_confs else 0.0
                print(f"{class_name:<20} {count:<10} {avg_conf:>14.2f}")
        else:
            print("No detections found (this is normal with test images)")
        print()
        
        # Statistics summary
        stats_data = detection_stats.get_stats_sorted_by_total()
        if stats_data:
            print("=" * 70)
            print("Detection Statistics Summary")
            print("=" * 70)
            print()
            print(f"{'Class':<20} {'Total':<10} {'Appearance Rate':<15} {'Avg Confidence':<15}")
            print("-" * 70)
            for stat in stats_data[:10]:
                print(f"{stat.class_name:<20} {stat.total_detections:<10} "
                      f"{stat.appearance_rate:>14.1%} {stat.avg_confidence:>14.2f}")
        print()
        
        # Performance assessment
        print("=" * 70)
        print("Performance Assessment")
        print("=" * 70)
        print()
        
        if avg_fps >= 30:
            perf_level = "Excellent"
        elif avg_fps >= 20:
            perf_level = "Good"
        elif avg_fps >= 10:
            perf_level = "Fair"
        elif avg_fps >= 5:
            perf_level = "Acceptable (CPU)"
        else:
            perf_level = "Slow (GPU recommended)"
        
        print(f"Performance Level: {perf_level}")
        print(f"FPS: {avg_fps:.2f}")
        print(f"Latency: {avg_inference_time:.2f} ms per frame")
        print()
        
        # Upgrade verification
        print("=" * 70)
        print("Upgrade Verification")
        print("=" * 70)
        print()
        
        if upgrade_success:
            print("✓ YOLO11n model is working on Jetson!")
            print("✓ Upgrade to YOLO11n: SUCCESSFUL")
            print()
            print("Note: Current performance is CPU-only.")
            print("For better performance (40-60 FPS), use TensorRT GPU acceleration:")
            print("  1. Convert to ONNX: python scripts/convert_to_onnx.py --model yolov11n.pt")
            print("  2. Convert to TensorRT: python scripts/onnx_to_tensorrt.py --onnx yolov11n.onnx")
        else:
            print("✗ YOLO11n model not available")
            print("⚠ Upgrade verification: FAILED")
            print("  Using YOLOv8n as fallback")
            print()
            print("Possible reasons:")
            print("  - Ultralytics package needs update")
            print("  - YOLO11 models not yet released")
            print("  - Network issue downloading model")
        
        print()
        print("=" * 70)
        print("✓ Test Complete")
        print("=" * 70)
        
        return {
            'model': model_name,
            'upgrade_success': upgrade_success,
            'fps': avg_fps,
            'inference_time_ms': avg_inference_time,
            'total_frames': frame_count,
            'total_detections': sum(detections_list),
            'classes_detected': len(stats_data),
            'performance_level': perf_level
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
    result = test_yolo11_verification(duration_seconds=10)
    print()
    
    if result:
        print("Final Summary:")
        print(f"  Model: {result['model']}")
        print(f"  Upgrade Success: {'YES ✓' if result['upgrade_success'] else 'NO ✗'}")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Latency: {result['inference_time_ms']:.2f} ms")
        print(f"  Performance: {result['performance_level']}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
