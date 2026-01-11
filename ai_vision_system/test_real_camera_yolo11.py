#!/usr/bin/env python3
"""
Real Camera Test with YOLO11n
==============================
Tests YOLO11n with real camera feeds for 10 seconds.
Reports FPS and object detection statistics.
"""

import sys
import time
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

# Setup display environment
try:
    from utils.display_fix import setup_display_environment
    setup_display_environment()
except:
    pass

from video_processing.capture.nvargus_capture import NVArgusCapture
from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
from ai_engine.inference.dual_yolo_inference import ParallelDualInference
from monitoring.detection_stats import DetectionStatistics


def test_real_camera_yolo11(duration_seconds=10):
    """Test YOLO11n with real cameras"""
    print("=" * 70)
    print("Real Camera Test with YOLO11n - Jetson Orin Nano Super")
    print("=" * 70)
    print(f"Test Duration: {duration_seconds} seconds")
    print("=" * 70)
    print()
    
    camera_capture = None
    
    try:
        # Step 1: Initialize cameras
        print("1. Initializing cameras...")
        try:
            camera_capture = NVArgusCapture(width=1280, height=720, fps=30)
            camera_capture.start()
            print("   ✓ Cameras initialized")
            print(f"   Resolution: 1280x720 @ 30fps")
        except Exception as e:
            print(f"   ✗ Camera initialization failed: {e}")
            print("   Trying GStreamer capture...")
            try:
                from video_processing.capture.gstreamer_capture import GStreamerDualCapture
                camera_capture = GStreamerDualCapture(width=1280, height=720, fps=30)
                camera_capture.start()
                print("   ✓ GStreamer cameras initialized")
            except Exception as e2:
                print(f"   ✗ GStreamer also failed: {e2}")
                print("   ⚠ Cannot test with real cameras")
                return None
        print()
        
        # Step 2: Load YOLO11n model
        print("2. Loading YOLO11n model...")
        model_manager = YOLOModelManager()
        try:
            yolo_model = model_manager.load_model(version='v11', size='n', device='cpu')
            model_name = "YOLO11n"
            print(f"   ✓ YOLO11n model loaded")
        except Exception as e:
            print(f"   ⚠ YOLO11n failed, trying YOLOv8n: {e}")
            yolo_model = model_manager.load_model(version='v8', size='n', device='cpu')
            model_name = "YOLOv8n"
            print(f"   ✓ YOLOv8n loaded (fallback)")
        print()
        
        # Step 3: Initialize inference
        print("3. Initializing inference engine...")
        inference = ParallelDualInference(
            yolo_model,
            confidence_threshold=0.25,
            iou_threshold=0.45,
            device='cpu',
            inference_size=(640, 640)
        )
        print("   ✓ Inference engine ready (640x640)")
        print()
        
        # Step 4: Initialize statistics
        print("4. Initializing detection statistics...")
        detection_stats = DetectionStatistics(history_size=1000)
        print("   ✓ Statistics tracker ready")
        print()
        
        # Step 5: Run inference loop
        print("5. Starting real camera inference loop...")
        print(f"   Running for {duration_seconds} seconds...")
        print("   (Press Ctrl+C to stop early)")
        print()
        
        start_time = time.time()
        end_time = start_time + duration_seconds
        frame_count = 0
        total_inference_time = 0.0
        detections_list_0 = []
        detections_list_1 = []
        successful_frames = 0
        failed_frames = 0
        
        print("Progress: [", end="", flush=True)
        progress_interval = duration_seconds / 20
        last_progress = 0
        
        try:
            while time.time() < end_time:
                # Capture frames
                try:
                    ret, frame0, frame1 = camera_capture.read(timeout=0.5)
                    
                    if not ret or frame0 is None or frame1 is None or frame0.size == 0 or frame1.size == 0:
                        failed_frames += 1
                        time.sleep(0.01)
                        continue
                    
                    # Run inference
                    inference_start = time.time()
                    detections0, detections1 = inference.infer_dual(frame0, frame1, inference_size=(640, 640))
                    inference_time = (time.time() - inference_start) * 1000  # ms
                    
                    # Update statistics
                    detection_stats.update(detections0)
                    detection_stats.update(detections1)
                    
                    # Track metrics
                    frame_count += 1
                    successful_frames += 1
                    total_inference_time += inference_time
                    detections_list_0.append(len(detections0))
                    detections_list_1.append(len(detections1))
                    
                    # Progress indicator
                    elapsed = time.time() - start_time
                    progress_marks = int(elapsed / progress_interval)
                    if progress_marks > last_progress:
                        print("." * (progress_marks - last_progress), end="", flush=True)
                        last_progress = progress_marks
                    
                    # Small delay
                    time.sleep(0.01)
                    
                except KeyboardInterrupt:
                    print("\n   Interrupted by user")
                    break
                except Exception as e:
                    failed_frames += 1
                    print(f"\n   ⚠ Frame processing error: {e}")
                    time.sleep(0.1)
                    continue
            
            print("]")
            print()
            
        except KeyboardInterrupt:
            print("\n   Test interrupted by user")
            print()
        
        # Calculate statistics
        total_time = time.time() - start_time
        if successful_frames > 0:
            avg_fps = successful_frames / total_time
            avg_inference_time = total_inference_time / successful_frames
        else:
            avg_fps = 0.0
            avg_inference_time = 0.0
        
        total_detections = sum(detections_list_0) + sum(detections_list_1)
        
        # Print results
        print("=" * 70)
        print("Real Camera Test Results - YOLO11n")
        print("=" * 70)
        print()
        
        print(f"Model: {model_name}")
        print(f"Test Duration: {total_time:.2f} seconds")
        print(f"Successful Frames: {successful_frames}")
        print(f"Failed Frames: {failed_frames}")
        print()
        
        if successful_frames > 0:
            print(f"Average FPS: {avg_fps:.2f} FPS")
            print(f"Average Inference Time: {avg_inference_time:.2f} ms")
            print()
            
            print(f"Total Detections: {total_detections}")
            print(f"  Camera 0: {sum(detections_list_0)} detections")
            print(f"  Camera 1: {sum(detections_list_1)} detections")
            print(f"Average Detections per Frame: {total_detections / successful_frames:.2f}")
            print(f"Frames with Detections: {sum(1 for d in (detections_list_0 + detections_list_1) if d > 0)}/{successful_frames}")
            print()
        else:
            print("⚠ No successful frames captured!")
            print()
        
        # Detection statistics
        stats_data = detection_stats.get_stats_sorted_by_total()
        if stats_data:
            print("=" * 70)
            print("Object Detection Statistics")
            print("=" * 70)
            print()
            print(f"{'Class':<20} {'Total':<10} {'Appearance Rate':<15} {'Avg Confidence':<15}")
            print("-" * 70)
            for stat in stats_data[:15]:  # Top 15
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
        elif avg_fps > 0:
            perf_level = "Slow (GPU recommended)"
        else:
            perf_level = "Failed"
        
        print(f"Performance Level: {perf_level}")
        if successful_frames > 0:
            print(f"FPS: {avg_fps:.2f}")
            print(f"Latency: {avg_inference_time:.2f} ms per frame")
        print()
        
        # Upgrade verification
        print("=" * 70)
        print("YOLO11n Upgrade Verification")
        print("=" * 70)
        print()
        
        if model_name == "YOLO11n":
            print("✓ YOLO11n model is working with real cameras!")
            print("✓ Upgrade to YOLO11n: SUCCESSFUL")
            print()
            if avg_fps < 5:
                print("Note: Current performance is CPU-only.")
                print("For better performance (40-60 FPS), use TensorRT GPU acceleration:")
                print("  1. Convert to ONNX: python scripts/convert_to_onnx.py --model yolo11n.pt")
                print("  2. Convert to TensorRT: python scripts/onnx_to_tensorrt.py --onnx yolo11n.onnx")
        else:
            print("⚠ Using YOLOv8n (YOLO11n not available)")
            print("  Upgrade verification: PARTIAL")
        
        print()
        print("=" * 70)
        print("✓ Test Complete")
        print("=" * 70)
        
        return {
            'model': model_name,
            'fps': avg_fps,
            'inference_time_ms': avg_inference_time,
            'successful_frames': successful_frames,
            'total_detections': total_detections,
            'classes_detected': len(stats_data) if stats_data else 0,
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
        
    finally:
        # Cleanup
        if camera_capture:
            try:
                camera_capture.stop()
                print("\n✓ Cameras stopped")
            except:
                pass


def main():
    """Main function"""
    print()
    result = test_real_camera_yolo11(duration_seconds=10)
    print()
    
    if result:
        print("Final Summary:")
        print(f"  Model: {result['model']}")
        print(f"  FPS: {result['fps']:.2f}")
        print(f"  Latency: {result['inference_time_ms']:.2f} ms")
        print(f"  Successful Frames: {result['successful_frames']}")
        print(f"  Total Detections: {result['total_detections']}")
        print(f"  Classes Detected: {result['classes_detected']}")
        print(f"  Performance: {result['performance_level']}")
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
