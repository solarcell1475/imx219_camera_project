#!/usr/bin/env python3
"""
Test TensorRT Inference Engine with GPU
"""
import sys
from pathlib import Path
import numpy as np

# Add project root and system paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

try:
    from ai_engine.optimization.tensorrt_inference import TensorRTInference
    import cv2
    
    print("=" * 70)
    print("Testing TensorRT Inference Engine (GPU)")
    print("=" * 70)
    print()
    
    # Load TensorRT engine
    print("Loading TensorRT engine: yolov8n.engine")
    try:
        trt_model = TensorRTInference('yolov8n.engine', confidence_threshold=0.25)
        print("✓ TensorRT engine loaded successfully!")
        print()
        
        # Test with dummy image
        print("Testing inference with dummy image...")
        test_img = np.zeros((480, 480, 3), dtype=np.uint8)
        detections = trt_model.infer(test_img, (480, 480))
        print(f"✓ Inference successful!")
        print(f"  Detections: {len(detections)}")
        print(f"  Average inference time: {trt_model.stats['avg_inference_time']:.2f} ms")
        print()
        
        print("=" * 70)
        print("✓ TensorRT Inference Engine is working with GPU!")
        print("=" * 70)
        
    except FileNotFoundError:
        print("✗ Error: yolov8n.engine not found!")
        print("  Run: python scripts/onnx_to_tensorrt.py --onnx yolov8n.onnx")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure TensorRT and PyCUDA are installed")
    sys.exit(1)
