#!/usr/bin/env python3
"""
Test ONNX Inference Engine
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

try:
    from ai_engine.optimization.onnx_inference import ONNXInference
    import numpy as np
    
    print("=" * 70)
    print("Testing ONNX Inference Engine")
    print("=" * 70)
    print()
    
    # Load ONNX model
    print("Loading ONNX model: yolov8n.onnx")
    try:
        onnx_model = ONNXInference('yolov8n.onnx', confidence_threshold=0.25)
        print("✓ ONNX model loaded successfully!")
        print()
        
        # Check providers
        providers = onnx_model.session.get_providers()
        print(f"Available providers: {providers}")
        print(f"Using provider: {providers[0]}")
        print()
        
        # Test with dummy image
        print("Testing inference with dummy image...")
        test_img = np.zeros((480, 640, 3), dtype=np.uint8)
        detections = onnx_model.infer(test_img, (640, 480))
        print(f"✓ Inference successful!")
        print(f"  Detections: {len(detections)}")
        print(f"  Average inference time: {onnx_model.stats['avg_inference_time']:.2f} ms")
        print()
        
        print("=" * 70)
        print("✓ ONNX Inference Engine is working!")
        print("=" * 70)
        
    except FileNotFoundError:
        print("✗ Error: yolov8n.onnx not found!")
        print("  Run: python scripts/convert_to_onnx.py --model yolov8n.pt")
        sys.exit(1)
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
except ImportError as e:
    print(f"✗ Import error: {e}")
    print("  Make sure you're in the ai_vision_system directory")
    sys.exit(1)
