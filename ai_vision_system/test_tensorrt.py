#!/usr/bin/env python3
"""
Test TensorRT Inference Engine with GPU
"""
import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root and system paths
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

def main():
    parser = argparse.ArgumentParser(description='Test TensorRT Inference Engine')
    parser.add_argument('--engine', type=str, default='yolov8n.engine',
                       help='Path to TensorRT engine file')
    parser.add_argument('--width', type=int, default=480,
                       help='Test image width')
    parser.add_argument('--height', type=int, default=480,
                       help='Test image height')
    parser.add_argument('--confidence', type=float, default=0.25,
                       help='Confidence threshold')

    args = parser.parse_args()

    try:
        from ai_engine.optimization.tensorrt_inference import TensorRTInference
        import cv2

        print("=" * 70)
        print("Testing TensorRT Inference Engine (GPU)")
        print("=" * 70)
        print()

        # Load TensorRT engine
        print(f"Loading TensorRT engine: {args.engine}")
        try:
            trt_model = TensorRTInference(args.engine, confidence_threshold=args.confidence)
            print("✓ TensorRT engine loaded successfully!")
            print()

            # Test with dummy image
            print(f"Testing inference with dummy image ({args.width}x{args.height})...")
            test_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            detections = trt_model.infer(test_img, (args.width, args.height))
            print(f"✓ Inference successful!")
            print(f"  Detections: {len(detections)}")
            print(f"  Average inference time: {trt_model.stats['avg_inference_time']:.2f} ms")
            print()

            print("=" * 70)
            print("✓ TensorRT Inference Engine is working with GPU!")
            print("=" * 70)

        except FileNotFoundError:
            print(f"✗ Error: {args.engine} not found!")
            print(f"  Run: python scripts/onnx_to_tensorrt.py --onnx {Path(args.engine).stem}.onnx")
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


if __name__ == "__main__":
    main()
