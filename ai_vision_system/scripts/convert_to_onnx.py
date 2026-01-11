#!/usr/bin/env python3
"""
Convert YOLO Model to ONNX
===========================
Converts YOLO models to ONNX format for GPU acceleration with ONNX Runtime.
This uses CUDA 12.6 directly without needing PyTorch CUDA wheels!
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_engine.optimization.onnx_inference import ONNXConverter


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to ONNX')
    parser.add_argument('--model', type=str, default='yolov11n.pt',
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for ONNX model (default: model_name.onnx)')
    parser.add_argument('--size', type=str, default='640,480',
                       help='Input size as width,height (default: 640,480)')
    parser.add_argument('--simplify', action='store_true', default=True,
                       help='Simplify ONNX model (default: True)')
    
    args = parser.parse_args()
    
    # Parse input size
    try:
        width, height = map(int, args.size.split(','))
        input_size = (width, height)
    except:
        print("Error: Invalid size format. Use width,height (e.g., 640,480)")
        return 1
    
    # Determine output path
    if args.output is None:
        model_path = Path(args.model)
        output_path = model_path.parent / f"{model_path.stem}.onnx"
    else:
        output_path = args.output
    
    print("=" * 70)
    print("YOLO to ONNX Converter")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Simplify: {args.simplify}")
    print("=" * 70)
    print()
    
    try:
        ONNXConverter.convert_yolo_to_onnx(
            model_path=args.model,
            output_path=str(output_path),
            input_size=input_size,
            simplify=args.simplify
        )
        print()
        print("=" * 70)
        print("✓ Conversion successful!")
        print(f"ONNX model: {output_path}")
        print()
        print("Next steps:")
        print("1. Install ONNX Runtime GPU: pip install onnxruntime-gpu")
        print("2. Update settings.yaml to use ONNX inference")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
