#!/usr/bin/env python3
"""
Convert YOLO Model to TensorRT
===============================
Converts YOLO models to TensorRT engines for GPU acceleration without PyTorch CUDA.
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ai_engine.optimization.tensorrt_inference import TensorRTConverter


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO model to TensorRT')
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                       help='Path to YOLO model (.pt file)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for TensorRT engine (default: model_name.engine)')
    parser.add_argument('--size', type=str, default='640,480',
                       help='Input size as width,height (default: 640,480)')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision mode (default: fp16)')
    
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
        output_path = model_path.parent / f"{model_path.stem}.engine"
    else:
        output_path = args.output
    
    print("=" * 70)
    print("YOLO to TensorRT Converter")
    print("=" * 70)
    print(f"Model: {args.model}")
    print(f"Output: {output_path}")
    print(f"Input size: {input_size[0]}x{input_size[1]}")
    print(f"Precision: {args.precision}")
    print("=" * 70)
    print()
    
    try:
        TensorRTConverter.convert_yolo_to_tensorrt(
            model_path=args.model,
            output_path=str(output_path),
            input_size=input_size,
            precision=args.precision
        )
        print()
        print("=" * 70)
        print("✓ Conversion successful!")
        print(f"TensorRT engine: {output_path}")
        print("=" * 70)
        return 0
    except Exception as e:
        print(f"\n✗ Conversion failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
