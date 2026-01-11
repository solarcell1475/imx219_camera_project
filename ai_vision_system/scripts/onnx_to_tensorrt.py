#!/usr/bin/env python3
"""
Convert ONNX to TensorRT Engine
================================
Converts ONNX models to TensorRT engines using TensorRT's ONNX parser.
This uses CUDA 12.6 directly - no PyTorch CUDA needed!
"""

import sys
import argparse
from pathlib import Path

# Add system Python path for TensorRT
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    print("✓ TensorRT and PyCUDA loaded successfully")
except ImportError as e:
    print(f"Error: TensorRT or PyCUDA not available: {e}")
    print("TensorRT should be installed system-wide on Jetson")
    print("Install PyCUDA with: pip install pycuda")
    sys.exit(1)


def build_engine(onnx_path: str, engine_path: str, precision: str = 'fp16', 
                 input_shape: tuple = (1, 3, 480, 480)):
    """
    Build TensorRT engine from ONNX model
    
    Args:
        onnx_path: Path to ONNX model
        engine_path: Output path for TensorRT engine
        precision: 'fp32', 'fp16', or 'int8'
        input_shape: Input shape (batch, channels, height, width)
    """
    print("=" * 70)
    print("ONNX to TensorRT Converter")
    print("=" * 70)
    print(f"ONNX Model: {onnx_path}")
    print(f"Output: {engine_path}")
    print(f"Precision: {precision}")
    print(f"Input Shape: {input_shape}")
    print("=" * 70)
    print()
    
    # Create TensorRT logger
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX model
    print("Parsing ONNX model...")
    with open(onnx_path, 'rb') as model:
        if not parser.parse(model.read()):
            print("ERROR: Failed to parse ONNX file")
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return False
    
    print("✓ ONNX model parsed successfully")
    
    # Configure builder
    config = builder.create_builder_config()
    # TensorRT 10.x uses memory_pool_limit instead of max_workspace_size
    if hasattr(config, 'memory_pool_limit'):
        config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    elif hasattr(config, 'max_workspace_size'):
        config.max_workspace_size = 1 << 30  # 1GB
    
    # Set precision
    if precision == 'fp16':
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            print("✓ FP16 precision enabled")
        else:
            print("⚠ FP16 not supported, using FP32")
    elif precision == 'int8':
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            print("✓ INT8 precision enabled")
        else:
            print("⚠ INT8 not supported, using FP32")
    
    # Build engine
    print()
    print("Building TensorRT engine (this may take 5-10 minutes)...")
    print("Please wait...")
    
    # TensorRT 10.x API
    try:
        # Try TensorRT 10.x API (build_serialized_network)
        if hasattr(builder, 'build_serialized_network'):
            serialized_engine = builder.build_serialized_network(network, config)
            if serialized_engine is None:
                print("✗ Failed to build TensorRT engine")
                return False
            # Save engine
            print("Saving TensorRT engine...")
            with open(engine_path, 'wb') as f:
                f.write(serialized_engine)
        else:
            # Fallback to older API (build_engine)
            engine = builder.build_engine(network, config)
            if engine is None:
                print("✗ Failed to build TensorRT engine")
                return False
            # Save engine
            print("Saving TensorRT engine...")
            with open(engine_path, 'wb') as f:
                f.write(engine.serialize())
    except Exception as e:
        print(f"✗ Error building engine: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print()
    print("=" * 70)
    print("✓ TensorRT engine built successfully!")
    print(f"Engine saved: {engine_path}")
    print("=" * 70)
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Convert ONNX to TensorRT')
    parser.add_argument('--onnx', type=str, default='yolov11n.onnx',
                       help='Path to ONNX model')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path for TensorRT engine')
    parser.add_argument('--precision', type=str, default='fp16',
                       choices=['fp32', 'fp16', 'int8'],
                       help='Precision mode (default: fp16)')
    parser.add_argument('--size', type=str, default='480,480',
                       help='Input size as height,width (default: 480,480)')
    
    args = parser.parse_args()
    
    # Parse input size
    try:
        height, width = map(int, args.size.split(','))
        input_shape = (1, 3, height, width)
    except:
        print("Error: Invalid size format. Use height,width (e.g., 480,480)")
        return 1
    
    # Determine output path
    if args.output is None:
        onnx_path = Path(args.onnx)
        output_path = onnx_path.parent / f"{onnx_path.stem}.engine"
    else:
        output_path = args.output
    
    # Check if ONNX file exists
    if not Path(args.onnx).exists():
        print(f"Error: ONNX file not found: {args.onnx}")
        return 1
    
    # Build engine
    success = build_engine(
        onnx_path=args.onnx,
        engine_path=str(output_path),
        precision=args.precision,
        input_shape=input_shape
    )
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())
