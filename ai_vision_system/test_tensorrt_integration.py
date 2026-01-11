#!/usr/bin/env python3
"""
Test TensorRT Integration in Main Application
==============================================
Tests if TensorRT inference is properly integrated into the main application.
"""

import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Setup display environment
try:
    from utils.display_fix import setup_display_environment
    setup_display_environment()
except:
    pass

import numpy as np
from main import YOLOVisionSystem


def test_tensorrt_integration():
    """Test TensorRT integration in main application"""
    print("=" * 70)
    print("TensorRT Integration Test")
    print("=" * 70)
    print()

    try:
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        print("1. Creating test system...")

        # Initialize system with minimal setup (skip camera initialization)
        system = YOLOVisionSystem()

        # Manually initialize just the inference components (skip camera)
        print("2. Initializing inference components...")
        system.config = system._load_config()
        model_config = system.config.get('model', {})
        system.confidence_threshold = system.config.get('confidence_threshold', 0.25)

        # Initialize model manager and load model
        from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
        system.model_manager = YOLOModelManager()
        system.yolo_model = system.model_manager.load_model(
            version=model_config.get('version', 'v11'),
            size=model_config.get('size', 'n'),
            device=system.config.get('device', 'cpu')
        )

        # Initialize inference (this is where TensorRT should be detected)
        inference_resolution = system.config.get('inference_resolution', [640, 640])
        inference_size = tuple(inference_resolution) if isinstance(inference_resolution, list) else inference_resolution

        # Try TensorRT inference first (GPU acceleration)
        tensorrt_engine = system._find_tensorrt_engine()
        if tensorrt_engine and system.config.get('use_tensorrt', True):
            try:
                print(f"   Initializing TensorRT inference with: {tensorrt_engine}")
                from ai_engine.inference.tensorrt_dual_inference import TensorRTDualInference
                system.inference = TensorRTDualInference(
                    tensorrt_engine,
                    confidence_threshold=system.confidence_threshold,
                    iou_threshold=system.config['iou_threshold'],
                    inference_size=inference_size
                )
                print("   ✓ TensorRT GPU acceleration enabled!")
            except Exception as e:
                print(f"   ⚠ TensorRT initialization failed: {e}")
                print("   Falling back to PyTorch inference...")
                tensorrt_engine = None

        if not tensorrt_engine:
            # Fallback to PyTorch inference
            from ai_engine.inference.dual_yolo_inference import ParallelDualInference
            print("   Initializing PyTorch parallel inference (CPU)...")
            system.inference = ParallelDualInference(
                system.yolo_model,
                confidence_threshold=system.confidence_threshold,
                iou_threshold=system.config['iou_threshold'],
                device=system.config['device'],
                inference_size=inference_size
            )

        print("3. Checking inference engine type...")
        inference_type = type(system.inference).__name__
        print(f"   Inference engine: {inference_type}")

        # Test inference
        print("3. Testing inference...")
        start_time = time.time()

        if hasattr(system.inference, 'infer_single'):
            detections = system.inference.infer_single(test_image, (640, 640))
            inference_time = (time.time() - start_time) * 1000
            print(f"   ✓ Single inference completed in {inference_time:.2f} ms")
            print(f"   Detections found: {len(detections)}")
        else:
            print("   ✗ No infer_single method found")

        # Check if TensorRT is being used
        is_tensorrt = "TensorRT" in inference_type
        print(f"4. TensorRT status: {'✓ ACTIVE' if is_tensorrt else '✗ NOT ACTIVE'}")

        # Get statistics
        if hasattr(system.inference, 'get_statistics'):
            stats = system.inference.get_statistics()
            print("5. Inference statistics:")
            for key, value in stats.items():
                if isinstance(value, float):
                    print(f"      {key}: {value:.2f}")
                else:
                    print(f"      {key}: {value}")

        print()
        print("=" * 70)
        print("Integration Test Results")
        print("=" * 70)
        print(f"Engine Type: {inference_type}")
        print(f"TensorRT Active: {is_tensorrt}")
        if 'inference_time' in locals():
            print(f"Inference Time: {inference_time:.2f} ms")
        print(f"Performance: {'Excellent' if is_tensorrt else 'CPU Mode'}")

        return is_tensorrt

    except Exception as e:
        print(f"\n✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_tensorrt_integration()
    sys.exit(0 if success else 1)