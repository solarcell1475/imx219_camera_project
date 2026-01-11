#!/usr/bin/env python3
"""
YOLO Setup Verification
========================
Verifies YOLO installation and CUDA/cuDNN availability.
"""

import sys


def verify_setup():
    """Verify YOLO setup"""
    print("=" * 70)
    print("YOLO Setup Verification")
    print("=" * 70)
    print()
    
    # Check Python
    print("Python Version:")
    print(f"  {sys.version}")
    print()
    
    # Check PyTorch
    print("PyTorch:")
    try:
        import torch
        print(f"  Version: {torch.__version__}")
        print(f"  CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  GPU Count: {torch.cuda.device_count()}")
        print(f"  cuDNN Enabled: {torch.backends.cudnn.enabled}")
        if torch.backends.cudnn.enabled:
            print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
    except ImportError:
        print("  ✗ PyTorch not installed")
        return False
    print()
    
    # Check Ultralytics
    print("Ultralytics YOLO:")
    try:
        from ultralytics import YOLO
        import ultralytics
        print(f"  Version: {ultralytics.__version__}")
        print("  ✓ Import successful")
        
        # Test model loading
        print("  Testing model loading...")
        try:
            model = YOLO('yolov8n.pt')
            print("  ✓ Model loading successful")
        except Exception as e:
            if "weights_only" in str(e):
                # Handle PyTorch 2.6+ requirement
                import torch
                torch.serialization.add_safe_globals(['ultralytics.nn.tasks.DetectionModel'])
                model = YOLO('yolov8n.pt')
                print("  ✓ Model loading successful (with weights_only fix)")
            else:
                raise
    except ImportError:
        print("  ✗ Ultralytics not installed")
        return False
    except Exception as e:
        print(f"  ✗ Error: {e}")
        return False
    print()
    
    print("=" * 70)
    print("✓ Setup verification complete!")
    print("=" * 70)
    return True


if __name__ == "__main__":
    success = verify_setup()
    sys.exit(0 if success else 1)
