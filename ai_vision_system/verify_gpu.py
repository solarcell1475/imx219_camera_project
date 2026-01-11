#!/usr/bin/env python3
"""
GPU Verification Script for YOLO System
========================================
Verifies that PyTorch can access the GPU for YOLO inference.
"""

import sys
import subprocess

def check_cuda_system():
    """Check CUDA installation at system level"""
    print("=" * 70)
    print("System CUDA Check")
    print("=" * 70)
    
    try:
        result = subprocess.run(['nvcc', '--version'], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✓ CUDA compiler found")
            version_line = [l for l in result.stdout.split('\n') if 'release' in l.lower()]
            if version_line:
                print(f"  {version_line[0].strip()}")
            return True
    except Exception as e:
        print(f"✗ CUDA compiler not found: {e}")
    
    # Check for CUDA directory
    import os
    cuda_dirs = [d for d in os.listdir('/usr/local') if d.startswith('cuda')]
    if cuda_dirs:
        print(f"✓ CUDA directories found: {', '.join(cuda_dirs)}")
        return True
    
    print("✗ CUDA not found")
    return False

def check_pytorch_cuda():
    """Check PyTorch CUDA availability"""
    print("\n" + "=" * 70)
    print("PyTorch CUDA Check")
    print("=" * 70)
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
            print(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                props = torch.cuda.get_device_properties(i)
                print(f"    Memory: {props.total_memory / 1024**3:.2f} GB")
            return True
        else:
            print("\n⚠ CUDA not available in PyTorch")
            print("This usually means:")
            print("  1. PyTorch was installed without CUDA support")
            print("  2. PyTorch version is incompatible with your CUDA version")
            print("  3. CUDA libraries are not properly linked")
            return False
    except ImportError:
        print("✗ PyTorch not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking PyTorch: {e}")
        return False

def check_yolo_gpu():
    """Check if YOLO can use GPU"""
    print("\n" + "=" * 70)
    print("YOLO GPU Check")
    print("=" * 70)
    
    try:
        from ultralytics import YOLO
        import torch
        
        if not torch.cuda.is_available():
            print("⚠ Cannot test YOLO GPU - PyTorch CUDA not available")
            return False
        
        print("Loading YOLOv8n model...")
        model = YOLO('yolov8n.pt')
        
        # Check device
        device = next(model.model.parameters()).device
        print(f"Model device: {device}")
        
        if device.type == 'cuda':
            print("✓ YOLO model is on GPU")
            return True
        else:
            print("⚠ YOLO model is on CPU")
            print("  Try: model.to('cuda') or use device='cuda' in predict()")
            return False
    except ImportError:
        print("✗ Ultralytics YOLO not installed")
        return False
    except Exception as e:
        print(f"✗ Error checking YOLO: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all checks"""
    print("\n")
    print("=" * 70)
    print("GPU Verification for YOLO AI Vision System")
    print("=" * 70)
    print()
    
    cuda_system = check_cuda_system()
    pytorch_cuda = check_pytorch_cuda()
    yolo_gpu = check_yolo_gpu()
    
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"System CUDA:     {'✓' if cuda_system else '✗'}")
    print(f"PyTorch CUDA:    {'✓' if pytorch_cuda else '✗'}")
    print(f"YOLO GPU:        {'✓' if yolo_gpu else '✗'}")
    print()
    
    if pytorch_cuda and yolo_gpu:
        print("✓ GPU acceleration is ready!")
        return 0
    else:
        print("⚠ GPU acceleration is not available")
        print("\nNext steps:")
        if not pytorch_cuda:
            print("1. Install PyTorch with CUDA support for Jetson")
            print("   See: ai_vision_system/INSTALL_PYTORCH_JETSON.md")
        if pytorch_cuda and not yolo_gpu:
            print("2. Ensure YOLO models are loaded with device='cuda'")
        return 1

if __name__ == '__main__':
    sys.exit(main())
