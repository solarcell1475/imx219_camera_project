#!/usr/bin/env python3
"""
YOLO Installation and Setup Script
====================================
Installs Ultralytics YOLO, PyTorch for Jetson, and verifies CUDA/cuDNN availability.
"""

import subprocess
import sys
import os
from pathlib import Path


class YOLOInstaller:
    """YOLO installation and verification"""
    
    def __init__(self):
        self.install_status = {
            'pytorch': False,
            'torchvision': False,
            'ultralytics': False,
            'cuda_available': False,
            'cudnn_available': False
        }
    
    def check_python_version(self):
        """Check Python version"""
        version = sys.version_info
        if version.major < 3 or (version.major == 3 and version.minor < 8):
            print("Error: Python 3.8+ required")
            return False
        print(f"✓ Python {version.major}.{version.minor}.{version.micro}")
        return True
    
    def check_cuda_available(self):
        """Check if CUDA is available in system"""
        try:
            result = subprocess.run(
                ['nvcc', '--version'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                print("✓ CUDA compiler found")
                return True
        except:
            pass
        
        # Check for CUDA in /usr/local/cuda
        if Path("/usr/local/cuda").exists():
            print("✓ CUDA directory found")
            return True
        
        print("⚠ CUDA not found (may still work with PyTorch)")
        return False
    
    def install_pytorch_jetson(self):
        """Install PyTorch for Jetson (ARM64)"""
        print("\nInstalling PyTorch for Jetson...")
        print("Note: This may take several minutes...")
        
        # PyTorch installation for Jetson
        # Check if already installed
        try:
            import torch
            print(f"✓ PyTorch already installed: {torch.__version__}")
            self.install_status['pytorch'] = True
            return True
        except ImportError:
            pass
        
        # Install PyTorch for Jetson
        # Note: User should install from NVIDIA's pre-built wheels
        print("Installing PyTorch...")
        print("For Jetson, install from: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
        print("Or use: pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118")
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
            ])
            
            # Try installing PyTorch
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision',
                '--index-url', 'https://download.pytorch.org/whl/cu118'
            ])
            
            import torch
            print(f"✓ PyTorch installed: {torch.__version__}")
            self.install_status['pytorch'] = True
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ PyTorch installation failed: {e}")
            print("Please install PyTorch manually for Jetson")
            return False
        except ImportError:
            print("✗ PyTorch import failed after installation")
            return False
    
    def install_ultralytics(self):
        """Install Ultralytics YOLO"""
        print("\nInstalling Ultralytics YOLO...")
        
        try:
            import ultralytics
            print(f"✓ Ultralytics already installed: {ultralytics.__version__}")
            self.install_status['ultralytics'] = True
            return True
        except ImportError:
            pass
        
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', 'ultralytics'
            ])
            
            import ultralytics
            print(f"✓ Ultralytics installed: {ultralytics.__version__}")
            self.install_status['ultralytics'] = True
            return True
        except subprocess.CalledProcessError as e:
            print(f"✗ Ultralytics installation failed: {e}")
            return False
        except ImportError:
            print("✗ Ultralytics import failed after installation")
            return False
    
    def verify_pytorch_cuda(self):
        """Verify PyTorch CUDA availability"""
        try:
            import torch
            if torch.cuda.is_available():
                print(f"✓ PyTorch CUDA available")
                print(f"  CUDA Version: {torch.version.cuda}")
                print(f"  GPU Device: {torch.cuda.get_device_name(0)}")
                print(f"  GPU Count: {torch.cuda.device_count()}")
                self.install_status['cuda_available'] = True
                return True
            else:
                print("✗ PyTorch CUDA not available")
                return False
        except ImportError:
            print("✗ PyTorch not installed")
            return False
    
    def verify_cudnn(self):
        """Verify cuDNN availability"""
        try:
            import torch
            if torch.backends.cudnn.enabled:
                print(f"✓ cuDNN enabled")
                print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
                self.install_status['cudnn_available'] = True
                return True
            else:
                print("⚠ cuDNN not enabled")
                return False
        except:
            print("⚠ Could not verify cuDNN")
            return False
    
    def test_yolo_import(self):
        """Test YOLO import and basic functionality"""
        print("\nTesting YOLO import...")
        try:
            from ultralytics import YOLO
            print("✓ YOLO import successful")
            
            # Try to load a small model
            print("Testing model loading...")
            model = YOLO('yolov8n.pt')  # Nano model
            print("✓ Model loading successful")
            
            return True
        except Exception as e:
            print(f"✗ YOLO test failed: {e}")
            return False
    
    def install_all(self):
        """Install all components"""
        print("=" * 70)
        print("YOLO Installation and Setup")
        print("=" * 70)
        
        if not self.check_python_version():
            return False
        
        self.check_cuda_available()
        
        if not self.install_pytorch_jetson():
            print("\n⚠ PyTorch installation incomplete. Please install manually.")
            print("See: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048")
        
        if not self.install_ultralytics():
            return False
        
        self.verify_pytorch_cuda()
        self.verify_cudnn()
        
        if not self.test_yolo_import():
            return False
        
        print("\n" + "=" * 70)
        print("Installation Summary")
        print("=" * 70)
        for component, status in self.install_status.items():
            status_str = "✓" if status else "✗"
            print(f"{status_str} {component}")
        
        all_ok = all([
            self.install_status['pytorch'],
            self.install_status['ultralytics'],
            self.install_status['cuda_available']
        ])
        
        if all_ok:
            print("\n✓ YOLO setup complete!")
            return True
        else:
            print("\n⚠ Some components may need manual installation")
            return False


def main():
    """Main installation function"""
    installer = YOLOInstaller()
    success = installer.install_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
