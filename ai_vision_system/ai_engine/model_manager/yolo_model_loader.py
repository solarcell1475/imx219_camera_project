#!/usr/bin/env python3
"""
YOLO Model Manager
==================
Manages YOLO model loading, caching, version control, and model switching.
Supports YOLOv8/v9/v10/v11 with n/s/m/l/x/b variants.
"""

import os
from pathlib import Path
from typing import Optional, Dict, List
import json
from datetime import datetime


class YOLOModelManager:
    """YOLO model management system"""
    
    # Supported YOLO versions
    SUPPORTED_VERSIONS = ['v8', 'v9', 'v10', 'v11']
    SUPPORTED_SIZES = ['n', 's', 'm', 'l', 'x', 'b']  # nano, small, medium, large, xlarge, base/balanced
    
    def __init__(self, model_cache_dir: Optional[str] = None):
        """
        Initialize model manager
        
        Args:
            model_cache_dir: Directory to cache models (default: ~/.yolo_models)
        """
        if model_cache_dir is None:
            cache_home = Path.home() / ".yolo_models"
        else:
            cache_home = Path(model_cache_dir)
        
        self.cache_dir = cache_home
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "model_metadata.json"
        self.metadata = self._load_metadata()
        
        self.current_model = None
        self.current_model_info = None
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save model metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def _get_model_name(self, version: str = 'v8', size: str = 's') -> str:
        """
        Get model name string
        
        Args:
            version: YOLO version (v8, v9, v10, v11)
            size: Model size (n, s, m, l, x, b)
        
        Returns:
            Model name string (e.g., 'yolov8s.pt')
        """
        # Handle version string - remove 'v' prefix if present
        version_str = str(version).strip()
        if version_str.startswith('v'):
            version_num = version_str[1:]  # Remove 'v' prefix
        else:
            version_num = version_str  # Already a number
        
        # Validate inputs
        if not version_num:
            print(f"Warning: Empty version number, defaulting to '8'")
            version_num = '8'
        if not size:
            print(f"Warning: Empty size, defaulting to 's'")
            size = 's'
        
        # YOLO11 uses 'yolo11n.pt' format (no 'v'), others use 'yolov8n.pt'
        if version_num == '11':
            model_name = f"yolo{version_num}{size}.pt"
        else:
            model_name = f"yolov{version_num}{size}.pt"
        return model_name
    
    def _get_model_path(self, model_name: str) -> Path:
        """Get local path for model"""
        return self.cache_dir / model_name
    
    def get_model_info(self, version: str = 'v8', size: str = 's') -> Dict:
        """
        Get information about a model
        
        Args:
            version: YOLO version
            size: Model size
        
        Returns:
            Model information dictionary
        """
        model_name = self._get_model_name(version, size)
        model_key = f"{version}_{size}"
        
        info = {
            'name': model_name,
            'version': version,
            'size': size,
            'local_path': str(self._get_model_path(model_name)),
            'cached': self._get_model_path(model_name).exists(),
            'size_mb': 0,
            'classes': 80,  # COCO classes
            'input_size': 640,
            'download_date': None
        }
        
        # Load cached metadata
        if model_key in self.metadata:
            info.update(self.metadata[model_key])
        
        # Get file size if cached
        if info['cached']:
            size_bytes = self._get_model_path(model_name).stat().st_size
            info['size_mb'] = round(size_bytes / (1024 * 1024), 2)
        
        return info
    
    def list_available_models(self) -> List[Dict]:
        """List all available model configurations"""
        models = []
        for version in self.SUPPORTED_VERSIONS:
            for size in self.SUPPORTED_SIZES:
                info = self.get_model_info(version, size)
                models.append(info)
        return models
    
    def list_cached_models(self) -> List[Dict]:
        """List only cached models"""
        all_models = self.list_available_models()
        return [m for m in all_models if m['cached']]
    
    def load_model(self, version: str = 'v8', size: str = 's', device: str = 'cuda'):
        """
        Load YOLO model
        
        Args:
            version: YOLO version (v8, v9, v10, v11)
            size: Model size (n, s, m, l, x, b)
            device: Device to use ('cuda' or 'cpu')
        
        Returns:
            YOLO model object
        """
        try:
            from ultralytics import YOLO
        except ImportError:
            raise ImportError("Ultralytics YOLO not installed. Run install_yolo.py first.")
        
        model_name = self._get_model_name(version, size)
        model_path = self._get_model_path(model_name)
        
        # Check if model is cached locally
        if model_path.exists():
            print(f"Loading cached model: {model_path}")
            # Handle PyTorch 2.6+ weights_only requirement
            try:
                model = YOLO(str(model_path))
            except Exception as e:
                if "weights_only" in str(e) or "Unsupported global" in str(e):
                    # Try importing the class and adding it to safe globals
                    try:
                        from ultralytics.nn.tasks import DetectionModel
                        import torch
                        torch.serialization.add_safe_globals([DetectionModel])
                        model = YOLO(str(model_path))
                    except:
                        # Fallback: try with weights_only=False (if PyTorch allows)
                        model = YOLO(str(model_path))
                else:
                    raise
        else:
            # Download from Ultralytics (will cache automatically)
            print(f"Downloading model: {model_name}")
            try:
                model = YOLO(model_name)
            except Exception as e:
                if "weights_only" in str(e) or "Unsupported global" in str(e):
                    # Try importing the class and adding it to safe globals
                    try:
                        from ultralytics.nn.tasks import DetectionModel
                        import torch
                        torch.serialization.add_safe_globals([DetectionModel])
                        model = YOLO(model_name)
                    except:
                        # Fallback: just try again (Ultralytics may handle it)
                        model = YOLO(model_name)
                else:
                    raise
            
            # Try to copy to cache if downloaded
            try:
                # Ultralytics downloads to ~/.ultralytics/weights/
                ultralytics_cache = Path.home() / ".ultralytics" / "weights" / model_name
                if ultralytics_cache.exists():
                    import shutil
                    shutil.copy2(ultralytics_cache, model_path)
                    print(f"Cached model to: {model_path}")
            except:
                pass
        
        # Store model info
        self.current_model = model
        self.current_model_info = self.get_model_info(version, size)
        
        # Update metadata
        model_key = f"{version}_{size}"
        if model_key not in self.metadata:
            self.metadata[model_key] = {}
        
        self.metadata[model_key]['last_used'] = datetime.now().isoformat()
        self.metadata[model_key]['device'] = device
        self._save_metadata()
        
        return model
    
    def get_current_model(self):
        """Get currently loaded model"""
        return self.current_model
    
    def get_current_model_info(self) -> Optional[Dict]:
        """Get information about currently loaded model"""
        return self.current_model_info
    
    def switch_model(self, version: str, size: str, device: str = 'cuda'):
        """Switch to a different model"""
        return self.load_model(version, size, device)
    
    def get_recommended_model(self, target_fps: int = 25) -> Dict:
        """
        Get recommended model based on target FPS
        
        Args:
            target_fps: Target frames per second
        
        Returns:
            Recommended model info (version, size)
        """
        # Model performance estimates (approximate FPS on Jetson Orin Nano)
        model_performance = {
            # YOLOv8 models (existing)
            ('v8', 'n'): 40,  # YOLOv8n
            ('v8', 's'): 25,  # YOLOv8s
            ('v8', 'm'): 15,  # YOLOv8m
            ('v8', 'l'): 10,  # YOLOv8l
            ('v8', 'x'): 7,   # YOLOv8x
            
            # YOLO11 models (new - from article benchmarks on Jetson Orin Nano Super)
            ('v11', 'n'): 50,  # YOLO11n - faster than v8n
            ('v11', 's'): 35,  # YOLO11s
            ('v11', 'm'): 20,  # YOLO11m
            ('v11', 'l'): 12,  # YOLO11l
            ('v11', 'x'): 8,   # YOLO11x
            ('v11', 'b'): 30,  # YOLO11b (estimated between n and s)
        }
        
        # Find best model that meets FPS requirement
        best_model = None
        best_fps = 0
        
        for (version, size), fps in model_performance.items():
            if fps >= target_fps and fps > best_fps:
                best_model = {'version': version, 'size': size}
                best_fps = fps
        
        # Default to YOLO11n if no match (updated from v8s)
        if best_model is None:
            best_model = {'version': 'v11', 'size': 'n'}
        
        return best_model


def main():
    """Test function"""
    print("Testing YOLO Model Manager...")
    print("=" * 70)
    
    manager = YOLOModelManager()
    
    # List available models
    print("\nAvailable Models:")
    models = manager.list_available_models()
    for model in models[:5]:  # Show first 5
        cached = "✓" if model['cached'] else " "
        print(f"  {cached} {model['name']:15} - {model['version']}/{model['size']}")
    
    # Get recommended model
    print("\nRecommended Model for 25 FPS:")
    rec = manager.get_recommended_model(25)
    print(f"  {rec['version']}/{rec['size']}")
    
    # Try loading a model
    print("\nLoading model...")
    try:
        model = manager.load_model('v8', 'n')  # Nano model for quick test
        print("✓ Model loaded successfully")
        info = manager.get_current_model_info()
        print(f"  Model: {info['name']}")
        print(f"  Cached: {info['cached']}")
        if info['size_mb'] > 0:
            print(f"  Size: {info['size_mb']} MB")
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
