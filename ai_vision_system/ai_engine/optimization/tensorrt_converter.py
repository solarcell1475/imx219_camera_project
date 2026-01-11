#!/usr/bin/env python3
"""
TensorRT Optimization for YOLO
================================
Converts YOLO PyTorch models to TensorRT engines with FP16/INT8 quantization
for maximum performance on Jetson.
"""

import os
from pathlib import Path
from typing import Optional, Dict
import json


class TensorRTConverter:
    """TensorRT model conversion and optimization"""
    
    def __init__(self, engine_cache_dir: Optional[str] = None):
        """
        Initialize TensorRT converter
        
        Args:
            engine_cache_dir: Directory to cache TensorRT engines
        """
        if engine_cache_dir is None:
            cache_home = Path.home() / ".yolo_tensorrt"
        else:
            cache_home = Path(engine_cache_dir)
        
        self.cache_dir = cache_home
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.metadata_file = self.cache_dir / "tensorrt_metadata.json"
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load TensorRT metadata"""
        if self.metadata_file.exists():
            try:
                with open(self.metadata_file, 'r') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_metadata(self):
        """Save TensorRT metadata"""
        try:
            with open(self.metadata_file, 'w') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save metadata: {e}")
    
    def _get_engine_path(self, model_name: str, precision: str = 'fp16') -> Path:
        """Get TensorRT engine file path"""
        engine_name = f"{Path(model_name).stem}_{precision}.engine"
        return self.cache_dir / engine_name
    
    def is_tensorrt_available(self) -> bool:
        """Check if TensorRT is available"""
        try:
            import tensorrt as trt
            return True
        except ImportError:
            return False
    
    def convert_to_tensorrt(self, yolo_model, model_name: str, 
                          precision: str = 'fp16',
                          input_size: int = 640,
                          batch_size: int = 1) -> Optional[str]:
        """
        Convert YOLO model to TensorRT
        
        Args:
            yolo_model: YOLO model object
            model_name: Name of the model (for caching)
            precision: 'fp32', 'fp16', or 'int8'
            input_size: Input image size (default 640)
            batch_size: Batch size for engine
        
        Returns:
            Path to TensorRT engine file, or None if failed
        """
        if not self.is_tensorrt_available():
            print("TensorRT not available. Install TensorRT for Jetson.")
            return None
        
        engine_path = self._get_engine_path(model_name, precision)
        
        # Check if engine already exists
        if engine_path.exists():
            print(f"TensorRT engine already exists: {engine_path}")
            return str(engine_path)
        
        print(f"Converting {model_name} to TensorRT ({precision})...")
        print("This may take 5-10 minutes...")
        
        try:
            # Export to ONNX first (if not already done)
            onnx_path = self.cache_dir / f"{Path(model_name).stem}.onnx"
            if not onnx_path.exists():
                print("Exporting to ONNX...")
                yolo_model.export(format='onnx', imgsz=input_size, simplify=True)
                # Find the exported ONNX file
                onnx_files = list(Path(yolo_model.ckpt_path).parent.glob("*.onnx"))
                if onnx_files:
                    import shutil
                    shutil.copy2(onnx_files[0], onnx_path)
            
            # Convert ONNX to TensorRT
            print(f"Converting ONNX to TensorRT ({precision})...")
            yolo_model.export(
                format='engine',
                imgsz=input_size,
                half=(precision == 'fp16'),
                int8=(precision == 'int8'),
                device=0,  # GPU 0
                simplify=True
            )
            
            # Find the exported engine file
            engine_files = list(Path(yolo_model.ckpt_path).parent.glob("*.engine"))
            if engine_files:
                import shutil
                shutil.copy2(engine_files[0], engine_path)
                print(f"✓ TensorRT engine saved: {engine_path}")
                
                # Update metadata
                key = f"{model_name}_{precision}"
                self.metadata[key] = {
                    'engine_path': str(engine_path),
                    'precision': precision,
                    'input_size': input_size,
                    'batch_size': batch_size
                }
                self._save_metadata()
                
                return str(engine_path)
            else:
                print("✗ Engine file not found after export")
                return None
                
        except Exception as e:
            print(f"✗ TensorRT conversion failed: {e}")
            return None
    
    def load_tensorrt_engine(self, engine_path: str):
        """
        Load TensorRT engine (for inference)
        
        Note: This is a placeholder. Actual TensorRT inference
        should be done through YOLO's built-in TensorRT support.
        """
        if not Path(engine_path).exists():
            raise FileNotFoundError(f"Engine file not found: {engine_path}")
        
        return engine_path
    
    def get_engine_info(self, model_name: str, precision: str = 'fp16') -> Optional[Dict]:
        """Get information about a TensorRT engine"""
        engine_path = self._get_engine_path(model_name, precision)
        key = f"{model_name}_{precision}"
        
        if key in self.metadata:
            info = self.metadata[key].copy()
            info['exists'] = engine_path.exists()
            if engine_path.exists():
                info['size_mb'] = round(engine_path.stat().st_size / (1024 * 1024), 2)
            return info
        
        return None
    
    def list_engines(self) -> list:
        """List all available TensorRT engines"""
        engines = []
        for engine_file in self.cache_dir.glob("*.engine"):
            engines.append({
                'name': engine_file.name,
                'path': str(engine_file),
                'size_mb': round(engine_file.stat().st_size / (1024 * 1024), 2)
            })
        return engines


def main():
    """Test function"""
    print("Testing TensorRT Converter...")
    print("=" * 70)
    
    converter = TensorRTConverter()
    
    if not converter.is_tensorrt_available():
        print("⚠ TensorRT not available")
        print("TensorRT is typically pre-installed on Jetson devices.")
        print("If missing, install from: https://developer.nvidia.com/tensorrt")
        return 0
    
    print("✓ TensorRT available")
    
    # List existing engines
    engines = converter.list_engines()
    if engines:
        print(f"\nFound {len(engines)} TensorRT engine(s):")
        for engine in engines:
            print(f"  {engine['name']} ({engine['size_mb']} MB)")
    else:
        print("\nNo TensorRT engines found")
        print("Engines will be created when models are exported with TensorRT format")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
