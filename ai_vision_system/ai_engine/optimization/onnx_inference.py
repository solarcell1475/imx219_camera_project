#!/usr/bin/env python3
"""
ONNX Runtime Inference Engine for YOLO
========================================
ONNX Runtime with CUDA execution provider - uses CUDA 12.6 directly without PyTorch CUDA.
This provides GPU acceleration without needing PyTorch CUDA wheels.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False
    print("Warning: ONNX Runtime not available. Install with: pip install onnxruntime-gpu")


class ONNXInference:
    """ONNX Runtime-based YOLO inference engine"""
    
    def __init__(self, onnx_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, providers: Optional[List[str]] = None):
        """
        Initialize ONNX Runtime inference engine
        
        Args:
            onnx_path: Path to ONNX model file (.onnx)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            providers: Execution providers (default: ['CUDAExecutionProvider', 'CPUExecutionProvider'])
        """
        if not ONNX_AVAILABLE:
            raise ImportError("ONNX Runtime not available. Install with: pip install onnxruntime-gpu")
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.onnx_path = onnx_path
        
        # Setup execution providers (prefer CUDA, fallback to CPU)
        if providers is None:
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        
        # Create inference session
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.session = ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=providers
        )
        
        # Get input/output details
        self.input_name = self.session.get_inputs()[0].name
        self.input_shape = self.session.get_inputs()[0].shape
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        print(f"✓ ONNX Runtime initialized")
        print(f"  Model: {onnx_path}")
        print(f"  Providers: {self.session.get_providers()}")
        print(f"  Input: {self.input_name}, Shape: {self.input_shape}")
        print(f"  Outputs: {self.output_names}")
        
        # Statistics
        self.stats = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0
        }
        
        # Class names (COCO dataset - 80 classes)
        self.class_names = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck',
            'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench',
            'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
            'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
            'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
            'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
            'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
            'toothbrush'
        ]
    
    def infer(self, image: np.ndarray, inference_size: Tuple[int, int] = (640, 480)) -> List[Dict]:
        """
        Run inference on image using ONNX Runtime
        
        Args:
            image: Input image (BGR format)
            inference_size: Target size for inference (width, height)
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self._preprocess(image, inference_size)
        
        # Run inference
        outputs = self.session.run(
            self.output_names,
            {self.input_name: preprocessed}
        )
        
        # Postprocess
        detections = self._postprocess(outputs[0], image.shape[:2], inference_size)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['inference_count'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['inference_count']
        )
        
        return detections
    
    def _preprocess(self, image: np.ndarray, inference_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for ONNX inference"""
        # Get model input shape (usually square)
        model_h = self.input_shape[2] if len(self.input_shape) >= 3 else inference_size[1]
        model_w = self.input_shape[3] if len(self.input_shape) >= 4 else inference_size[0]
        
        # Use square input if model expects square (common for YOLO)
        if model_h == model_w:
            target_size = model_h
        else:
            # Use provided inference_size
            target_w, target_h = inference_size
        
        # Resize to target size (maintain aspect ratio with letterbox if needed)
        h, w = image.shape[:2]
        if model_h == model_w:
            # Square input - use letterbox resize
            scale = min(target_size / w, target_size / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            
            # Pad to square
            top = (target_size - new_h) // 2
            bottom = target_size - new_h - top
            left = (target_size - new_w) // 2
            right = target_size - new_w - left
            resized = cv2.copyMakeBorder(resized, top, bottom, left, right, 
                                        cv2.BORDER_CONSTANT, value=(114, 114, 114))
        else:
            # Non-square input
            resized = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0
        
        # Transpose to CHW format
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0).astype(np.float32)
        
        return batched
    
    def _postprocess(self, output: np.ndarray, original_shape: Tuple[int, int],
                    inference_size: Tuple[int, int]) -> List[Dict]:
        """
        Postprocess ONNX output to detection format
        
        Args:
            output: Raw ONNX output (shape: [batch, num_detections, 85] or [batch, 84, 8400])
            original_shape: Original image shape (height, width)
            inference_size: Inference size (width, height)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Handle different output formats
        if len(output.shape) == 3:
            # Format: [batch, num_detections, 85] (x, y, w, h, conf, class_scores...)
            output = output[0]  # Remove batch dimension
        elif len(output.shape) == 2:
            # Format: [num_detections, 85]
            output = output
        
        scale_x = original_shape[1] / inference_size[0]
        scale_y = original_shape[0] / inference_size[1]
        
        for detection in output:
            # Parse detection
            # YOLO format: [x_center, y_center, width, height, confidence, class_scores...]
            x_center = detection[0] * scale_x
            y_center = detection[1] * scale_y
            width = detection[2] * scale_x
            height = detection[3] * scale_y
            confidence = detection[4]
            
            # Get class scores
            class_scores = detection[5:]
            class_id = np.argmax(class_scores)
            class_confidence = class_scores[class_id] * confidence
            
            # Filter by confidence
            if class_confidence < self.confidence_threshold:
                continue
            
            # Convert to bbox format [x1, y1, x2, y2]
            x1 = x_center - width / 2
            y1 = y_center - height / 2
            x2 = x_center + width / 2
            y2 = y_center + height / 2
            
            # Get class name
            class_name = self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}'
            
            detections.append({
                'class_id': int(class_id),
                'class_name': class_name,
                'confidence': float(class_confidence),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'camera_id': None
            })
        
        return detections
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        return self.stats.copy()


class ONNXConverter:
    """Convert YOLO models to ONNX format"""
    
    @staticmethod
    def convert_yolo_to_onnx(model_path: str, output_path: str,
                             input_size: Tuple[int, int] = (640, 480),
                             simplify: bool = True) -> str:
        """
        Convert YOLO model to ONNX format
        
        Args:
            model_path: Path to YOLO model (.pt file)
            output_path: Output path for ONNX model
            input_size: Input size (width, height)
            simplify: Simplify ONNX model
            
        Returns:
            Path to converted ONNX file
        """
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            
            print(f"Exporting to ONNX format...")
            # Export to ONNX (doesn't require CUDA!)
            # Use opset=11 for compatibility with older ONNX Runtime versions
            result_path = model.export(
                format='onnx',
                imgsz=input_size[1],  # Height
                simplify=simplify,
                device='cpu',  # Can use CPU for export
                opset=11  # Use opset 11 for compatibility
            )
            
            # Move to output path if different
            import shutil
            if result_path != output_path:
                shutil.move(result_path, output_path)
            
            # Convert IR version to 11 for compatibility with older ONNX Runtime
            try:
                import onnx
                print("Converting IR version to 11 for compatibility...")
                model = onnx.load(output_path)
                if model.ir_version > 11:
                    model.ir_version = 11
                    onnx.save(model, output_path)
                    print("✓ IR version converted to 11")
            except Exception as e:
                print(f"Warning: Could not convert IR version: {e}")
            
            print(f"✓ ONNX model saved: {output_path}")
            return output_path
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert to ONNX: {e}")
