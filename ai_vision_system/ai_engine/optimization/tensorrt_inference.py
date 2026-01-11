#!/usr/bin/env python3
"""
TensorRT Inference Engine for YOLO
===================================
TensorRT-based inference that uses CUDA 12.6 directly without PyTorch CUDA.
This provides 2-5x faster inference than PyTorch on Jetson devices.
"""

import numpy as np
import cv2
import time
from typing import List, Dict, Tuple, Optional
import os
from pathlib import Path

try:
    # Add system paths for TensorRT
    import sys
    if '/usr/lib/python3.10/dist-packages' not in sys.path:
        sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

    import tensorrt as trt
    import pycuda.driver as cuda
    # Don't auto-init pycuda here, let the user handle CUDA context
    TENSORRT_AVAILABLE = True
except ImportError as e:
    TENSORRT_AVAILABLE = False
    print(f"Warning: TensorRT or PyCUDA not available: {e}")


class TensorRTInference:
    """TensorRT-based YOLO inference engine"""
    
    def __init__(self, engine_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45):
        """
        Initialize TensorRT inference engine
        
        Args:
            engine_path: Path to TensorRT engine file (.engine)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT not available. Install TensorRT and pycuda.")
        
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.engine_path = engine_path
        
        # Load TensorRT engine with optimized settings
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)

        with open(engine_path, 'rb') as f:
            engine_data = f.read()

        # Deserialize engine
        self.engine = self.runtime.deserialize_cuda_engine(engine_data)

        # Create execution context with optimized settings
        self.context = self.engine.create_execution_context()

        # Enable CUDA graphs for better performance (TensorRT 8.5+)
        if hasattr(self.context, 'set_optimization_profile_async'):
            try:
                self.context.set_optimization_profile_async(0)
            except:
                pass
        
        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()
        
        # Get input/output shapes
        self.input_shape = self._get_input_shape()
        self.output_shape = self._get_output_shape()
        
        # Statistics
        self.stats = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0
        }
    
    def _allocate_buffers(self):
        """Allocate GPU memory buffers (TensorRT 10.x API)"""
        # Initialize CUDA context if not already done
        try:
            cuda.init()
            device = cuda.Device(0)  # Use GPU 0
            self.cuda_context = device.make_context()
        except:
            # Context might already be active
            self.cuda_context = None

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()
        
        # TensorRT 10.x uses tensor-based API instead of binding-based
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                tensor_shape = self.engine.get_tensor_shape(tensor_name)
                tensor_dtype = trt.nptype(self.engine.get_tensor_dtype(tensor_name))
                
                size = trt.volume(tensor_shape)
                if size < 0:  # Dynamic shape
                    size = abs(size) * 4  # Estimate
                
                # Allocate host and device buffers
                host_mem = cuda.pagelocked_empty(size, tensor_dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                bindings.append(int(device_mem))
                
                tensor_info = {
                    'name': tensor_name,
                    'host': host_mem,
                    'device': device_mem,
                    'shape': tensor_shape,
                    'dtype': tensor_dtype
                }
                
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    inputs.append(tensor_info)
                else:
                    outputs.append(tensor_info)
        else:
            # Legacy API (TensorRT 8.x and earlier)
            for binding in self.engine:
                size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                
                host_mem = cuda.pagelocked_empty(size, dtype)
                device_mem = cuda.mem_alloc(host_mem.nbytes)
                
                bindings.append(int(device_mem))
                
                if self.engine.binding_is_input(binding):
                    inputs.append({'host': host_mem, 'device': device_mem})
                else:
                    outputs.append({'host': host_mem, 'device': device_mem})
        
        return inputs, outputs, bindings, stream
    
    def _get_input_shape(self):
        """Get input shape from engine"""
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.INPUT:
                    shape = self.engine.get_tensor_shape(tensor_name)
                    return tuple(shape[1:])  # Remove batch dimension
        else:
            # Legacy API
            for binding in self.engine:
                if self.engine.binding_is_input(binding):
                    shape = self.engine.get_binding_shape(binding)
                    return tuple(shape[1:])  # Remove batch dimension
        return None
    
    def _get_output_shape(self):
        """Get output shape from engine"""
        if hasattr(self.engine, 'num_io_tensors'):
            # TensorRT 10.x API
            for i in range(self.engine.num_io_tensors):
                tensor_name = self.engine.get_tensor_name(i)
                if self.engine.get_tensor_mode(tensor_name) == trt.TensorIOMode.OUTPUT:
                    shape = self.engine.get_tensor_shape(tensor_name)
                    return tuple(shape[1:])  # Remove batch dimension
        else:
            # Legacy API
            for binding in self.engine:
                if not self.engine.binding_is_input(binding):
                    shape = self.engine.get_binding_shape(binding)
                    return tuple(shape[1:])  # Remove batch dimension
        return None
    
    def infer(self, image: np.ndarray, inference_size: Tuple[int, int] = (640, 480)) -> List[Dict]:
        """
        Run inference on image using TensorRT
        
        Args:
            image: Input image (BGR format)
            inference_size: Target size for inference (width, height)
            
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self._preprocess(image, inference_size)
        
        # Copy input to GPU and run inference (TensorRT 10.x API)
        if hasattr(self.context, 'set_tensor_address'):
            # TensorRT 10.x API
            input_tensor_name = self.inputs[0]['name']
            output_tensor_name = self.outputs[0]['name']
            
            # Set tensor addresses
            self.context.set_tensor_address(input_tensor_name, self.inputs[0]['device'])
            self.context.set_tensor_address(output_tensor_name, self.outputs[0]['device'])
            
            # Copy input to GPU
            np.copyto(self.inputs[0]['host'], preprocessed.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            # Run inference
            self.context.execute_async_v3(stream_handle=self.stream.handle)
            
            # Copy output from GPU
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        else:
            # Legacy API
            np.copyto(self.inputs[0]['host'], preprocessed.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            cuda.memcpy_dtoh_async(self.outputs[0]['host'], self.outputs[0]['device'], self.stream)
        
        self.stream.synchronize()
        
        # Postprocess
        output = self.outputs[0]['host']
        # Get actual output shape
        if hasattr(self.outputs[0], 'shape'):
            output_shape = self.outputs[0]['shape']
        else:
            # Estimate from size
            output_size = output.size
            # YOLO output is typically [1, 84, 8400] or [1, 25200, 84]
            if output_size % 84 == 0:
                output_shape = (output_size // 84, 84)
            else:
                output_shape = output.shape if hasattr(output, 'shape') else (output_size,)
        detections = self._postprocess(output, output_shape, image.shape[:2], inference_size)
        
        inference_time = (time.time() - start_time) * 1000
        
        # Update statistics
        self.stats['inference_count'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['inference_count']
        )
        
        return detections
    
    def _preprocess(self, image: np.ndarray, inference_size: Tuple[int, int]) -> np.ndarray:
        """Preprocess image for TensorRT"""
        # Use the model's expected input shape
        # input_shape is [channels, height, width] after removing batch dimension
        channels, height, width = self.input_shape

        # Resize to model's expected dimensions
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        # Transpose to CHW format (channels, height, width)
        transposed = np.transpose(normalized, (2, 0, 1))

        # Add batch dimension
        batched = np.expand_dims(transposed, axis=0).astype(np.float32)

        return batched
    
    def _postprocess(self, output: np.ndarray, output_shape: Tuple, 
                    original_shape: Tuple[int, int],
                    inference_size: Tuple[int, int]) -> List[Dict]:
        """
        Postprocess TensorRT output to detection format
        
        Args:
            output: Raw TensorRT output
            output_shape: Actual output shape
            original_shape: Original image shape (height, width)
            inference_size: Inference size (width, height)
            
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        # Reshape output based on actual shape
        # YOLO output format: [batch, num_detections, 85] or [batch, 84, num_detections]
        if len(output_shape) == 1:
            # Flattened output, try to reshape
            total_elements = output.size
            if total_elements % 84 == 0:
                # Format: [num_detections, 84] (x, y, w, h, conf, class_scores...)
                num_detections = total_elements // 84
                output = output.reshape(num_detections, 84)
            elif total_elements % 85 == 0:
                # Format: [num_detections, 85]
                num_detections = total_elements // 85
                output = output.reshape(num_detections, 85)
            else:
                # Try common YOLO shapes
                if total_elements == 25200 * 84:  # [1, 84, 25200]
                    output = output.reshape(84, 25200).transpose(1, 0)
                elif total_elements == 8400 * 84:  # [1, 84, 8400]
                    output = output.reshape(84, 8400).transpose(1, 0)
                else:
                    # Default: assume [num_detections, 84]
                    num_detections = total_elements // 84
                    output = output.reshape(num_detections, 84)
        else:
            # Already shaped
            if len(output_shape) == 3:
                # [batch, channels, detections] -> [detections, channels]
                output = output.reshape(output_shape).transpose(0, 2, 1)[0].transpose(1, 0)
            elif len(output_shape) == 2:
                # [detections, features]
                output = output.reshape(output_shape)
            else:
                output = output.flatten().reshape(-1, 84)
        
        scale_x = original_shape[1] / inference_size[0]
        scale_y = original_shape[0] / inference_size[1]
        
        for detection in output:
            # Parse detection (adjust based on actual YOLO output format)
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
            
            detections.append({
                'class_id': int(class_id),
                'class_name': f'class_{class_id}',  # Will need class names mapping
                'confidence': float(class_confidence),
                'bbox': [float(x1), float(y1), float(x2), float(y2)],
                'camera_id': None
            })
        
        return detections
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        return self.stats.copy()

    def cleanup(self):
        """Clean up CUDA context"""
        if hasattr(self, 'cuda_context') and self.cuda_context is not None:
            try:
                self.cuda_context.pop()
            except:
                pass


class TensorRTConverter:
    """Convert YOLO models to TensorRT engines"""
    
    @staticmethod
    def convert_yolo_to_tensorrt(model_path: str, output_path: str,
                                 input_size: Tuple[int, int] = (640, 480),
                                 precision: str = 'fp16') -> str:
        """
        Convert YOLO model to TensorRT engine
        
        Args:
            model_path: Path to YOLO model (.pt file)
            output_path: Output path for TensorRT engine
            input_size: Input size (width, height)
            precision: Precision mode ('fp32', 'fp16', 'int8')
            
        Returns:
            Path to converted engine file
        """
        try:
            from ultralytics import YOLO
            
            print(f"Loading YOLO model: {model_path}")
            model = YOLO(model_path)
            
            print(f"Exporting to TensorRT format...")
            # Export to TensorRT
            model.export(
                format='engine',
                imgsz=input_size[1],  # Height
                half=(precision == 'fp16'),
                device=0,  # GPU 0
                simplify=True
            )
            
            # Find the exported engine file
            model_dir = Path(model_path).parent
            engine_file = model_dir / f"{Path(model_path).stem}.engine"
            
            if engine_file.exists():
                # Move to output path
                import shutil
                shutil.move(str(engine_file), output_path)
                print(f"✓ TensorRT engine saved: {output_path}")
                return output_path
            else:
                raise FileNotFoundError(f"Engine file not found: {engine_file}")
                
        except Exception as e:
            raise RuntimeError(f"Failed to convert to TensorRT: {e}")
