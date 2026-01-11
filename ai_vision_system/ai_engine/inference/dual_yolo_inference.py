#!/usr/bin/env python3
"""
Dual-Camera YOLO Inference Pipeline
=====================================
Parallel inference on both camera feeds with result aggregation,
NMS, and confidence threshold management.
"""

import time
import numpy as np
import cv2
from typing import List, Dict, Tuple, Optional
import threading
from queue import Queue
from PIL import Image
import tempfile
import os


class DualYOLOInference:
    """Dual-camera YOLO inference pipeline"""
    
    def __init__(self, yolo_model, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cuda'):
        """
        Initialize dual-camera inference
        
        Args:
            yolo_model: YOLO model object
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            device: Device to use ('cuda' or 'cpu')
        """
        self.model = yolo_model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        
        # Statistics
        self.stats = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'frames_processed': 0
        }
        
        # Class filter (None = all classes)
        self.class_filter = None
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold"""
        self.confidence_threshold = max(0.0, min(1.0, threshold))
    
    def set_iou_threshold(self, threshold: float):
        """Set IoU threshold for NMS"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
    
    def set_class_filter(self, class_ids: Optional[List[int]]):
        """
        Set class filter
        
        Args:
            class_ids: List of class IDs to keep (None = all classes)
        """
        self.class_filter = class_ids
    
    def infer_single(self, image: np.ndarray, inference_size: Tuple[int, int] = (640, 480)) -> List[Dict]:
        """
        Run inference on a single image
        
        Args:
            image: Input image (BGR format)
            inference_size: Target size for inference (width, height) - smaller = faster
        
        Returns:
            List of detection dictionaries
        """
        start_time = time.time()
        
        # Store original size for bbox scaling
        original_height, original_width = image.shape[:2]
        scale_x = original_width / inference_size[0]
        scale_y = original_height / inference_size[1]
        
        # Resize image for faster inference (keep aspect ratio)
        inference_width, inference_height = inference_size
        resized_image = cv2.resize(image, (inference_width, inference_height), interpolation=cv2.INTER_LINEAR)
        
        # Convert BGR to RGB (YOLO expects RGB)
        if len(resized_image.shape) == 3 and resized_image.shape[2] == 3:
            rgb_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = resized_image
        
        # Save to temporary file for YOLO (most reliable method)
        # Use lock if available (for thread-safe access)
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            # Save PIL Image to temp file
            pil_image = Image.fromarray(rgb_image)
            pil_image.save(tmp_path, 'JPEG', quality=95)
        
        try:
            if hasattr(self, 'model_lock'):
                with self.model_lock:
                    results = self.model.predict(
                        source=tmp_path,
                        conf=self.confidence_threshold,
                        iou=self.iou_threshold,
                        device=self.device,
                        verbose=False,
                        imgsz=inference_height  # Use inference height
                    )
            else:
                results = self.model.predict(
                    source=tmp_path,
                    conf=self.confidence_threshold,
                    iou=self.iou_threshold,
                    device=self.device,
                    verbose=False,
                    imgsz=inference_height  # Use inference height
                )
        finally:
            # Clean up temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
        
        inference_time = (time.time() - start_time) * 1000  # Convert to ms
        
        # Parse results - results is a list of Results objects
        detections = []
        if results and len(results) > 0:
            result = results[0]
            
            # Check if result is a Results object with boxes
            # Check if result has boxes attribute (Results object)
            if hasattr(result, 'boxes') and result.boxes is not None:
                boxes = result.boxes
                num_boxes = len(boxes)
                
                if num_boxes > 0:
                    # Get all boxes data at once (more efficient)
                    cls_tensor = boxes.cls  # Shape: [N]
                    conf_tensor = boxes.conf  # Shape: [N]
                    xyxy_tensor = boxes.xyxy  # Shape: [N, 4]
                    
                    # Convert to numpy
                    cls_array = cls_tensor.cpu().numpy() if hasattr(cls_tensor, 'cpu') else np.array(cls_tensor)
                    conf_array = conf_tensor.cpu().numpy() if hasattr(conf_tensor, 'cpu') else np.array(conf_tensor)
                    xyxy_array = xyxy_tensor.cpu().numpy() if hasattr(xyxy_tensor, 'cpu') else np.array(xyxy_tensor)
                    
                    # Process each detection
                    for i in range(num_boxes):
                        class_id = int(cls_array[i])
                        confidence = float(conf_array[i])
                        
                        # Check confidence threshold
                        if confidence < self.confidence_threshold:
                            continue
                        
                        # Apply class filter
                        if self.class_filter is not None and class_id not in self.class_filter:
                            continue
                        
                        # Get bounding box [x1, y1, x2, y2] and scale back to original image size
                        bbox_inference = xyxy_array[i]
                        bbox = [
                            bbox_inference[0] * scale_x,  # x1
                            bbox_inference[1] * scale_y,  # y1
                            bbox_inference[2] * scale_x,  # x2
                            bbox_inference[3] * scale_y   # y2
                        ]
                        
                        detection = {
                            'class_id': class_id,
                            'class_name': self.model.names[class_id],
                            'confidence': confidence,
                            'bbox': bbox,
                            'camera_id': None  # Will be set by caller
                        }
                        detections.append(detection)
        
        # Update statistics
        self.stats['inference_count'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['inference_count']
        )
        
        return detections
    
    def infer_dual(self, image0: np.ndarray, image1: np.ndarray, 
                   inference_size: Tuple[int, int] = (640, 480)) -> Tuple[List[Dict], List[Dict]]:
        """
        Run inference on both camera images
        
        Args:
            image0: First camera image (BGR)
            image1: Second camera image (BGR)
            inference_size: Target size for inference (width, height)
        
        Returns:
            (detections0, detections1) - Detection lists for each camera
        """
        # Run inference on both images sequentially
        detections0 = self.infer_single(image0, inference_size)
        detections1 = self.infer_single(image1, inference_size)
        
        # Mark camera IDs
        for det in detections0:
            det['camera_id'] = 0
        for det in detections1:
            det['camera_id'] = 1
        
        self.stats['frames_processed'] += 1
        
        return detections0, detections1
    
    def infer_batch(self, images: List[np.ndarray]) -> List[List[Dict]]:
        """
        Run batch inference on multiple images
        
        Args:
            images: List of input images
        
        Returns:
            List of detection lists (one per image)
        """
        if len(images) == 0:
            return []
        
        start_time = time.time()
        
        # Run batch inference
        results = self.model(
            images,
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False
        )
        
        inference_time = (time.time() - start_time) * 1000
        
        # Parse results
        all_detections = []
        for result in results:
            detections = []
            boxes = result.boxes
            
            for i in range(len(boxes)):
                box = boxes[i]
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                bbox = box.xyxy[0].cpu().numpy()
                
                # Apply class filter
                if self.class_filter is not None and class_id not in self.class_filter:
                    continue
                
                detection = {
                    'class_id': class_id,
                    'class_name': self.model.names[class_id],
                    'confidence': confidence,
                    'bbox': bbox.tolist(),
                    'camera_id': None
                }
                detections.append(detection)
            
            all_detections.append(detections)
        
        # Update statistics
        self.stats['inference_count'] += len(images)
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['inference_count']
        )
        self.stats['frames_processed'] += len(images)
        
        return all_detections
    
    def merge_detections(self, detections0: List[Dict], detections1: List[Dict],
                        merge_strategy: str = 'union') -> List[Dict]:
        """
        Merge detections from both cameras
        
        Args:
            detections0: Detections from camera 0
            detections1: Detections from camera 1
            merge_strategy: 'union' (all detections) or 'intersection' (only overlapping)
        
        Returns:
            Merged detection list
        """
        if merge_strategy == 'union':
            # Simple union - return all detections
            merged = detections0 + detections1
            return merged
        
        elif merge_strategy == 'intersection':
            # Only keep detections that appear in both cameras
            # This is a simplified version - would need spatial matching
            merged = []
            # For now, just return union (proper intersection requires 3D matching)
            merged = detections0 + detections1
            return merged
        
        else:
            return detections0 + detections1
    
    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        return self.stats.copy()
    
    def reset_statistics(self):
        """Reset statistics"""
        self.stats = {
            'inference_count': 0,
            'total_inference_time': 0.0,
            'avg_inference_time': 0.0,
            'frames_processed': 0
        }


class ParallelDualInference:
    """Parallel inference using threading for true parallel processing"""
    
    def __init__(self, yolo_model, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, device: str = 'cuda',
                 inference_size: Tuple[int, int] = (640, 480)):
        self.model = yolo_model
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device
        self.inference_size = inference_size
        
        # Lock for thread-safe model access (YOLO models may not be fully thread-safe)
        self.model_lock = threading.Lock()
        
        # Create inference instances for each camera
        self.inference0 = DualYOLOInference(
            yolo_model, confidence_threshold, iou_threshold, device
        )
        self.inference1 = DualYOLOInference(
            yolo_model, confidence_threshold, iou_threshold, device
        )
        
        # Add lock to inference instances for thread-safe model access
        self.inference0.model_lock = self.model_lock
        self.inference1.model_lock = self.model_lock
    
    def infer_parallel(self, image0: np.ndarray, image1: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Run parallel inference on both images using threads"""
        results0 = Queue()
        results1 = Queue()
        
        def infer_camera0():
            detections = self.inference0.infer_single(image0, self.inference_size)
            for det in detections:
                det['camera_id'] = 0
            results0.put(detections)
        
        def infer_camera1():
            detections = self.inference1.infer_single(image1, self.inference_size)
            for det in detections:
                det['camera_id'] = 1
            results1.put(detections)
        
        # Run in parallel
        thread0 = threading.Thread(target=infer_camera0, daemon=True)
        thread1 = threading.Thread(target=infer_camera1, daemon=True)
        
        thread0.start()
        thread1.start()
        
        thread0.join()
        thread1.join()
        
        detections0 = results0.get()
        detections1 = results1.get()
        
        return detections0, detections1
    
    def infer_dual(self, image0: np.ndarray, image1: np.ndarray,
                   inference_size: Tuple[int, int] = (640, 480)) -> Tuple[List[Dict], List[Dict]]:
        """
        Run parallel inference on both camera feeds
        
        Args:
            image0: First camera image
            image1: Second camera image
            inference_size: Target size for inference (width, height)
            
        Returns:
            Tuple of (detections0, detections1)
        """
        return self.infer_parallel(image0, image1)
    
    def set_confidence_threshold(self, threshold: float):
        """Set confidence threshold for both inference instances"""
        self.confidence_threshold = threshold
        self.inference0.set_confidence_threshold(threshold)
        self.inference1.set_confidence_threshold(threshold)