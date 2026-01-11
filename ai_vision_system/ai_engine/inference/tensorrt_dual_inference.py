#!/usr/bin/env python3
"""
TensorRT Dual-Camera Inference Pipeline
========================================
GPU-accelerated parallel inference on both camera feeds using TensorRT.
Provides 5-10x faster inference than CPU-based approaches.
"""

import time
import numpy as np
from typing import List, Dict, Tuple, Optional
import threading
from queue import Queue

try:
    from ai_engine.optimization.tensorrt_inference import TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("Warning: TensorRT inference not available")


class TensorRTDualInference:
    """TensorRT-based dual-camera inference with GPU acceleration"""

    def __init__(self, engine_path: str, confidence_threshold: float = 0.25,
                 iou_threshold: float = 0.45, inference_size: Tuple[int, int] = (640, 640)):
        """
        Initialize TensorRT dual-camera inference

        Args:
            engine_path: Path to TensorRT engine file (.engine)
            confidence_threshold: Confidence threshold for detections
            iou_threshold: IoU threshold for NMS
            inference_size: Target size for inference (width, height)
        """
        if not TENSORRT_AVAILABLE:
            raise ImportError("TensorRT inference not available. Install TensorRT and PyCUDA.")

        self.engine_path = engine_path
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.inference_size = inference_size

        # Initialize TensorRT inference engines
        print("Loading TensorRT engines for dual inference...")
        try:
            self.inference0 = TensorRTInference(
                engine_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            self.inference1 = TensorRTInference(
                engine_path,
                confidence_threshold=confidence_threshold,
                iou_threshold=iou_threshold
            )
            print("✓ TensorRT engines loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load TensorRT engines: {e}")

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
        self.inference0.confidence_threshold = self.confidence_threshold
        self.inference1.confidence_threshold = self.confidence_threshold

    def set_iou_threshold(self, threshold: float):
        """Set IoU threshold for NMS"""
        self.iou_threshold = max(0.0, min(1.0, threshold))
        self.inference0.iou_threshold = self.iou_threshold
        self.inference1.iou_threshold = self.iou_threshold

    def set_class_filter(self, class_ids: Optional[List[int]]):
        """
        Set class filter

        Args:
            class_ids: List of class IDs to keep (None = all classes)
        """
        self.class_filter = class_ids

    def infer_parallel(self, image0: np.ndarray, image1: np.ndarray) -> Tuple[List[Dict], List[Dict]]:
        """Run parallel inference on both images using threads"""
        results0 = Queue()
        results1 = Queue()

        def infer_camera0():
            detections = self.inference0.infer(image0, self.inference_size)
            for det in detections:
                det['camera_id'] = 0
            results0.put(detections)

        def infer_camera1():
            detections = self.inference1.infer(image1, self.inference_size)
            for det in detections:
                det['camera_id'] = 1
            results1.put(detections)

        # Start parallel inference
        start_time = time.time()
        thread0 = threading.Thread(target=infer_camera0, daemon=True)
        thread1 = threading.Thread(target=infer_camera1, daemon=True)

        thread0.start()
        thread1.start()

        # Wait for completion with timeout
        thread0.join(timeout=5.0)
        thread1.join(timeout=5.0)

        # Get results
        try:
            detections0 = results0.get(timeout=1.0)
        except:
            detections0 = []

        try:
            detections1 = results1.get(timeout=1.0)
        except:
            detections1 = []

        inference_time = (time.time() - start_time) * 1000

        # Update statistics
        self.stats['inference_count'] += 1
        self.stats['total_inference_time'] += inference_time
        self.stats['avg_inference_time'] = (
            self.stats['total_inference_time'] / self.stats['inference_count']
        )
        self.stats['frames_processed'] += 2

        return detections0, detections1

    def infer_dual(self, image0: np.ndarray, image1: np.ndarray,
                   inference_size: Tuple[int, int] = None) -> Tuple[List[Dict], List[Dict]]:
        """
        Run parallel inference on both camera feeds

        Args:
            image0: First camera image
            image1: Second camera image
            inference_size: Target size for inference (width, height)

        Returns:
            Tuple of (detections0, detections1)
        """
        if inference_size is None:
            inference_size = self.inference_size

        # Use parallel inference
        detections0, detections1 = self.infer_parallel(image0, image1)

        return detections0, detections1

    def infer_single(self, image: np.ndarray, inference_size: Tuple[int, int] = None) -> List[Dict]:
        """
        Run inference on a single image

        Args:
            image: Input image (BGR format)
            inference_size: Target size for inference (width, height)

        Returns:
            List of detection dictionaries
        """
        if inference_size is None:
            inference_size = self.inference_size

        # Use inference0 for single image
        detections = self.inference0.infer(image, inference_size)

        # Update statistics
        self.stats['frames_processed'] += 1

        return detections

    def get_statistics(self) -> Dict:
        """Get inference statistics"""
        return self.stats.copy()

    def cleanup(self):
        """Clean up resources"""
        # TensorRT engines don't need explicit cleanup
        pass