#!/usr/bin/env python3
"""
Asynchronous Processing Pipeline
=================================
Implements asynchronous camera capture and inference processing for maximum performance.
Camera capture and inference run in parallel threads to eliminate bottlenecks.
"""

import time
import threading
import queue
import numpy as np
from typing import Optional, Tuple, List, Dict, Callable
from concurrent.futures import ThreadPoolExecutor
import cv2


class AsyncFrameProcessor:
    """Asynchronous frame capture and inference processor"""

    def __init__(self, camera_capture, inference_engine,
                 max_queue_size: int = 2,
                 target_fps: float = 30.0):
        """
        Initialize async processor

        Args:
            camera_capture: Camera capture object with read() method
            inference_engine: Inference engine with infer_dual() method
            max_queue_size: Maximum frames to buffer
            target_fps: Target processing FPS
        """
        self.camera_capture = camera_capture
        self.inference_engine = inference_engine
        self.max_queue_size = max_queue_size
        self.target_fps = target_fps

        # Frame buffers (producer-consumer pattern)
        self.frame_queue = queue.Queue(maxsize=max_queue_size)
        self.result_queue = queue.Queue(maxsize=max_queue_size)

        # Processing threads
        self.capture_thread: Optional[threading.Thread] = None
        self.inference_thread: Optional[threading.Thread] = None

        # Control flags
        self.running = False
        self.capture_active = False
        self.inference_active = False

        # Statistics
        self.stats = {
            'frames_captured': 0,
            'frames_processed': 0,
            'frames_dropped': 0,
            'avg_capture_fps': 0.0,
            'avg_processing_fps': 0.0,
            'queue_size_avg': 0.0,
            'capture_time': 0.0,
            'inference_time': 0.0
        }

        # Timing
        self.last_capture_time = 0.0
        self.last_process_time = 0.0
        self.frame_interval = 1.0 / target_fps

    def start(self) -> bool:
        """Start asynchronous processing"""
        if self.running:
            return True

        self.running = True

        # Start capture thread
        self.capture_thread = threading.Thread(
            target=self._capture_loop,
            name="CameraCapture",
            daemon=True
        )
        self.capture_thread.start()

        # Start inference thread
        self.inference_thread = threading.Thread(
            target=self._inference_loop,
            name="InferenceProcessor",
            daemon=True
        )
        self.inference_thread.start()

        return True

    def stop(self):
        """Stop asynchronous processing"""
        self.running = False

        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.inference_thread and self.inference_thread.is_alive():
            self.inference_thread.join(timeout=2.0)

        # Clear queues
        while not self.frame_queue.empty():
            try:
                self.frame_queue.get_nowait()
            except queue.Empty:
                break

        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except queue.Empty:
                break

    def get_latest_result(self, timeout: float = 0.1) -> Optional[Tuple[np.ndarray, np.ndarray, List, List]]:
        """
        Get latest processing result

        Returns:
            Tuple of (frame0, frame1, detections0, detections1) or None if no result available
        """
        try:
            return self.result_queue.get(timeout=timeout)
        except queue.Queue.Empty:
            return None

    def get_statistics(self) -> Dict:
        """Get processing statistics"""
        # Update queue statistics
        self.stats['queue_size_avg'] = self.frame_queue.qsize()

        return self.stats.copy()

    def _capture_loop(self):
        """Camera capture loop - runs continuously"""
        self.capture_active = True
        capture_times = []

        while self.running and self.capture_active:
            try:
                start_time = time.time()

                # Capture frame pair
                ret, frame0, frame1 = self.camera_capture.read(timeout=0.1)

                if ret and frame0 is not None and frame1 is not None:
                    # Put frames in queue (non-blocking)
                    try:
                        frame_data = (frame0.copy(), frame1.copy(), time.time())
                        self.frame_queue.put(frame_data, timeout=0.1)
                        self.stats['frames_captured'] += 1
                    except queue.Full:
                        # Drop oldest frame if queue is full
                        try:
                            self.frame_queue.get_nowait()
                            self.frame_queue.put(frame_data, timeout=0.1)
                            self.stats['frames_dropped'] += 1
                        except queue.Empty:
                            pass

                capture_time = (time.time() - start_time) * 1000
                self.stats['capture_time'] = capture_time
                capture_times.append(capture_time)

                # Maintain target FPS by sleeping if needed
                elapsed = time.time() - start_time
                if elapsed < self.frame_interval:
                    time.sleep(self.frame_interval - elapsed)

                # Update FPS statistics
                if len(capture_times) > 10:
                    capture_times.pop(0)
                    self.stats['avg_capture_fps'] = 1000.0 / (sum(capture_times) / len(capture_times))

            except Exception as e:
                print(f"Capture loop error: {e}")
                time.sleep(0.1)

        self.capture_active = False

    def _inference_loop(self):
        """Inference processing loop - processes frames from queue"""
        self.inference_active = True
        process_times = []

        while self.running and self.inference_active:
            try:
                # Get frame from queue
                frame_data = self.frame_queue.get(timeout=0.1)
                frame0, frame1, capture_timestamp = frame_data

                start_time = time.time()

                # Run inference
                detections0, detections1 = self.inference_engine.infer_dual(frame0, frame1)

                process_time = (time.time() - start_time) * 1000
                self.stats['inference_time'] = process_time
                process_times.append(process_time)

                # Put result in output queue
                try:
                    result = (frame0, frame1, detections0, detections1, capture_timestamp, time.time())
                    self.result_queue.put(result, timeout=0.1)
                    self.stats['frames_processed'] += 1
                except queue.Full:
                    # Drop oldest result if queue is full
                    try:
                        self.result_queue.get_nowait()
                        self.result_queue.put(result, timeout=0.1)
                    except queue.Empty:
                        pass

                # Update FPS statistics
                if len(process_times) > 10:
                    process_times.pop(0)
                    self.stats['avg_processing_fps'] = 1000.0 / (sum(process_times) / len(process_times))

            except queue.Empty:
                # No frames available, continue
                time.sleep(0.01)
            except Exception as e:
                print(f"Inference loop error: {e}")
                time.sleep(0.1)

        self.inference_active = False


class AsyncPerformanceOptimizer:
    """Optimizes async processing parameters based on performance"""

    def __init__(self, async_processor: AsyncFrameProcessor):
        self.async_processor = async_processor
        self.optimization_history = []

    def optimize_parameters(self, current_fps: float, target_fps: float) -> Dict:
        """
        Optimize processing parameters based on current performance

        Args:
            current_fps: Current processing FPS
            target_fps: Target FPS

        Returns:
            Dictionary of optimized parameters
        """
        stats = self.async_processor.get_statistics()

        # Analyze bottlenecks
        capture_fps = stats.get('avg_capture_fps', 0)
        processing_fps = stats.get('avg_processing_fps', 0)
        queue_size = stats.get('queue_size_avg', 0)
        dropped_frames = stats.get('frames_dropped', 0)

        # Determine bottleneck
        if processing_fps < capture_fps * 0.8:
            bottleneck = "inference"
        elif capture_fps < target_fps * 0.8:
            bottleneck = "capture"
        else:
            bottleneck = "balanced"

        # Calculate optimal queue size
        if bottleneck == "inference":
            # Increase queue to buffer more frames for inference
            optimal_queue_size = min(4, self.async_processor.max_queue_size + 1)
        elif bottleneck == "capture":
            # Reduce queue to prevent memory buildup
            optimal_queue_size = max(1, self.async_processor.max_queue_size - 1)
        else:
            optimal_queue_size = self.async_processor.max_queue_size

        # Calculate optimal target FPS
        if current_fps > target_fps * 1.2:
            # We're over-performing, can increase target
            optimal_target_fps = min(target_fps * 1.1, 60.0)
        elif current_fps < target_fps * 0.8:
            # We're under-performing, reduce target
            optimal_target_fps = max(target_fps * 0.9, 15.0)
        else:
            optimal_target_fps = target_fps

        optimization = {
            'bottleneck': bottleneck,
            'optimal_queue_size': optimal_queue_size,
            'optimal_target_fps': optimal_target_fps,
            'capture_fps': capture_fps,
            'processing_fps': processing_fps,
            'queue_size': queue_size,
            'dropped_frames': dropped_frames
        }

        self.optimization_history.append(optimization)
        return optimization

    def apply_optimization(self, optimization: Dict):
        """Apply optimization parameters"""
        if optimization['optimal_queue_size'] != self.async_processor.max_queue_size:
            # Reconfigure queue size (would require restart in real implementation)
            print(f"Queue size optimization: {self.async_processor.max_queue_size} -> {optimization['optimal_queue_size']}")

        if abs(optimization['optimal_target_fps'] - self.async_processor.target_fps) > 1.0:
            self.async_processor.target_fps = optimization['optimal_target_fps']
            self.async_processor.frame_interval = 1.0 / self.async_processor.target_fps
            print(f"Target FPS optimization: {self.async_processor.target_fps:.1f} FPS")