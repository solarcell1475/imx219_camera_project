#!/usr/bin/env python3
"""
YOLO AI Vision System - Main Application
==========================================
Main orchestrator integrating all modules for real-time dual-camera
object detection with YOLO.
"""

import sys
import time
import argparse
import yaml
import numpy as np
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Apply PyTorch patch for YOLO compatibility
try:
    from utils.pytorch_patch import patch_torch_load
    patch_torch_load()
except:
    pass

# Setup display environment
try:
    from utils.display_fix import setup_display_environment
    setup_display_environment()
except:
    pass

# Import modules
from firmware.camera_init.dual_camera import DualCamera, ResolutionPreset
from video_processing.capture.gstreamer_capture import GStreamerDualCapture
from video_processing.capture.nvargus_capture import NVArgusCapture
from ai_engine.model_manager.yolo_model_loader import YOLOModelManager
from ai_engine.inference.dual_yolo_inference import DualYOLOInference, ParallelDualInference
try:
    from ai_engine.inference.tensorrt_dual_inference import TensorRTDualInference
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
from video_processing.postprocessing.yolo_postprocess import YOLOPostprocessor
from video_processing.display.realtime_display import RealtimeDisplay, ViewMode
from monitoring.metrics.performance_metrics import PerformanceMetrics
from monitoring.dashboard.performance_dashboard import PerformanceDashboard
from monitoring.logging.system_logger import SystemLogger
from monitoring.alerts.performance_alerts import PerformanceAlerts
from monitoring.power_optimization import PerformanceBalancer
from monitoring.detection_stats import DetectionStatistics
try:
    from ai_engine.processing.async_processor import AsyncFrameProcessor, AsyncPerformanceOptimizer
    ASYNC_AVAILABLE = True
except ImportError:
    ASYNC_AVAILABLE = False


class YOLOVisionSystem:
    """Main YOLO vision system orchestrator"""
    
    def __init__(self, config_path: str = None):
        """
        Initialize system
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Initialize components
        self.camera_capture = None
        self.model_manager = None
        self.yolo_model = None
        self.inference = None
        self.postprocessor = None
        self.display = None
        self.metrics = None
        self.dashboard = None
        self.logger = None
        self.alerts = None
        self.balancer = None
        self.detection_stats = None  # Version 2: Detection statistics tracker
        self.async_processor = None  # Async processing for maximum performance
        self.async_optimizer = None  # Performance optimizer

        self.running = False
        self.confidence_threshold = self.config.get('confidence_threshold', 0.25)
    
    def _find_tensorrt_engine(self) -> Optional[str]:
        """Find available TensorRT engine for current model configuration"""
        model_config = self.config.get('model', {})
        version = model_config.get('version', 'v8')
        size = model_config.get('size', 'n')
        engine_suffix = model_config.get('engine_suffix', '')

        # Try with configured suffix first (for optimized engines like _320)
        if version == 'v11':
            engine_path = f"yolo11{size}{engine_suffix}.engine"
            if Path(engine_path).exists():
                return engine_path
            # Fallback to default
            engine_path = f"yolo11{size}.engine"
            if Path(engine_path).exists():
                return engine_path

        # Try YOLOv8 as fallback
        engine_path = f"yolov8{size}{engine_suffix}.engine"
        if Path(engine_path).exists():
            return engine_path
        engine_path = f"yolov8{size}.engine"
        if Path(engine_path).exists():
            return engine_path

        return None

    def _load_config(self, config_path: str = None) -> dict:
        """Load configuration from file"""
        # Check CUDA availability
        try:
            import torch
            cuda_available = torch.cuda.is_available()
        except:
            cuda_available = False
        
        default_config = {
            'model': {'version': 'v11', 'size': 'n'},
            'resolution': {'width': 1280, 'height': 720, 'fps': 30},
            'inference_resolution': [640, 640],
            'confidence_threshold': 0.25,
            'iou_threshold': 0.45,
            'device': 'cuda' if cuda_available else 'cpu',
            'view_mode': 'side_by_side',
            'target_fps': 25.0,
            'use_parallel_inference': True
        }
        
        if config_path and Path(config_path).exists():
            try:
                with open(config_path, 'r') as f:
                    file_config = yaml.safe_load(f)
                    default_config.update(file_config)
                    # Override device if CUDA not available
                    if default_config.get('device', 'cuda') == 'cuda' and not cuda_available:
                        print("⚠ CUDA not available, using CPU instead")
                        default_config['device'] = 'cpu'
            except Exception as e:
                print(f"Warning: Could not load config file: {e}")
        else:
            # If no config file, use CPU if CUDA not available
            if not cuda_available:
                print("⚠ CUDA not available, using CPU instead")
                default_config['device'] = 'cpu'
        
        return default_config
    
    def initialize(self):
        """Initialize all system components"""
        print("Initializing YOLO AI Vision System...")
        print("=" * 70)
        
        # Initialize logger
        self.logger = SystemLogger()
        self.logger.log_info("System initialization started")
        
        # Initialize camera capture
        print("Initializing cameras...")
        res_config = self.config['resolution']
        
        # Try NVArgus capture first (subprocess-based, matches nvgstcapture)
        print("  Attempting NVArgus subprocess capture (matches nvgstcapture)...")
        self.camera_capture = NVArgusCapture(
            width=res_config['width'],
            height=res_config['height'],
            fps=res_config['fps']
        )
        
        if not self.camera_capture.start():
            print("  NVArgus capture failed, trying OpenCV GStreamer...")
            self.camera_capture = GStreamerDualCapture(
                width=res_config['width'],
                height=res_config['height'],
                fps=res_config['fps']
            )
            
            if not self.camera_capture.start():
                print("⚠ Warning: Failed to initialize cameras")
                print("  The program will continue but cameras may not work.")
                print("  To fix: Run 'sudo ./activate_imx219_cameras.sh' and reboot")
                self.logger.log_warning("Camera initialization failed - continuing with fallback")
                # Don't return False - allow demo mode to continue
                # return False
        
        print("✓ Camera capture initialized")
        
        # Initialize model manager
        print("Loading YOLO model...")
        self.model_manager = YOLOModelManager()
        model_config = self.config['model']
        self.yolo_model = self.model_manager.load_model(
            version=model_config['version'],
            size=model_config['size'],
            device=self.config['device']
        )
        print("✓ Model loaded")
        
        # Initialize parallel inference for better performance
        inference_resolution = self.config.get('inference_resolution', [640, 640])
        inference_size = tuple(inference_resolution) if isinstance(inference_resolution, list) else inference_resolution

        # Try TensorRT inference first (GPU acceleration)
        tensorrt_engine = self._find_tensorrt_engine()
        if tensorrt_engine and TENSORRT_AVAILABLE:
            try:
                print(f"Initializing TensorRT inference with: {tensorrt_engine}")
                self.inference = TensorRTDualInference(
                    tensorrt_engine,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.config['iou_threshold'],
                    inference_size=inference_size
                )
                print("✓ TensorRT GPU acceleration enabled!")
            except Exception as e:
                print(f"⚠ TensorRT initialization failed: {e}")
                print("  Falling back to PyTorch inference...")
                tensorrt_engine = None

        if not tensorrt_engine or not TENSORRT_AVAILABLE:
            # Fallback to PyTorch inference
            if self.config.get('use_parallel_inference', True):
                print("Initializing PyTorch parallel inference (CPU)...")
                self.inference = ParallelDualInference(
                    self.yolo_model,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.config['iou_threshold'],
                    device=self.config['device'],
                    inference_size=inference_size
                )
            else:
                # Fallback to sequential inference
                print("Initializing PyTorch sequential inference (CPU)...")
                self.inference = DualYOLOInference(
                    self.yolo_model,
                    confidence_threshold=self.confidence_threshold,
                    iou_threshold=self.config['iou_threshold'],
                    device=self.config['device']
                )

        # Initialize async processor for maximum performance
        if ASYNC_AVAILABLE and self.config.get('use_async_processing', True):
            try:
                target_fps = self.config.get('target_fps', 30.0)
                self.async_processor = AsyncFrameProcessor(
                    camera_capture=None,  # Will be set after camera initialization
                    inference_engine=self.inference,
                    max_queue_size=2,
                    target_fps=target_fps
                )
                self.async_optimizer = AsyncPerformanceOptimizer(self.async_processor)
                print("✓ Async processor initialized for maximum performance")
            except Exception as e:
                print(f"⚠ Async processor initialization failed: {e}")
                self.async_processor = None
                self.async_optimizer = None
        else:
            print("Async processing not available or disabled")

        # Initialize postprocessor
        self.postprocessor = YOLOPostprocessor()
        
        # Initialize display
        self.display = RealtimeDisplay()
        view_mode_str = self.config.get('view_mode', 'side_by_side')
        if view_mode_str == 'side_by_side':
            self.display.view_mode = ViewMode.SIDE_BY_SIDE
        elif view_mode_str == 'overlay':
            self.display.view_mode = ViewMode.OVERLAY
        
        # Initialize monitoring
        self.metrics = PerformanceMetrics()
        self.dashboard = PerformanceDashboard()
        self.alerts = PerformanceAlerts()
        
        # Version 2: Initialize detection statistics tracker
        self.detection_stats = DetectionStatistics(history_size=1000)
        
        # Initialize performance balancer
        self.balancer = PerformanceBalancer(
            target_fps=self.config.get('target_fps', 25.0)
        )
        
        # Setup alert callback
        def alert_callback(alert):
            self.logger.log_warning(f"Alert: {alert['message']}", **alert['data'])
        
        self.alerts.register_callback(alert_callback)
        
        self.logger.log_info("System initialization complete")
        print("✓ System initialized")
        return True
    
    def run(self):
        """Run main loop"""
        if not self.initialize():
            return
        
        print("\n" + "=" * 70)
        print("Starting real-time detection...")
        print("Controls:")
        print("  Q/ESC - Quit")
        print("  S - Save frame")
        print("  D - Toggle detections")
        print("  V - Change view mode")
        print("  +/- - Adjust confidence")
        print("  T - Toggle stats table (Version 2)")
        print("  1/2/3 - Sort stats by appearance/confidence/total (Version 2)")
        print("  R - Reset statistics")
        print("=" * 70)
        
        self.running = True
        frame_count = 0
        start_time = time.time()
        
        try:
            # Start async processor if available
            if self.async_processor and self.camera_capture:
                self.async_processor.camera_capture = self.camera_capture
                self.async_processor.start()
                print("✓ Async processing started - camera capture and inference now run in parallel")

            while self.running:
                # Try async processing first if available
                if self.async_processor and frame_count > 0:  # Let sync processing handle first frame
                    result = self.async_processor.get_latest_result(timeout=0.01)
                    if result:
                        frame0, frame1, detections0, detections1, capture_time, process_time = result
                        ret = True
                        # Skip the synchronous processing below
                        frame_count += 1
                        # Continue with the rest of the processing loop
                    else:
                        ret = False
                else:
                    ret, frame0, frame1 = self.camera_capture.read(timeout=2.0)
                    
                    # Print frame info on first successful read
                    if ret and frame0 is not None and frame1 is not None and frame_count == 0:
                        print(f"✓ First frames captured: Camera 0: {frame0.shape}, Camera 1: {frame1.shape}")
                        print(f"  Frame 0 stats: mean={np.mean(frame0):.2f}, std={np.std(frame0):.2f}")
                        print(f"  Frame 1 stats: mean={np.mean(frame1):.2f}, std={np.std(frame1):.2f}")
                    
                    if not ret or frame0 is None or frame1 is None:
                        # Create placeholder frames if capture fails
                        frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame0, "Camera 0: Not Available", (10, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
                        cv2.putText(frame1, "Camera 1: Not Available", (10, 240), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        detections0 = []
                        detections1 = []
                        # Still display to show the error message
                    else:
                        # Validate frames are not empty
                        if frame0.size == 0 or frame1.size == 0:
                            frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame0, "Camera 0: Empty Frame", (10, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
                            cv2.putText(frame1, "Camera 1: Empty Frame", (10, 240), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            detections0 = []
                            detections1 = []
                            ret = False
                else:
                    # Cameras not initialized - use placeholders
                    frame0 = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame0, "Camera 0: Not Initialized", (10, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    frame1 = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(frame1, "Camera 1: Not Initialized", (10, 240), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    detections0 = []
                    detections1 = []
                    ret = True  # Continue anyway
                
                if not ret:
                    # Still display error frames
                    self.display.display(
                        frame0, frame1,
                        detections0, detections1,
                        self.postprocessor,
                        metrics if 'metrics' in locals() else None,
                        detection_stats=self.detection_stats if hasattr(self, 'detection_stats') else None
                    )
                    continue
                
                # Version 2: Optimized inference with better frame skipping
                # Calculate optimal skip rate based on target FPS and current performance
                target_fps = self.config.get('target_fps', 25.0)
                frame_time_target = 1000.0 / target_fps  # ms per frame for target FPS
                
                # Adaptive frame skipping: skip more frames if inference is slow
                if 'inference_time' in locals() and inference_time > 0:
                    # Calculate how many frames to skip to maintain target FPS
                    skip_frames = max(1, int(inference_time / frame_time_target))
                else:
                    skip_frames = 1
                
                # Version 2: Only run inference every N frames (more aggressive skipping)
                if frame_count % skip_frames == 0:
                    inference_start = time.time()
                    try:
                        # Use parallel or sequential inference based on config
                        if isinstance(self.inference, ParallelDualInference):
                            detections0, detections1 = self.inference.infer_parallel(frame0, frame1)
                        else:
                            detections0, detections1 = self.inference.infer_dual(frame0, frame1)
                    except Exception as e:
                        self.logger.log_error(f"Inference error: {e}", exc_info=True)
                        detections0, detections1 = [], []  # Continue with empty detections
                    inference_time = (time.time() - inference_start) * 1000
                    
                    # Version 2: Update detection statistics
                    all_detections = detections0 + detections1
                    self.detection_stats.update(all_detections, time.time())
                else:
                    # Skip inference on this frame, reuse previous detections
                    detections0 = detections0 if 'detections0' in locals() else []
                    detections1 = detections1 if 'detections1' in locals() else []
                    inference_time = 0
                
                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Collect metrics
                metrics = self.metrics.collect_all_metrics()
                self.metrics.add_inference_metrics(inference_time, fps)
                metrics.update({
                    'inference_time_ms': inference_time,
                    'fps': fps
                })
                
                # Check alerts
                self.alerts.check_metrics(metrics)
                
                # Update dashboard
                if self.dashboard.should_update():
                    self.dashboard.update(metrics)
                    self.dashboard.print_dashboard()
                
                # Display (Version 2: Include detection statistics)
                self.display.display(
                    frame0, frame1,
                    detections0, detections1,
                    self.postprocessor,
                    metrics,
                    detection_stats=self.detection_stats  # Version 2: Pass stats for table display
                )
                
                # Handle keyboard
                action = self.display.handle_keyboard()
                if action == 'quit':
                    break
                elif action == 'increase_confidence':
                    self.confidence_threshold = min(1.0, self.confidence_threshold + 0.05)
                    self.inference.set_confidence_threshold(self.confidence_threshold)
                    self.logger.log_info(f"Confidence threshold increased to {self.confidence_threshold:.2f}")
                elif action == 'decrease_confidence':
                    self.confidence_threshold = max(0.0, self.confidence_threshold - 0.05)
                    self.inference.set_confidence_threshold(self.confidence_threshold)
                    self.logger.log_info(f"Confidence threshold decreased to {self.confidence_threshold:.2f}")
                elif action == 'save':
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    cv2.imwrite(f"frame_{timestamp}_0.jpg", frame0)
                    cv2.imwrite(f"frame_{timestamp}_1.jpg", frame1)
                    print(f"Frames saved: frame_{timestamp}_*.jpg")
                elif action == 'reset':
                    # Reset statistics
                    self.metrics = PerformanceMetrics()
                    self.dashboard = PerformanceDashboard()
                    # Version 2: Reset detection statistics
                    if hasattr(self, 'detection_stats'):
                        self.detection_stats.reset()
                    print("Statistics reset")
                elif action == 'toggle_stats_table':
                    print(f"Stats table: {'ON' if self.display.show_stats_table else 'OFF'}")
                elif action in ['sort_appearance', 'sort_confidence', 'sort_total']:
                    print(f"Sort mode: {self.display.stats_sort_mode}")
                
                # Log performance periodically
                if frame_count % 100 == 0:
                    self.logger.log_performance(metrics)
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        except Exception as e:
            self.logger.log_error("Runtime error", exc_info=True)
            print(f"Error: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Cleanup resources"""
        print("\nCleaning up...")
        if self.camera_capture:
            self.camera_capture.stop()
        if self.display:
            self.display.cleanup()
        if self.logger:
            self.logger.log_info("System shutdown")
        print("✓ Cleanup complete")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="YOLO AI Vision System")
    parser.add_argument('--config', type=str, help='Configuration file path')
    parser.add_argument('--model', type=str, default=None, help='Model (e.g., v8s, v8n)')
    parser.add_argument('--confidence', type=float, default=None, help='Confidence threshold')
    parser.add_argument('--resolution', type=str, default=None, help='Resolution (WxH)')
    
    args = parser.parse_args()
    
    # Create system
    system = YOLOVisionSystem(config_path=args.config)
    
    # Override config with command line args (only if provided)
    if args.model:
        # Parse model string like "v8s" or "v8n"
        model_str = args.model.strip()
        if model_str.startswith('v') and len(model_str) >= 3:
            # Extract version number and size: "v8s" -> version="v8", size="s"
            version_num = model_str[1:-1]  # Everything between 'v' and last char
            size = model_str[-1]  # Last character
            system.config['model'] = {'version': f'v{version_num}', 'size': size}
        else:
            print(f"Warning: Invalid model format '{args.model}', using config file default")
    
    if args.confidence is not None:
        system.config['confidence_threshold'] = args.confidence
    
    if args.resolution:
        w, h = map(int, args.resolution.split('x'))
        system.config['resolution'] = {'width': w, 'height': h, 'fps': 30}
    
    # Run system
    system.run()
    
    return 0


if __name__ == "__main__":
    import cv2
    sys.exit(main())
