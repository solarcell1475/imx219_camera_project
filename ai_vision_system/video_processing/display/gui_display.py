#!/usr/bin/env python3
"""
Advanced GUI Display System
============================
Full-featured GUI with control panel, buttons, sliders, and real-time video display.
Uses OpenCV's highgui with trackbars for interactive controls.
"""

import cv2
import numpy as np
import time
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum


class ViewMode(Enum):
    """Display view modes"""
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    SPLIT = "split"
    SINGLE_0 = "single_0"
    SINGLE_1 = "single_1"


class GUIDisplay:
    """Advanced GUI display with control panel"""
    
    def __init__(self, window_name: str = "YOLO Dual Camera Detection",
                 width: int = 1280, height: int = 720):
        """
        Initialize GUI display
        
        Args:
            window_name: Window title
            width: Display width
            height: Display height
        """
        self.window_name = window_name
        self.control_window = "Control Panel"
        self.width = width
        self.height = height
        
        # Display settings
        self.view_mode = ViewMode.SIDE_BY_SIDE
        self.show_detections = True
        self.show_fps = True
        self.show_metrics = True
        self.show_detection_count = True
        
        # Control values
        self.confidence_threshold = 25  # 0-100 scale
        self.iou_threshold = 45  # 0-100 scale
        
        # FPS tracking
        self.fps_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0
        
        # Callbacks
        self.on_confidence_change: Optional[Callable] = None
        self.on_iou_change: Optional[Callable] = None
        self.on_view_mode_change: Optional[Callable] = None
        
        # Create windows
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, width * 2, height)
        
        # Create control panel window
        self._create_control_panel()
    
    def _create_control_panel(self):
        """Create control panel with trackbars"""
        cv2.namedWindow(self.control_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.control_window, 400, 300)
        
        # Create trackbars
        cv2.createTrackbar('Confidence', self.control_window, 25, 100, self._on_confidence_trackbar)
        cv2.createTrackbar('IoU Threshold', self.control_window, 45, 100, self._on_iou_trackbar)
        cv2.createTrackbar('View Mode', self.control_window, 0, 4, self._on_view_mode_trackbar)
        
        # Create control panel image
        self.control_image = np.zeros((300, 400, 3), dtype=np.uint8)
        self._update_control_panel()
    
    def _on_confidence_trackbar(self, val):
        """Handle confidence threshold trackbar change"""
        self.confidence_threshold = val
        if self.on_confidence_change:
            self.on_confidence_change(val / 100.0)
        self._update_control_panel()
    
    def _on_iou_trackbar(self, val):
        """Handle IoU threshold trackbar change"""
        self.iou_threshold = val
        if self.on_iou_change:
            self.on_iou_change(val / 100.0)
        self._update_control_panel()
    
    def _on_view_mode_trackbar(self, val):
        """Handle view mode trackbar change"""
        modes = list(ViewMode)
        if val < len(modes):
            self.view_mode = modes[val]
            if self.on_view_mode_change:
                self.on_view_mode_change(self.view_mode)
        self._update_control_panel()
    
    def _update_control_panel(self):
        """Update control panel display"""
        img = np.zeros((300, 400, 3), dtype=np.uint8)
        
        # Title
        cv2.putText(img, "YOLO Control Panel", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Current settings
        y = 60
        cv2.putText(img, f"Confidence: {self.confidence_threshold}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(img, f"IoU: {self.iou_threshold}%", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(img, f"View: {self.view_mode.value}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 25
        cv2.putText(img, f"FPS: {self.current_fps:.1f}", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Instructions
        y += 40
        cv2.putText(img, "Controls:", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y += 20
        cv2.putText(img, "Q/ESC - Quit", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(img, "S - Save frame", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(img, "D - Toggle detections", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        y += 18
        cv2.putText(img, "F - Toggle FPS", (10, y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        self.control_image = img
        cv2.imshow(self.control_window, self.control_image)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:
            self.current_fps = self.frame_count / elapsed
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            self.frame_count = 0
            self.last_fps_time = current_time
            self._update_control_panel()
    
    def draw_fps(self, image: np.ndarray, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw FPS counter on image"""
        if not self.show_fps:
            return image
        
        result = image.copy()
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        (text_width, text_height), baseline = cv2.getTextSize(
            fps_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(
            result,
            (position[0] - 5, position[1] - text_height - 5),
            (position[0] + text_width + 5, position[1] + 5),
            (0, 0, 0),
            -1
        )
        cv2.putText(
            result,
            fps_text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
        return result
    
    def draw_metrics(self, image: np.ndarray, metrics: Dict,
                    position: Tuple[int, int] = (10, 70)) -> np.ndarray:
        """Draw performance metrics"""
        if not self.show_metrics:
            return image
        
        result = image.copy()
        y_offset = position[1]
        
        lines = []
        if 'inference_time_ms' in metrics:
            lines.append(f"Inference: {metrics['inference_time_ms']:.1f} ms")
        if 'gpu_utilization' in metrics:
            lines.append(f"GPU: {metrics['gpu_utilization']:.1f}%")
        if 'memory_usage_mb' in metrics:
            lines.append(f"Memory: {metrics['memory_usage_mb']:.1f} MB")
        
        if not lines:
            return result
        
        max_width = 0
        total_height = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_width = max(max_width, w)
            total_height += h + 5
        
        cv2.rectangle(
            result,
            (position[0] - 5, position[1] - 5),
            (position[0] + max_width + 10, position[1] + total_height + 5),
            (0, 0, 0),
            -1
        )
        
        for line in lines:
            cv2.putText(
                result,
                line,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1
            )
            y_offset += 20
        
        return result
    
    def create_side_by_side(self, image0: np.ndarray, image1: np.ndarray,
                           detections0: List[Dict], detections1: List[Dict],
                           postprocessor) -> np.ndarray:
        """Create side-by-side view"""
        if self.show_detections:
            display0 = postprocessor.draw_detections(image0, detections0)
            display1 = postprocessor.draw_detections(image1, detections1)
        else:
            display0 = image0.copy()
            display1 = image1.copy()
        
        h0, w0 = display0.shape[:2]
        h1, w1 = display1.shape[:2]
        target_height = max(h0, h1)
        
        if h0 != target_height:
            scale = target_height / h0
            new_w = int(w0 * scale)
            display0 = cv2.resize(display0, (new_w, target_height))
        if h1 != target_height:
            scale = target_height / h1
            new_w = int(w1 * scale)
            display1 = cv2.resize(display1, (new_w, target_height))
        
        combined = np.hstack([display0, display1])
        cv2.putText(combined, "Camera 0", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Camera 1", (display0.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def create_overlay(self, image0: np.ndarray, image1: np.ndarray,
                      detections0: List[Dict], detections1: List[Dict],
                      postprocessor) -> np.ndarray:
        """Create overlay view"""
        result = image0.copy()
        if self.show_detections:
            result = postprocessor.draw_detections(result, detections0)
            result = postprocessor.draw_detections(result, detections1)
        return result
    
    def create_split(self, image0: np.ndarray, image1: np.ndarray,
                    detections0: List[Dict], detections1: List[Dict],
                    postprocessor) -> np.ndarray:
        """Create split view"""
        if self.show_detections:
            display = postprocessor.draw_detections(image0, detections0)
        else:
            display = image0.copy()
        
        h0, w0 = image0.shape[:2]
        h1, w1 = display.shape[:2]
        if (h0, w0) != (h1, w1):
            display = cv2.resize(display, (w0, h0))
        
        combined = np.hstack([image0, display])
        cv2.putText(combined, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Detections", (w0 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def display(self, image0: np.ndarray, image1: np.ndarray,
               detections0: List[Dict], detections1: List[Dict],
               postprocessor, metrics: Optional[Dict] = None):
        """Display frames with detections"""
        self.update_fps()
        
        if self.view_mode == ViewMode.SIDE_BY_SIDE:
            display_image = self.create_side_by_side(
                image0, image1, detections0, detections1, postprocessor
            )
        elif self.view_mode == ViewMode.OVERLAY:
            display_image = self.create_overlay(
                image0, image1, detections0, detections1, postprocessor
            )
        elif self.view_mode == ViewMode.SPLIT:
            display_image = self.create_split(
                image0, image1, detections0, detections1, postprocessor
            )
        elif self.view_mode == ViewMode.SINGLE_0:
            if self.show_detections:
                display_image = postprocessor.draw_detections(image0, detections0)
            else:
                display_image = image0.copy()
        elif self.view_mode == ViewMode.SINGLE_1:
            if self.show_detections:
                display_image = postprocessor.draw_detections(image1, detections1)
            else:
                display_image = image1.copy()
        else:
            display_image = image0.copy()
        
        display_image = self.draw_fps(display_image)
        if metrics:
            display_image = self.draw_metrics(display_image, metrics)
        
        if self.show_detection_count:
            summary0 = postprocessor.get_detection_summary(detections0)
            summary1 = postprocessor.get_detection_summary(detections1)
            total = summary0['total'] + summary1['total']
            cv2.putText(display_image, f"Detections: {total}",
                       (10, display_image.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        cv2.imshow(self.window_name, display_image)
    
    def handle_keyboard(self) -> Optional[str]:
        """Handle keyboard input"""
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:
            return 'quit'
        elif key == ord('s'):
            return 'save'
        elif key == ord('d'):
            self.show_detections = not self.show_detections
            return 'toggle_detections'
        elif key == ord('v'):
            modes = list(ViewMode)
            current_idx = modes.index(self.view_mode)
            self.view_mode = modes[(current_idx + 1) % len(modes)]
            if self.on_view_mode_change:
                self.on_view_mode_change(self.view_mode)
            self._update_control_panel()
            return 'change_view'
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            return 'toggle_fps'
        elif key == ord('m'):
            self.show_metrics = not self.show_metrics
            return 'toggle_metrics'
        
        return None
    
    def cleanup(self):
        """Cleanup display resources"""
        cv2.destroyAllWindows()
