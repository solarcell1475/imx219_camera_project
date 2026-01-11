#!/usr/bin/env python3
"""
Real-time Display System
=========================
OpenCV-based display with YOLO visualization, multiple view modes,
FPS counter, performance metrics overlay, and keyboard controls.
"""

import cv2
import numpy as np
import time
import sys
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from enum import Enum
from monitoring.detection_stats import ObjectStats

# Add utils to path for display fixes
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
try:
    from utils.display_fix import setup_display_environment, validate_image_for_display, imshow_safe
    setup_display_environment()
except:
    def validate_image_for_display(img):
        return img
    def imshow_safe(win_name, img):
        cv2.imshow(win_name, img)


class ViewMode(Enum):
    """Display view modes"""
    SIDE_BY_SIDE = "side_by_side"
    OVERLAY = "overlay"
    SPLIT = "split"
    SINGLE_0 = "single_0"
    SINGLE_1 = "single_1"


class RealtimeDisplay:
    """Real-time display system for YOLO detections"""
    
    def __init__(self, window_name: str = "YOLO Dual Camera Detection",
                 width: int = 1280, height: int = 720):
        """
        Initialize display system
        
        Args:
            window_name: Window title
            width: Display width
            height: Display height
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        
        self.view_mode = ViewMode.SIDE_BY_SIDE
        self.show_detections = True
        self.show_fps = True
        self.show_metrics = True
        self.show_detection_count = True
        self.show_stats_table = True  # Version 2: Show detection statistics table
        self.stats_sort_mode = 'appearance'  # 'appearance', 'confidence', 'total'
        
        # FPS tracking
        self.fps_history = []
        self.last_fps_time = time.time()
        self.frame_count = 0
        self.current_fps = 0.0
        
        # Keyboard state
        self.keys_pressed = set()
        
        # Create window with proper backend
        # Try to set OpenCV to use a compatible backend
        try:
            # Set environment for better display compatibility
            import os
            if 'QT_QPA_PLATFORM' not in os.environ:
                os.environ['QT_QPA_PLATFORM'] = 'xcb'
        except:
            pass
        
        # Create window
        try:
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(self.window_name, width * 2, height)
        except Exception as e:
            print(f"Warning: Window creation issue: {e}")
            # Try alternative
            cv2.namedWindow(self.window_name, cv2.WINDOW_AUTOSIZE)
    
    def update_fps(self):
        """Update FPS calculation"""
        self.frame_count += 1
        current_time = time.time()
        elapsed = current_time - self.last_fps_time
        
        if elapsed >= 1.0:  # Update every second
            self.current_fps = self.frame_count / elapsed
            self.fps_history.append(self.current_fps)
            if len(self.fps_history) > 30:
                self.fps_history.pop(0)
            
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def draw_fps(self, image: np.ndarray, position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """Draw FPS counter on image"""
        if not self.show_fps:
            return image
        
        result = image.copy()
        fps_text = f"FPS: {self.current_fps:.1f}"
        
        # Draw background
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
        
        # Draw text
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
        
        # Draw background
        lines = []
        if 'inference_time_ms' in metrics:
            lines.append(f"Inference: {metrics['inference_time_ms']:.1f} ms")
        if 'gpu_utilization' in metrics:
            lines.append(f"GPU: {metrics['gpu_utilization']:.1f}%")
        if 'memory_usage_mb' in metrics:
            lines.append(f"Memory: {metrics['memory_usage_mb']:.1f} MB")
        
        if not lines:
            return result
        
        # Calculate text area
        max_width = 0
        total_height = 0
        for line in lines:
            (w, h), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            max_width = max(max_width, w)
            total_height += h + 5
        
        # Draw background
        cv2.rectangle(
            result,
            (position[0] - 5, position[1] - 5),
            (position[0] + max_width + 10, position[1] + total_height + 5),
            (0, 0, 0),
            -1
        )
        
        # Draw text
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
        # Validate input images
        if image0 is None or image0.size == 0:
            image0 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image0, "Camera 0: No Image", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if image1 is None or image1.size == 0:
            image1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image1, "Camera 1: No Image", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ensure images are uint8
        if image0.dtype != np.uint8:
            image0 = np.clip(image0, 0, 255).astype(np.uint8)
        if image1.dtype != np.uint8:
            image1 = np.clip(image1, 0, 255).astype(np.uint8)
        
        # Process both images
        if self.show_detections:
            display0 = postprocessor.draw_detections(image0, detections0)
            display1 = postprocessor.draw_detections(image1, detections1)
        else:
            display0 = image0.copy()
            display1 = image1.copy()
        
        # Resize to same height if needed
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
        
        # Concatenate horizontally
        combined = np.hstack([display0, display1])
        
        # Add labels
        cv2.putText(combined, "Camera 0", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Camera 1", (display0.shape[1] + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def create_overlay(self, image0: np.ndarray, image1: np.ndarray,
                      detections0: List[Dict], detections1: List[Dict],
                      postprocessor) -> np.ndarray:
        """Create overlay view (both detections on single image)"""
        # Use camera 0 as base
        result = image0.copy()
        
        if self.show_detections:
            # Draw detections from both cameras
            result = postprocessor.draw_detections(result, detections0)
            result = postprocessor.draw_detections(result, detections1)
        
        return result
    
    def create_split(self, image0: np.ndarray, image1: np.ndarray,
                    detections0: List[Dict], detections1: List[Dict],
                    postprocessor) -> np.ndarray:
        """Create split view (original + detections)"""
        # Left: original, Right: with detections
        if self.show_detections:
            display = postprocessor.draw_detections(image0, detections0)
        else:
            display = image0.copy()
        
        # Resize to same dimensions
        h0, w0 = image0.shape[:2]
        h1, w1 = display.shape[:2]
        
        if (h0, w0) != (h1, w1):
            display = cv2.resize(display, (w0, h0))
        
        # Concatenate
        combined = np.hstack([image0, display])
        
        cv2.putText(combined, "Original", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(combined, "Detections", (w0 + 10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return combined
    
    def display(self, image0: np.ndarray, image1: np.ndarray,
               detections0: List[Dict], detections1: List[Dict],
               postprocessor, metrics: Optional[Dict] = None,
               detection_stats = None):
        """
        Display frames with detections (Version 2: Added detection_stats)
        
        Args:
            image0: First camera image
            image1: Second camera image
            detections0: Detections from camera 0
            detections1: Detections from camera 1
            postprocessor: YOLO postprocessor instance
            metrics: Optional performance metrics
            detection_stats: Optional DetectionStatistics instance (Version 2)
        """
        self.update_fps()
        
        # Validate input images first
        if image0 is None or not isinstance(image0, np.ndarray) or image0.size == 0:
            image0 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image0, "Camera 0: No Data", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if image1 is None or not isinstance(image1, np.ndarray) or image1.size == 0:
            image1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image1, "Camera 1: No Data", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Ensure images are proper format and size
        # Resize if too large (max 1920x1080 for display)
        max_display_width = 1920
        max_display_height = 1080
        
        if image0.shape[1] > max_display_width or image0.shape[0] > max_display_height:
            scale = min(max_display_width / image0.shape[1], max_display_height / image0.shape[0])
            new_w = int(image0.shape[1] * scale)
            new_h = int(image0.shape[0] * scale)
            image0 = cv2.resize(image0, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        if image1.shape[1] > max_display_width or image1.shape[0] > max_display_height:
            scale = min(max_display_width / image1.shape[1], max_display_height / image1.shape[0])
            new_w = int(image1.shape[1] * scale)
            new_h = int(image1.shape[0] * scale)
            image1 = cv2.resize(image1, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        # Ensure uint8 format
        if image0.dtype != np.uint8:
            image0 = np.clip(image0, 0, 255).astype(np.uint8)
        if image1.dtype != np.uint8:
            image1 = np.clip(image1, 0, 255).astype(np.uint8)
        
        # Validate frame content (check if frame is not all zeros or uniform)
        if np.std(image0) < 5 or np.mean(image0) < 1:
            # Frame appears to be invalid (too uniform or empty)
            image0 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image0, "Camera 0: Invalid Frame", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        if np.std(image1) < 5 or np.mean(image1) < 1:
            # Frame appears to be invalid
            image1 = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(image1, "Camera 1: Invalid Frame", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Create display based on view mode
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
        
        # Add overlays
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
        
        # Version 2: Draw detection statistics table
        if self.show_stats_table and detection_stats is not None:
            display_image = self.draw_stats_table(display_image, detection_stats)
        
        # Show image - use safe display function
        display_image = validate_image_for_display(display_image)
        if not imshow_safe(self.window_name, display_image):
            print("Warning: Failed to display image")
    
    def handle_keyboard(self) -> Optional[str]:
        """
        Handle keyboard input
        
        Returns:
            Action string or None
        """
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # Q or ESC
            return 'quit'
        elif key == ord('s'):
            return 'save'
        elif key == ord('d'):
            self.show_detections = not self.show_detections
            return 'toggle_detections'
        elif key == ord('v'):
            # Cycle view modes
            modes = list(ViewMode)
            current_idx = modes.index(self.view_mode)
            self.view_mode = modes[(current_idx + 1) % len(modes)]
            return 'change_view'
        elif key == ord('f'):
            self.show_fps = not self.show_fps
            return 'toggle_fps'
        elif key == ord('m'):
            self.show_metrics = not self.show_metrics
            return 'toggle_metrics'
        elif key == ord('+') or key == ord('='):
            return 'increase_confidence'
        elif key == ord('-') or key == ord('_'):
            return 'decrease_confidence'
        elif key == ord('r'):
            return 'reset'
        elif key == ord('t'):
            self.show_stats_table = not self.show_stats_table
            return 'toggle_stats_table'
        elif key == ord('1'):
            self.stats_sort_mode = 'appearance'
            return 'sort_appearance'
        elif key == ord('2'):
            self.stats_sort_mode = 'confidence'
            return 'sort_confidence'
        elif key == ord('3'):
            self.stats_sort_mode = 'total'
            return 'sort_total'
        
        return None
    
    def draw_stats_table(self, image: np.ndarray, detection_stats, max_rows: int = 10) -> np.ndarray:
        """
        Draw detection statistics table (Version 2)
        
        Args:
            image: Input image
            detection_stats: DetectionStatistics instance
            max_rows: Maximum number of rows to display
            
        Returns:
            Image with table overlay
        """
        if not self.show_stats_table:
            return image
        
        # Get sorted statistics based on current sort mode
        if self.stats_sort_mode == 'appearance':
            stats_list = detection_stats.get_stats_sorted_by_appearance(min_appearances=1)
        elif self.stats_sort_mode == 'confidence':
            stats_list = detection_stats.get_stats_sorted_by_confidence(min_appearances=1)
        else:  # 'total'
            stats_list = detection_stats.get_stats_sorted_by_total(min_appearances=1)
        
        if not stats_list:
            return image
        
        # Limit rows
        stats_list = stats_list[:max_rows]
        
        # Table parameters
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        cell_padding = 5
        row_height = 25
        header_height = 30
        
        # Column widths
        col_widths = {
            'class': 120,
            'appearance': 100,
            'confidence': 100,
            'total': 80
        }
        
        table_width = sum(col_widths.values()) + cell_padding * 5
        table_height = header_height + len(stats_list) * row_height + cell_padding * 2
        
        # Position (top-right corner)
        start_x = image.shape[1] - table_width - 10
        start_y = 10
        
        # Create overlay
        overlay = image.copy()
        
        # Draw background
        cv2.rectangle(
            overlay,
            (start_x - 5, start_y - 5),
            (start_x + table_width + 5, start_y + table_height + 5),
            (0, 0, 0),
            -1
        )
        cv2.rectangle(
            overlay,
            (start_x - 5, start_y - 5),
            (start_x + table_width + 5, start_y + table_height + 5),
            (100, 100, 100),
            2
        )
        
        # Draw header
        header_y = start_y + header_height - 5
        x = start_x + cell_padding
        
        # Header text
        headers = ['Class', 'App.Rate', 'Avg.Conf', 'Total']
        header_cols = ['class', 'appearance', 'confidence', 'total']
        
        for i, (header, col_key) in enumerate(zip(headers, header_cols)):
            # Highlight current sort column
            if (self.stats_sort_mode == 'appearance' and col_key == 'appearance') or \
               (self.stats_sort_mode == 'confidence' and col_key == 'confidence') or \
               (self.stats_sort_mode == 'total' and col_key == 'total'):
                color = (0, 255, 255)  # Yellow for sorted column
            else:
                color = (255, 255, 255)  # White
            
            cv2.putText(
                overlay,
                header,
                (x, header_y),
                font,
                font_scale,
                color,
                font_thickness + 1
            )
            x += col_widths[col_key] + cell_padding
        
        # Draw separator line
        cv2.line(
            overlay,
            (start_x, start_y + header_height),
            (start_x + table_width, start_y + header_height),
            (100, 100, 100),
            1
        )
        
        # Draw data rows
        y = start_y + header_height + row_height - 5
        for i, stat in enumerate(stats_list):
            x = start_x + cell_padding
            
            # Alternate row colors
            if i % 2 == 0:
                cv2.rectangle(
                    overlay,
                    (start_x, y - row_height + 5),
                    (start_x + table_width, y + 5),
                    (20, 20, 20),
                    -1
                )
            
            # Class name
            class_name = stat.class_name[:15]  # Truncate if too long
            cv2.putText(
                overlay,
                class_name,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
            x += col_widths['class'] + cell_padding
            
            # Appearance rate
            app_rate_text = f"{stat.appearance_rate:.1%}"
            cv2.putText(
                overlay,
                app_rate_text,
                (x, y),
                font,
                font_scale,
                (0, 255, 0),
                font_thickness
            )
            x += col_widths['appearance'] + cell_padding
            
            # Average confidence
            conf_text = f"{stat.avg_confidence:.2f}"
            # Color based on confidence
            if stat.avg_confidence >= 0.7:
                conf_color = (0, 255, 0)  # Green
            elif stat.avg_confidence >= 0.5:
                conf_color = (0, 255, 255)  # Yellow
            else:
                conf_color = (0, 165, 255)  # Orange
            cv2.putText(
                overlay,
                conf_text,
                (x, y),
                font,
                font_scale,
                conf_color,
                font_thickness
            )
            x += col_widths['confidence'] + cell_padding
            
            # Total detections
            total_text = str(stat.total_detections)
            cv2.putText(
                overlay,
                total_text,
                (x, y),
                font,
                font_scale,
                (255, 255, 255),
                font_thickness
            )
            
            y += row_height
        
        # Draw sort mode indicator
        sort_text = f"Sort: {self.stats_sort_mode} (1/2/3)"
        cv2.putText(
            overlay,
            sort_text,
            (start_x + cell_padding, start_y + table_height + 20),
            font,
            0.4,
            (150, 150, 150),
            1
        )
        
        # Blend overlay
        result = cv2.addWeighted(overlay, 0.85, image, 0.15, 0)
        
        return result
    
    def cleanup(self):
        """Cleanup display resources"""
        cv2.destroyAllWindows()
