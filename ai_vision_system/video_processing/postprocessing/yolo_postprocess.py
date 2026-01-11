#!/usr/bin/env python3
"""
YOLO Postprocessing Module
===========================
Parse detections, apply NMS, scale bounding boxes, and visualization.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict


class YOLOPostprocessor:
    """YOLO detection postprocessing and visualization"""
    
    # COCO class names (80 classes)
    COCO_CLASSES = [
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
    
    def __init__(self, class_names: Optional[List[str]] = None):
        """
        Initialize postprocessor
        
        Args:
            class_names: List of class names (default: COCO classes)
        """
        self.class_names = class_names if class_names else self.COCO_CLASSES
        
        # Color palette for visualization
        self.colors = self._generate_colors(len(self.class_names))
    
    def _generate_colors(self, num_classes: int) -> List[Tuple[int, int, int]]:
        """Generate color palette for classes"""
        np.random.seed(42)  # For consistent colors
        colors = []
        for i in range(num_classes):
            color = tuple(np.random.randint(0, 255, 3).tolist())
            colors.append(color)
        return colors
    
    def scale_boxes(self, boxes: np.ndarray, original_shape: Tuple[int, int],
                   processed_shape: Tuple[int, int], scale: float, padding: Tuple[float, float]) -> np.ndarray:
        """
        Scale bounding boxes from processed image to original image
        
        Args:
            boxes: Bounding boxes [N, 4] in format [x1, y1, x2, y2]
            original_shape: (height, width) of original image
            processed_shape: (height, width) of processed image
            scale: Scale ratio used in preprocessing
            padding: Padding (dw, dh) used in preprocessing
        
        Returns:
            Scaled bounding boxes
        """
        boxes = boxes.copy()
        dw, dh = padding
        
        # Remove padding
        boxes[:, [0, 2]] -= dw  # x coordinates
        boxes[:, [1, 3]] -= dh  # y coordinates
        
        # Scale back
        boxes[:, [0, 2]] /= scale  # x coordinates
        boxes[:, [1, 3]] /= scale  # y coordinates
        
        # Clip to image bounds
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, original_shape[1])
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, original_shape[0])
        
        return boxes
    
    def draw_detections(self, image: np.ndarray, detections: List[Dict],
                       show_labels: bool = True, show_confidence: bool = True,
                       line_thickness: int = 2, font_scale: float = 0.5) -> np.ndarray:
        """
        Draw detections on image
        
        Args:
            image: Input image (BGR format)
            detections: List of detection dictionaries
            show_labels: Show class labels
            show_confidence: Show confidence scores
            line_thickness: Bounding box line thickness
            font_scale: Text font scale
        
        Returns:
            Image with detections drawn
        """
        result_image = image.copy()
        
        for det in detections:
            bbox = det['bbox']  # [x1, y1, x2, y2]
            class_id = det['class_id']
            confidence = det['confidence']
            class_name = det.get('class_name', self.class_names[class_id] if class_id < len(self.class_names) else f'class_{class_id}')
            
            # Get color for this class
            color = self.colors[class_id % len(self.colors)]
            
            # Draw bounding box
            x1, y1, x2, y2 = map(int, bbox)
            cv2.rectangle(result_image, (x1, y1), (x2, y2), color, line_thickness)
            
            # Prepare label text
            label_parts = []
            if show_labels:
                label_parts.append(class_name)
            if show_confidence:
                label_parts.append(f'{confidence:.2f}')
            
            if label_parts:
                label = ' '.join(label_parts)
                
                # Calculate text size
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
                )
                
                # Draw label background
                cv2.rectangle(
                    result_image,
                    (x1, y1 - text_height - baseline - 5),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                
                # Draw label text
                cv2.putText(
                    result_image,
                    label,
                    (x1, y1 - baseline - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    1
                )
        
        return result_image
    
    def draw_detection_count(self, image: np.ndarray, detections: List[Dict],
                           position: Tuple[int, int] = (10, 30)) -> np.ndarray:
        """
        Draw detection count by class
        
        Args:
            image: Input image
            detections: List of detections
            position: Text position (x, y)
        
        Returns:
            Image with count overlay
        """
        result_image = image.copy()
        
        # Count detections by class
        class_counts = defaultdict(int)
        for det in detections:
            class_name = det.get('class_name', f"class_{det['class_id']}")
            class_counts[class_name] += 1
        
        # Draw counts
        y_offset = position[1]
        for class_name, count in sorted(class_counts.items()):
            text = f"{class_name}: {count}"
            cv2.putText(
                result_image,
                text,
                (position[0], y_offset),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )
            y_offset += 25
        
        return result_image
    
    def filter_detections(self, detections: List[Dict],
                         min_confidence: float = 0.0,
                         class_filter: Optional[List[int]] = None) -> List[Dict]:
        """
        Filter detections by confidence and class
        
        Args:
            detections: List of detections
            min_confidence: Minimum confidence threshold
            class_filter: List of class IDs to keep (None = all)
        
        Returns:
            Filtered detections
        """
        filtered = []
        for det in detections:
            if det['confidence'] < min_confidence:
                continue
            if class_filter is not None and det['class_id'] not in class_filter:
                continue
            filtered.append(det)
        return filtered
    
    def get_detection_summary(self, detections: List[Dict]) -> Dict:
        """
        Get summary statistics of detections
        
        Args:
            detections: List of detections
        
        Returns:
            Summary dictionary
        """
        if not detections:
            return {
                'total': 0,
                'by_class': {},
                'avg_confidence': 0.0
            }
        
        by_class = defaultdict(int)
        confidences = []
        
        for det in detections:
            class_name = det.get('class_name', f"class_{det['class_id']}")
            by_class[class_name] += 1
            confidences.append(det['confidence'])
        
        return {
            'total': len(detections),
            'by_class': dict(by_class),
            'avg_confidence': np.mean(confidences) if confidences else 0.0,
            'min_confidence': np.min(confidences) if confidences else 0.0,
            'max_confidence': np.max(confidences) if confidences else 0.0
        }


def merge_detections(detections0: List[Dict], detections1: List[Dict],
                    merge_strategy: str = 'union') -> List[Dict]:
    """
    Merge detections from two cameras
    
    Args:
        detections0: Detections from camera 0
        detections1: Detections from camera 1
        merge_strategy: 'union' or 'intersection'
    
    Returns:
        Merged detections
    """
    if merge_strategy == 'union':
        return detections0 + detections1
    elif merge_strategy == 'intersection':
        # Simplified: would need spatial matching for true intersection
        return detections0 + detections1
    else:
        return detections0 + detections1
