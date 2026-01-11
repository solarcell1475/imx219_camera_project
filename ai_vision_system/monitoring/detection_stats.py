#!/usr/bin/env python3
"""
Detection Statistics Tracker
=============================
Tracks object detection statistics including appearance rate and probability.
"""

import time
from collections import defaultdict, deque
from typing import Dict, List, Tuple
from dataclasses import dataclass


@dataclass
class ObjectStats:
    """Statistics for a single object class"""
    class_name: str
    class_id: int
    total_detections: int = 0
    total_frames: int = 0
    appearance_count: int = 0
    confidence_sum: float = 0.0
    max_confidence: float = 0.0
    min_confidence: float = 1.0
    last_seen: float = 0.0
    
    @property
    def appearance_rate(self) -> float:
        """Appearance rate (frames with detection / total frames)"""
        if self.total_frames == 0:
            return 0.0
        return self.appearance_count / self.total_frames
    
    @property
    def avg_confidence(self) -> float:
        """Average confidence/probability"""
        if self.total_detections == 0:
            return 0.0
        return self.confidence_sum / self.total_detections


class DetectionStatistics:
    """Track detection statistics over time"""
    
    def __init__(self, history_size: int = 1000):
        """
        Initialize statistics tracker
        
        Args:
            history_size: Number of frames to keep in history
        """
        self.history_size = history_size
        self.stats: Dict[str, ObjectStats] = {}
        self.frame_history = deque(maxlen=history_size)
        self.total_frames = 0
        self.start_time = time.time()
        
    def update(self, detections: List[Dict], frame_time: float = None):
        """
        Update statistics with new detections
        
        Args:
            detections: List of detection dictionaries
            frame_time: Timestamp for this frame (defaults to current time)
        """
        if frame_time is None:
            frame_time = time.time()
        
        self.total_frames += 1
        
        # Track which classes appeared in this frame
        classes_in_frame = set()
        
        for det in detections:
            class_name = det.get('class_name', f"class_{det['class_id']}")
            class_id = det.get('class_id', -1)
            confidence = det.get('confidence', 0.0)
            
            # Initialize stats for this class if needed
            if class_name not in self.stats:
                self.stats[class_name] = ObjectStats(
                    class_name=class_name,
                    class_id=class_id
                )
            
            stat = self.stats[class_name]
            stat.total_detections += 1
            stat.confidence_sum += confidence
            stat.max_confidence = max(stat.max_confidence, confidence)
            stat.min_confidence = min(stat.min_confidence, confidence)
            stat.last_seen = frame_time
            
            classes_in_frame.add(class_name)
        
        # Update appearance count for classes detected in this frame
        for class_name in classes_in_frame:
            self.stats[class_name].appearance_count += 1
        
        # Update total frames for all classes
        for stat in self.stats.values():
            stat.total_frames = self.total_frames
        
        # Store frame data
        self.frame_history.append({
            'timestamp': frame_time,
            'detections': len(detections),
            'classes': list(classes_in_frame)
        })
    
    def get_stats_sorted_by_appearance(self, min_appearances: int = 1) -> List[ObjectStats]:
        """
        Get statistics sorted by appearance rate (descending)
        
        Args:
            min_appearances: Minimum number of appearances to include
            
        Returns:
            List of ObjectStats sorted by appearance rate
        """
        filtered = [
            stat for stat in self.stats.values()
            if stat.appearance_count >= min_appearances
        ]
        return sorted(filtered, key=lambda x: x.appearance_rate, reverse=True)
    
    def get_stats_sorted_by_confidence(self, min_appearances: int = 1) -> List[ObjectStats]:
        """
        Get statistics sorted by average confidence (descending)
        
        Args:
            min_appearances: Minimum number of appearances to include
            
        Returns:
            List of ObjectStats sorted by average confidence
        """
        filtered = [
            stat for stat in self.stats.values()
            if stat.appearance_count >= min_appearances
        ]
        return sorted(filtered, key=lambda x: x.avg_confidence, reverse=True)
    
    def get_stats_sorted_by_total(self, min_appearances: int = 1) -> List[ObjectStats]:
        """
        Get statistics sorted by total detections (descending)
        
        Args:
            min_appearances: Minimum number of appearances to include
            
        Returns:
            List of ObjectStats sorted by total detections
        """
        filtered = [
            stat for stat in self.stats.values()
            if stat.appearance_count >= min_appearances
        ]
        return sorted(filtered, key=lambda x: x.total_detections, reverse=True)
    
    def get_summary(self) -> Dict:
        """Get summary statistics"""
        return {
            'total_frames': self.total_frames,
            'total_classes': len(self.stats),
            'uptime_seconds': time.time() - self.start_time,
            'classes_detected': len([s for s in self.stats.values() if s.appearance_count > 0])
        }
    
    def reset(self):
        """Reset all statistics"""
        self.stats.clear()
        self.frame_history.clear()
        self.total_frames = 0
        self.start_time = time.time()
