#!/usr/bin/env python3
"""
Performance Alerts
==================
Alerts for FPS drops, high memory usage, temperature warnings, etc.
"""

from typing import Dict, List, Callable, Optional
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class PerformanceAlerts:
    """Performance alert system"""
    
    def __init__(self):
        """Initialize alert system"""
        self.alert_callbacks: List[Callable] = []
        self.alert_history = []
        self.thresholds = {
            'fps_min': 15.0,
            'memory_max_percent': 85.0,
            'gpu_temp_max_c': 80.0,
            'cpu_temp_max_c': 85.0,
            'inference_time_max_ms': 100.0
        }
        self.alert_cooldown = {}  # Prevent spam
    
    def set_threshold(self, key: str, value: float):
        """Set alert threshold"""
        if key in self.thresholds:
            self.thresholds[key] = value
    
    def register_callback(self, callback: Callable):
        """Register alert callback"""
        self.alert_callbacks.append(callback)
    
    def check_metrics(self, metrics: Dict) -> List[Dict]:
        """Check metrics and generate alerts"""
        alerts = []
        current_time = time.time()
        
        # FPS check
        if 'fps' in metrics:
            fps = metrics['fps']
            if fps < self.thresholds['fps_min']:
                alert = self._create_alert(
                    'fps_low',
                    AlertLevel.WARNING,
                    f"FPS dropped below threshold: {fps:.1f} < {self.thresholds['fps_min']}",
                    {'fps': fps, 'threshold': self.thresholds['fps_min']}
                )
                if self._check_cooldown('fps_low', current_time):
                    alerts.append(alert)
        
        # Memory check
        if 'memory_percent' in metrics:
            mem = metrics['memory_percent']
            if mem > self.thresholds['memory_max_percent']:
                alert = self._create_alert(
                    'memory_high',
                    AlertLevel.WARNING,
                    f"Memory usage high: {mem:.1f}% > {self.thresholds['memory_max_percent']}%",
                    {'memory_percent': mem, 'threshold': self.thresholds['memory_max_percent']}
                )
                if self._check_cooldown('memory_high', current_time):
                    alerts.append(alert)
        
        # GPU temperature check
        if 'gpu_temperature_c' in metrics:
            temp = metrics['gpu_temperature_c']
            if temp > self.thresholds['gpu_temp_max_c']:
                level = AlertLevel.ERROR if temp > self.thresholds['gpu_temp_max_c'] + 10 else AlertLevel.WARNING
                alert = self._create_alert(
                    'gpu_temp_high',
                    level,
                    f"GPU temperature high: {temp:.1f}°C > {self.thresholds['gpu_temp_max_c']}°C",
                    {'temperature': temp, 'threshold': self.thresholds['gpu_temp_max_c']}
                )
                if self._check_cooldown('gpu_temp_high', current_time):
                    alerts.append(alert)
        
        # Inference time check
        if 'inference_time_ms' in metrics:
            inf_time = metrics['inference_time_ms']
            if inf_time > self.thresholds['inference_time_max_ms']:
                alert = self._create_alert(
                    'inference_slow',
                    AlertLevel.WARNING,
                    f"Inference time high: {inf_time:.1f}ms > {self.thresholds['inference_time_max_ms']}ms",
                    {'inference_time': inf_time, 'threshold': self.thresholds['inference_time_max_ms']}
                )
                if self._check_cooldown('inference_slow', current_time):
                    alerts.append(alert)
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    print(f"Error in alert callback: {e}")
        
        # Store alerts
        self.alert_history.extend(alerts)
        if len(self.alert_history) > 100:
            self.alert_history = self.alert_history[-100:]
        
        return alerts
    
    def _create_alert(self, alert_id: str, level: AlertLevel, message: str, data: Dict) -> Dict:
        """Create alert dictionary"""
        return {
            'id': alert_id,
            'level': level.value,
            'message': message,
            'data': data,
            'timestamp': time.time()
        }
    
    def _check_cooldown(self, alert_id: str, current_time: float, cooldown: float = 5.0) -> bool:
        """Check if alert is in cooldown period"""
        if alert_id not in self.alert_cooldown:
            self.alert_cooldown[alert_id] = current_time
            return True
        
        if current_time - self.alert_cooldown[alert_id] > cooldown:
            self.alert_cooldown[alert_id] = current_time
            return True
        
        return False
    
    def get_recent_alerts(self, count: int = 10) -> List[Dict]:
        """Get recent alerts"""
        return self.alert_history[-count:]


# Import time for cooldown
import time
