#!/usr/bin/env python3
"""
Performance Metrics Collection
================================
Collects CPU/GPU utilization, memory usage, inference latency, FPS,
temperature, and YOLO-specific metrics.
"""

import time
import psutil
from typing import Dict, Optional
from collections import deque


class PerformanceMetrics:
    """Performance metrics collection"""
    
    def __init__(self, history_size: int = 100):
        """
        Initialize metrics collector
        
        Args:
            history_size: Size of history buffer
        """
        self.history_size = history_size
        self.metrics_history = {
            'cpu_percent': deque(maxlen=history_size),
            'gpu_percent': deque(maxlen=history_size),
            'memory_percent': deque(maxlen=history_size),
            'gpu_memory_mb': deque(maxlen=history_size),
            'inference_time_ms': deque(maxlen=history_size),
            'fps': deque(maxlen=history_size),
            'temperature_cpu': deque(maxlen=history_size),
            'temperature_gpu': deque(maxlen=history_size)
        }
        
        # Current metrics
        self.current_metrics = {}
        
        # Try to import nvidia-ml-py for GPU metrics
        try:
            import pynvml
            pynvml.nvmlInit()
            self.gpu_available = True
            self.gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except:
            self.gpu_available = False
            self.gpu_handle = None
    
    def collect_cpu_metrics(self) -> Dict:
        """Collect CPU metrics"""
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        return {
            'cpu_percent': cpu_percent,
            'cpu_count': cpu_count,
            'cpu_freq_mhz': psutil.cpu_freq().current if psutil.cpu_freq() else None
        }
    
    def collect_memory_metrics(self) -> Dict:
        """Collect memory metrics"""
        memory = psutil.virtual_memory()
        
        return {
            'memory_total_gb': memory.total / (1024**3),
            'memory_available_gb': memory.available / (1024**3),
            'memory_used_gb': memory.used / (1024**3),
            'memory_percent': memory.percent,
            'memory_usage_mb': memory.used / (1024**2)
        }
    
    def collect_gpu_metrics(self) -> Dict:
        """Collect GPU metrics"""
        if not self.gpu_available:
            return {
                'gpu_available': False,
                'gpu_utilization': 0.0,
                'gpu_memory_used_mb': 0.0,
                'gpu_memory_total_mb': 0.0,
                'gpu_temperature_c': 0.0
            }
        
        try:
            import pynvml
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(self.gpu_handle)
            gpu_util = util.gpu
            
            # GPU memory
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.gpu_handle)
            gpu_mem_used = mem_info.used / (1024**2)  # MB
            gpu_mem_total = mem_info.total / (1024**2)  # MB
            
            # GPU temperature
            try:
                temp = pynvml.nvmlDeviceGetTemperature(self.gpu_handle, pynvml.NVML_TEMPERATURE_GPU)
            except:
                temp = 0.0
            
            return {
                'gpu_available': True,
                'gpu_utilization': float(gpu_util),
                'gpu_memory_used_mb': float(gpu_mem_used),
                'gpu_memory_total_mb': float(gpu_mem_total),
                'gpu_temperature_c': float(temp)
            }
        except Exception as e:
            return {
                'gpu_available': False,
                'gpu_error': str(e)
            }
    
    def collect_temperature_metrics(self) -> Dict:
        """Collect temperature metrics"""
        temps = {}
        
        # CPU temperature (if available)
        try:
            cpu_temp = psutil.sensors_temperatures()
            if cpu_temp:
                # Try to get CPU temperature
                for name, entries in cpu_temp.items():
                    if entries:
                        temps[f'cpu_temp_{name}'] = entries[0].current
        except:
            pass
        
        # GPU temperature (from GPU metrics)
        gpu_metrics = self.collect_gpu_metrics()
        if gpu_metrics.get('gpu_available'):
            temps['gpu_temperature_c'] = gpu_metrics.get('gpu_temperature_c', 0.0)
        
        return temps
    
    def collect_all_metrics(self) -> Dict:
        """Collect all metrics"""
        metrics = {}
        
        # CPU
        cpu_metrics = self.collect_cpu_metrics()
        metrics.update(cpu_metrics)
        
        # Memory
        memory_metrics = self.collect_memory_metrics()
        metrics.update(memory_metrics)
        
        # GPU
        gpu_metrics = self.collect_gpu_metrics()
        metrics.update(gpu_metrics)
        
        # Temperature
        temp_metrics = self.collect_temperature_metrics()
        metrics.update(temp_metrics)
        
        # Timestamp
        metrics['timestamp'] = time.time()
        
        # Update history
        self.metrics_history['cpu_percent'].append(metrics.get('cpu_percent', 0))
        self.metrics_history['gpu_percent'].append(metrics.get('gpu_utilization', 0))
        self.metrics_history['memory_percent'].append(metrics.get('memory_percent', 0))
        if 'gpu_memory_used_mb' in metrics:
            self.metrics_history['gpu_memory_mb'].append(metrics['gpu_memory_used_mb'])
        
        self.current_metrics = metrics
        
        return metrics
    
    def add_inference_metrics(self, inference_time_ms: float, fps: float):
        """Add YOLO-specific metrics"""
        self.metrics_history['inference_time_ms'].append(inference_time_ms)
        self.metrics_history['fps'].append(fps)
        
        self.current_metrics['inference_time_ms'] = inference_time_ms
        self.current_metrics['fps'] = fps
    
    def get_current_metrics(self) -> Dict:
        """Get current metrics"""
        return self.current_metrics.copy()
    
    def get_average_metrics(self, window: int = 10) -> Dict:
        """Get average metrics over a window"""
        if window > self.history_size:
            window = self.history_size
        
        averages = {}
        
        for key, history in self.metrics_history.items():
            if len(history) > 0:
                recent = list(history)[-window:]
                averages[key] = sum(recent) / len(recent) if recent else 0.0
        
        return averages
    
    def get_summary(self) -> Dict:
        """Get metrics summary"""
        summary = {
            'current': self.get_current_metrics(),
            'averages': self.get_average_metrics(10)
        }
        
        # Add min/max for key metrics
        for key in ['cpu_percent', 'gpu_percent', 'memory_percent', 'fps', 'inference_time_ms']:
            if key in self.metrics_history and len(self.metrics_history[key]) > 0:
                values = list(self.metrics_history[key])
                summary[f'{key}_min'] = min(values)
                summary[f'{key}_max'] = max(values)
        
        return summary
