#!/usr/bin/env python3
"""
Performance Dashboard
======================
Text-based or simple GUI dashboard for real-time performance monitoring.
"""

import time
from typing import Dict, Optional
from datetime import datetime


class PerformanceDashboard:
    """Performance monitoring dashboard"""
    
    def __init__(self, update_interval: float = 1.0):
        """
        Initialize dashboard
        
        Args:
            update_interval: Update interval in seconds
        """
        self.update_interval = update_interval
        self.last_update = time.time()
        self.metrics = {}
    
    def update(self, metrics: Dict):
        """Update dashboard with new metrics"""
        self.metrics = metrics
        self.last_update = time.time()
    
    def print_dashboard(self):
        """Print text-based dashboard"""
        print("\n" + "=" * 70)
        print(f"Performance Dashboard - {datetime.now().strftime('%H:%M:%S')}")
        print("=" * 70)
        
        # CPU
        if 'cpu_percent' in self.metrics:
            cpu = self.metrics['cpu_percent']
            print(f"CPU Usage:     {cpu:6.1f}%")
        
        # Memory
        if 'memory_percent' in self.metrics:
            mem = self.metrics['memory_percent']
            mem_used = self.metrics.get('memory_used_gb', 0)
            mem_total = self.metrics.get('memory_total_gb', 0)
            print(f"Memory:        {mem:6.1f}% ({mem_used:.2f}/{mem_total:.2f} GB)")
        
        # GPU
        if 'gpu_utilization' in self.metrics:
            gpu = self.metrics['gpu_utilization']
            gpu_mem = self.metrics.get('gpu_memory_used_mb', 0)
            gpu_mem_total = self.metrics.get('gpu_memory_total_mb', 0)
            print(f"GPU Usage:     {gpu:6.1f}%")
            print(f"GPU Memory:    {gpu_mem:6.1f} MB / {gpu_mem_total:.1f} MB")
        
        # Temperature
        if 'gpu_temperature_c' in self.metrics:
            temp = self.metrics['gpu_temperature_c']
            print(f"GPU Temp:      {temp:6.1f}°C")
        
        # Inference
        if 'inference_time_ms' in self.metrics:
            inf_time = self.metrics['inference_time_ms']
            print(f"Inference:     {inf_time:6.1f} ms")
        
        # FPS
        if 'fps' in self.metrics:
            fps = self.metrics['fps']
            print(f"FPS:           {fps:6.1f}")
        
        print("=" * 70)
    
    def get_dashboard_text(self) -> str:
        """Get dashboard as text string"""
        lines = []
        lines.append("=" * 70)
        lines.append(f"Performance Dashboard - {datetime.now().strftime('%H:%M:%S')}")
        lines.append("=" * 70)
        
        if 'cpu_percent' in self.metrics:
            lines.append(f"CPU Usage:     {self.metrics['cpu_percent']:6.1f}%")
        
        if 'memory_percent' in self.metrics:
            mem = self.metrics['memory_percent']
            mem_used = self.metrics.get('memory_used_gb', 0)
            mem_total = self.metrics.get('memory_total_gb', 0)
            lines.append(f"Memory:        {mem:6.1f}% ({mem_used:.2f}/{mem_total:.2f} GB)")
        
        if 'gpu_utilization' in self.metrics:
            gpu = self.metrics['gpu_utilization']
            lines.append(f"GPU Usage:     {gpu:6.1f}%")
        
        if 'inference_time_ms' in self.metrics:
            lines.append(f"Inference:     {self.metrics['inference_time_ms']:6.1f} ms")
        
        if 'fps' in self.metrics:
            lines.append(f"FPS:           {self.metrics['fps']:6.1f}")
        
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def should_update(self) -> bool:
        """Check if dashboard should update"""
        return (time.time() - self.last_update) >= self.update_interval
