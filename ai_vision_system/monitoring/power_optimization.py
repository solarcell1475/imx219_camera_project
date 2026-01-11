#!/usr/bin/env python3
"""
Power Optimization and Performance Balancing
=============================================
MAXN power mode configuration and dynamic performance balancing.
"""

import subprocess
import sys
from typing import Dict, Optional
from enum import Enum


class PowerMode(Enum):
    """Jetson power modes"""
    MAXN = 0  # Maximum performance
    MODE_15W = 1
    MODE_10W = 2
    MODE_5W = 3


class PerformanceBalancer:
    """Performance balancing system"""
    
    def __init__(self, target_fps: float = 25.0):
        """
        Initialize performance balancer
        
        Args:
            target_fps: Target FPS for real-time processing
        """
        self.target_fps = target_fps
        self.current_model_size = 's'  # Default: YOLOv8s
        self.current_resolution = (1280, 720)
        self.performance_history = []
    
    def check_power_mode(self) -> Optional[int]:
        """Check current power mode"""
        try:
            result = subprocess.run(
                ['nvpmodel', '-q'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                # Parse output to get mode
                for line in result.stdout.split('\n'):
                    if 'Power Mode' in line or 'nvpmodel' in line:
                        # Try to extract mode number
                        try:
                            mode = int(line.split()[-1])
                            return mode
                        except:
                            pass
            return None
        except:
            return None
    
    def set_power_mode(self, mode: PowerMode) -> bool:
        """Set power mode (requires sudo)"""
        try:
            result = subprocess.run(
                ['sudo', 'nvpmodel', '-m', str(mode.value)],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def set_maxn_mode(self) -> bool:
        """Set to MAXN (maximum performance) mode"""
        return self.set_power_mode(PowerMode.MAXN)
    
    def enable_jetson_clocks(self) -> bool:
        """Enable maximum GPU clocks (requires sudo)"""
        try:
            result = subprocess.run(
                ['sudo', 'jetson_clocks'],
                capture_output=True,
                text=True,
                timeout=10
            )
            return result.returncode == 0
        except:
            return False
    
    def balance_performance(self, current_fps: float, metrics: Dict) -> Dict:
        """
        Balance performance based on current FPS and metrics
        
        Args:
            current_fps: Current FPS
            metrics: Performance metrics
        
        Returns:
            Adjustment recommendations
        """
        recommendations = {
            'model_size': None,
            'resolution': None,
            'confidence_threshold': None,
            'action': None
        }
        
        # If FPS is too low
        if current_fps < self.target_fps * 0.8:
            # Try to improve performance
            if self.current_model_size == 'm':
                recommendations['model_size'] = 's'
                recommendations['action'] = 'switch_to_smaller_model'
            elif self.current_model_size == 's':
                recommendations['model_size'] = 'n'
                recommendations['action'] = 'switch_to_nano_model'
            
            # Reduce resolution
            if self.current_resolution[0] > 640:
                recommendations['resolution'] = (640, 480)
                recommendations['action'] = 'reduce_resolution'
            
            # Increase confidence threshold (fewer detections = faster)
            recommendations['confidence_threshold'] = 0.3
        
        # If FPS is good, can increase quality
        elif current_fps > self.target_fps * 1.2:
            if self.current_model_size == 'n':
                recommendations['model_size'] = 's'
                recommendations['action'] = 'switch_to_larger_model'
            elif self.current_model_size == 's' and current_fps > self.target_fps * 1.5:
                recommendations['model_size'] = 'm'
                recommendations['action'] = 'switch_to_medium_model'
            
            # Increase resolution if possible
            if self.current_resolution[0] < 1280:
                recommendations['resolution'] = (1280, 720)
                recommendations['action'] = 'increase_resolution'
        
        return recommendations
    
    def get_optimal_model(self, target_fps: float) -> str:
        """Get optimal model size for target FPS"""
        # Model performance estimates (approximate FPS on Jetson Orin Nano)
        model_performance = {
            'n': 40,  # YOLOv8n
            's': 25,  # YOLOv8s
            'm': 15,  # YOLOv8m
            'l': 10,  # YOLOv8l
            'x': 7,   # YOLOv8x
        }
        
        # Find best model
        for size, fps in model_performance.items():
            if fps >= target_fps:
                return size
        
        # Default to nano if no match
        return 'n'
    
    def get_optimal_resolution(self, target_fps: float, model_size: str) -> tuple:
        """Get optimal resolution for target FPS and model"""
        # Resolution performance estimates
        if model_size == 'n':
            # Nano model can handle higher resolutions
            if target_fps >= 30:
                return (1280, 720)
            elif target_fps >= 20:
                return (1280, 720)
            else:
                return (640, 480)
        elif model_size == 's':
            # Small model
            if target_fps >= 25:
                return (1280, 720)
            else:
                return (640, 480)
        else:
            # Medium and larger
            return (640, 480)


def verify_maxn_mode() -> bool:
    """Verify MAXN mode is active"""
    try:
        result = subprocess.run(
            ['nvpmodel', '-q'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            output = result.stdout.lower()
            return 'mode 0' in output or 'maxn' in output
        return False
    except:
        return False


def main():
    """Test function"""
    print("Testing Power Optimization...")
    print("=" * 70)
    
    balancer = PerformanceBalancer(target_fps=25.0)
    
    # Check power mode
    mode = balancer.check_power_mode()
    if mode is not None:
        print(f"Current power mode: {mode}")
        if mode == 0:
            print("✓ MAXN mode is active")
        else:
            print("⚠ Not in MAXN mode. Run: sudo nvpmodel -m 0")
    else:
        print("⚠ Could not determine power mode")
    
    # Get optimal model
    optimal_model = balancer.get_optimal_model(25.0)
    print(f"\nOptimal model for 25 FPS: YOLOv8{optimal_model}")
    
    # Get optimal resolution
    optimal_res = balancer.get_optimal_resolution(25.0, optimal_model)
    print(f"Optimal resolution: {optimal_res[0]}x{optimal_res[1]}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
