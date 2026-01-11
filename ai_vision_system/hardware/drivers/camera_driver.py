#!/usr/bin/env python3
"""
Camera Driver Health Check and Management
==========================================
Provides driver verification, status checking, and health monitoring
for IMX219 dual cameras on Jetson Orin Nano.
"""

import subprocess
import os
import sys
from pathlib import Path


class CameraDriver:
    """Camera driver management and health checking"""
    
    def __init__(self):
        self.video_devices = []
        self.driver_status = {}
        
    def check_video_devices(self):
        """Check for available video devices"""
        video_path = Path("/dev")
        devices = list(video_path.glob("video*"))
        self.video_devices = sorted([str(d) for d in devices], 
                                    key=lambda x: int(x.replace("/dev/video", "")))
        return len(self.video_devices) > 0
    
    def get_driver_info(self, device="/dev/video0"):
        """Get driver information for a video device"""
        try:
            result = subprocess.run(
                ["v4l2-ctl", "--device", device, "--all"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.stdout if result.returncode == 0 else None
        except Exception as e:
            return None
    
    def check_kernel_messages(self):
        """Check kernel messages for camera-related information"""
        try:
            result = subprocess.run(
                ["dmesg"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                imx219_lines = [line for line in lines if 'imx219' in line.lower()]
                return imx219_lines
            return []
        except Exception as e:
            return []
    
    def verify_drivers(self):
        """Verify camera drivers are loaded and working"""
        status = {
            'devices_found': False,
            'device_count': 0,
            'devices': [],
            'driver_loaded': False,
            'kernel_messages': []
        }
        
        # Check for video devices
        if self.check_video_devices():
            status['devices_found'] = True
            status['device_count'] = len(self.video_devices)
            status['devices'] = self.video_devices
            
            # Try to get driver info
            if self.video_devices:
                driver_info = self.get_driver_info(self.video_devices[0])
                status['driver_loaded'] = driver_info is not None
                status['driver_info'] = driver_info
        
        # Check kernel messages
        status['kernel_messages'] = self.check_kernel_messages()
        
        return status
    
    def get_driver_status_report(self):
        """Generate a comprehensive driver status report"""
        status = self.verify_drivers()
        
        report = []
        report.append("=" * 70)
        report.append("Camera Driver Status Report")
        report.append("=" * 70)
        report.append("")
        
        if status['devices_found']:
            report.append(f"✓ Video devices found: {status['device_count']}")
            for device in status['devices']:
                report.append(f"  - {device}")
        else:
            report.append("✗ No video devices found")
        
        report.append("")
        
        if status['driver_loaded']:
            report.append("✓ Driver is loaded and accessible")
        else:
            report.append("✗ Driver may not be loaded or accessible")
        
        report.append("")
        
        if status['kernel_messages']:
            report.append(f"Kernel messages ({len(status['kernel_messages'])} found):")
            for msg in status['kernel_messages'][-5:]:  # Last 5 messages
                report.append(f"  {msg}")
        
        report.append("")
        report.append("=" * 70)
        
        return "\n".join(report)


def main():
    """Main function for command-line usage"""
    driver = CameraDriver()
    status = driver.verify_drivers()
    report = driver.get_driver_status_report()
    
    print(report)
    
    # Return exit code based on status
    if status['devices_found'] and status['driver_loaded']:
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
