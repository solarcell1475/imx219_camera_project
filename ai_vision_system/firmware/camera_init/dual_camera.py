#!/usr/bin/env python3
"""
Dual Camera Initialization Module
==================================
Unified camera initialization class with support for multiple resolution presets,
automatic fallback, and capability detection.
"""

import cv2
import sys
from enum import Enum
from typing import Optional, Tuple, Dict


class ResolutionPreset(Enum):
    """Camera resolution presets"""
    VGA = (640, 480, 60)
    HD = (1280, 720, 30)
    FULL_HD = (1920, 1080, 15)
    BALANCED = (1280, 720, 30)  # Default
    HIGH_QUALITY = (1920, 1080, 15)
    FAST = (640, 480, 60)


class DualCamera:
    """Dual camera initialization and management"""
    
    def __init__(self, resolution: ResolutionPreset = ResolutionPreset.BALANCED):
        self.resolution = resolution
        self.width, self.height, self.fps = resolution.value
        self.cap0: Optional[cv2.VideoCapture] = None
        self.cap1: Optional[cv2.VideoCapture] = None
        self.initialized = False
        self.capabilities = {}
        
    def _create_gstreamer_pipeline(self, sensor_id: int) -> str:
        """Create GStreamer pipeline string for a camera"""
        return (
            f"nvarguscamerasrc sensor-id={sensor_id} ! "
            f"video/x-raw(memory:NVMM), width={self.width}, height={self.height}, "
            f"format=NV12, framerate={self.fps}/1 ! "
            f"nvvidconv ! video/x-raw, format=BGRx ! "
            f"videoconvert ! video/x-raw, format=BGR ! appsink"
        )
    
    def detect_capabilities(self) -> Dict:
        """Detect camera capabilities"""
        capabilities = {
            'camera0_available': False,
            'camera1_available': False,
            'resolutions_supported': [],
            'max_fps': {}
        }
        
        # Test camera 0
        try:
            cap = cv2.VideoCapture(0)
            if cap.isOpened():
                capabilities['camera0_available'] = True
                cap.release()
        except:
            pass
        
        # Test camera 1
        try:
            cap = cv2.VideoCapture(1)
            if cap.isOpened():
                capabilities['camera1_available'] = True
                cap.release()
        except:
            pass
        
        self.capabilities = capabilities
        return capabilities
    
    def initialize(self, fallback: bool = True) -> bool:
        """Initialize both cameras with automatic fallback"""
        # Detect capabilities first
        self.detect_capabilities()
        
        # Try GStreamer first (preferred for Jetson)
        success = self._initialize_gstreamer()
        
        if not success and fallback:
            # Fallback to direct OpenCV access
            print("GStreamer initialization failed, trying OpenCV direct access...")
            success = self._initialize_opencv()
        
        self.initialized = success
        return success
    
    def _initialize_gstreamer(self) -> bool:
        """Initialize cameras using GStreamer"""
        try:
            gst_str_0 = self._create_gstreamer_pipeline(0)
            gst_str_1 = self._create_gstreamer_pipeline(1)
            
            self.cap0 = cv2.VideoCapture(gst_str_0, cv2.CAP_GSTREAMER)
            self.cap1 = cv2.VideoCapture(gst_str_1, cv2.CAP_GSTREAMER)
            
            if self.cap0.isOpened() and self.cap1.isOpened():
                # Test reading a frame
                ret0, _ = self.cap0.read()
                ret1, _ = self.cap1.read()
                
                if ret0 and ret1:
                    return True
                else:
                    self.release()
                    return False
            else:
                self.release()
                return False
        except Exception as e:
            print(f"GStreamer initialization error: {e}")
            self.release()
            return False
    
    def _initialize_opencv(self) -> bool:
        """Initialize cameras using direct OpenCV access (fallback)"""
        try:
            self.cap0 = cv2.VideoCapture(0)
            self.cap1 = cv2.VideoCapture(1)
            
            if self.cap0.isOpened() and self.cap1.isOpened():
                # Test reading a frame
                ret0, _ = self.cap0.read()
                ret1, _ = self.cap1.read()
                
                if ret0 and ret1:
                    # Set properties
                    self.cap0.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap0.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap0.set(cv2.CAP_PROP_FPS, self.fps)
                    self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    self.cap1.set(cv2.CAP_PROP_FPS, self.fps)
                    return True
                else:
                    self.release()
                    return False
            else:
                self.release()
                return False
        except Exception as e:
            print(f"OpenCV initialization error: {e}")
            self.release()
            return False
    
    def read(self) -> Tuple[bool, Optional[cv2.Mat], Optional[cv2.Mat]]:
        """Read frames from both cameras"""
        if not self.initialized:
            return False, None, None
        
        ret0, frame0 = self.cap0.read()
        ret1, frame1 = self.cap1.read()
        
        if ret0 and ret1:
            return True, frame0, frame1
        else:
            return False, frame0, frame1
    
    def set_resolution(self, resolution: ResolutionPreset) -> bool:
        """Change resolution (requires reinitialization)"""
        if self.initialized:
            self.release()
        
        self.resolution = resolution
        self.width, self.height, self.fps = resolution.value
        return self.initialize()
    
    def get_info(self) -> Dict:
        """Get camera information"""
        info = {
            'initialized': self.initialized,
            'resolution': f"{self.width}x{self.height}",
            'fps': self.fps,
            'capabilities': self.capabilities
        }
        
        if self.cap0 and self.cap0.isOpened():
            info['camera0'] = {
                'width': int(self.cap0.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap0.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap0.get(cv2.CAP_PROP_FPS)
            }
        
        if self.cap1 and self.cap1.isOpened():
            info['camera1'] = {
                'width': int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
                'height': int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                'fps': self.cap1.get(cv2.CAP_PROP_FPS)
            }
        
        return info
    
    def release(self):
        """Release camera resources"""
        if self.cap0:
            self.cap0.release()
            self.cap0 = None
        if self.cap1:
            self.cap1.release()
            self.cap1 = None
        self.initialized = False
    
    def __enter__(self):
        """Context manager entry"""
        if not self.initialized:
            self.initialize()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.release()


def main():
    """Test function"""
    print("Testing Dual Camera Initialization...")
    
    camera = DualCamera(ResolutionPreset.BALANCED)
    
    if camera.initialize():
        print("✓ Cameras initialized successfully")
        info = camera.get_info()
        print(f"  Resolution: {info['resolution']}")
        print(f"  FPS: {info['fps']}")
        
        # Test reading frames
        print("\nTesting frame capture...")
        for i in range(5):
            ret, frame0, frame1 = camera.read()
            if ret:
                print(f"  Frame {i+1}: ✓ ({frame0.shape}, {frame1.shape})")
            else:
                print(f"  Frame {i+1}: ✗")
        
        camera.release()
        print("\n✓ Test completed successfully")
        return 0
    else:
        print("✗ Failed to initialize cameras")
        return 1


if __name__ == "__main__":
    sys.exit(main())
