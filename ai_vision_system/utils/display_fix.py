#!/usr/bin/env python3
"""
Display Fix Utilities
=====================
Fixes for OpenCV display issues on Jetson (noise, black screen, etc.)
"""

import os
import cv2
import numpy as np


def setup_display_environment():
    """Setup environment variables for proper display"""
    # Set Qt platform if not set
    if 'QT_QPA_PLATFORM' not in os.environ:
        os.environ['QT_QPA_PLATFORM'] = 'xcb'
    
    # Set OpenCV to prefer GTK backend
    os.environ['OPENCV_VIDEOIO_PRIORITY_MSMF'] = '0'
    
    # Try to use X11
    if 'DISPLAY' not in os.environ:
        os.environ['DISPLAY'] = ':0'


def create_window_safe(window_name, width, height):
    """Safely create OpenCV window with fallbacks"""
    try:
        # Try WINDOW_NORMAL first
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, width, height)
        return True
    except:
        try:
            # Fallback to AUTOSIZE
            cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
            return True
        except Exception as e:
            print(f"Failed to create window: {e}")
            return False


def validate_image_for_display(image):
    """Validate and fix image for display"""
    if image is None:
        return None
    
    # Check if empty
    if image.size == 0:
        return None
    
    # Convert to uint8 if needed
    if image.dtype != np.uint8:
        if image.dtype == np.float32 or image.dtype == np.float64:
            # Assume 0-1 range, convert to 0-255
            image = (image * 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Handle different channel counts
    if len(image.shape) == 2:
        # Grayscale to BGR
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    elif len(image.shape) == 3:
        if image.shape[2] == 1:
            # Single channel to BGR
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        elif image.shape[2] == 4:
            # RGBA/BGRA to BGR
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
        elif image.shape[2] == 3:
            # Already 3 channels, ensure BGR
            pass  # Assume already BGR
    
    return image


def imshow_safe(window_name, image):
    """Safely display image with error handling"""
    try:
        image = validate_image_for_display(image)
        if image is None:
            print("Warning: Cannot display None or empty image")
            return False
        
        cv2.imshow(window_name, image)
        return True
    except Exception as e:
        print(f"Display error: {e}")
        return False
