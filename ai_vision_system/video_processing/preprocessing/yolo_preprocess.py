#!/usr/bin/env python3
"""
YOLO Preprocessing Module
===========================
Frame resizing (640x640), letterbox padding, normalization, and tensor conversion.
"""

import cv2
import numpy as np
from typing import Tuple, Optional
import torch


class YOLOPreprocessor:
    """YOLO image preprocessing"""
    
    def __init__(self, input_size: int = 640, normalize: bool = True):
        """
        Initialize preprocessor
        
        Args:
            input_size: Input image size (default 640 for YOLO)
            normalize: Whether to normalize to [0, 1] range
        """
        self.input_size = input_size
        self.normalize = normalize
        
        # Letterbox padding parameters
        self.pad_color = (114, 114, 114)  # Gray padding
    
    def letterbox(self, image: np.ndarray, new_shape: Tuple[int, int] = (640, 640),
                 color: Tuple[int, int, int] = (114, 114, 114),
                 auto: bool = True, scaleFill: bool = False, scaleup: bool = True) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resize image with letterbox padding to maintain aspect ratio
        
        Args:
            image: Input image (BGR format)
            new_shape: Target size (width, height)
            color: Padding color
            auto: Auto padding
            scaleFill: Scale to fill
            scaleup: Allow upscaling
        
        Returns:
            (processed_image, scale_ratio, padding)
        """
        shape = image.shape[:2]  # Current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # Only scale down, don't scale up
            r = min(r, 1.0)
        
        # Compute padding
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        
        if auto:  # Minimum rectangle
            dw, dh = np.mod(dw, 32), np.mod(dh, 32)  # wh padding
        elif scaleFill:  # Stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            r = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios
        
        dw /= 2  # Divide padding into 2 sides
        dh /= 2
        
        if shape[::-1] != new_unpad:  # Resize
            image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        
        image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # Add border
        
        return image, r, (dw, dh)
    
    def preprocess(self, image: np.ndarray, device: str = 'cuda') -> torch.Tensor:
        """
        Preprocess image for YOLO inference
        
        Args:
            image: Input image (BGR format, numpy array)
            device: Target device ('cuda' or 'cpu')
        
        Returns:
            Preprocessed tensor ready for YOLO
        """
        # Letterbox resize
        processed, scale, padding = self.letterbox(image, (self.input_size, self.input_size))
        
        # BGR to RGB
        processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
        
        # HWC to CHW
        processed = processed.transpose((2, 0, 1))
        
        # Convert to float and normalize
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0
        else:
            processed = processed.astype(np.float32)
        
        # Convert to tensor
        tensor = torch.from_numpy(processed)
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(device)
        
        return tensor
    
    def preprocess_batch(self, images: list, device: str = 'cuda') -> torch.Tensor:
        """
        Preprocess batch of images
        
        Args:
            images: List of input images (BGR format)
            device: Target device
        
        Returns:
            Batch tensor [N, C, H, W]
        """
        tensors = []
        for image in images:
            tensor = self.preprocess(image, device)
            tensors.append(tensor.squeeze(0))  # Remove batch dim for stacking
        
        # Stack into batch
        batch_tensor = torch.stack(tensors)
        return batch_tensor
    
    def get_scale_and_padding(self, original_shape: Tuple[int, int]) -> Tuple[float, Tuple[int, int]]:
        """
        Get scale and padding for a given original image shape
        
        Args:
            original_shape: (height, width) of original image
        
        Returns:
            (scale, padding)
        """
        shape = original_shape
        new_shape = (self.input_size, self.input_size)
        
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        dw, dh = np.mod(dw, 32), np.mod(dh, 32)
        dw /= 2
        dh /= 2
        
        return r, (dw, dh)


def scale_coords(coords: np.ndarray, original_shape: Tuple[int, int],
                processed_shape: Tuple[int, int], scale: float, padding: Tuple[float, float]) -> np.ndarray:
    """
    Scale coordinates from processed image back to original image
    
    Args:
        coords: Coordinates in processed image [x1, y1, x2, y2]
        original_shape: (height, width) of original image
        processed_shape: (height, width) of processed image
        scale: Scale ratio used
        padding: Padding (dw, dh) used
    
    Returns:
        Scaled coordinates in original image
    """
    # Rescale from processed to original
    coords = coords.copy()
    
    # Remove padding
    dw, dh = padding
    coords[:, [0, 2]] -= dw  # x coordinates
    coords[:, [1, 3]] -= dh  # y coordinates
    
    # Scale back
    coords[:, [0, 2]] /= scale  # x coordinates
    coords[:, [1, 3]] /= scale  # y coordinates
    
    # Clip to image bounds
    coords[:, [0, 2]] = np.clip(coords[:, [0, 2]], 0, original_shape[1])
    coords[:, [1, 3]] = np.clip(coords[:, [1, 3]], 0, original_shape[0])
    
    return coords
