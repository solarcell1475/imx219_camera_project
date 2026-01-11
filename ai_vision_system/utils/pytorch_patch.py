#!/usr/bin/env python3
"""
PyTorch 2.6+ Compatibility Patch
==================================
Patches torch.load to handle YOLO model loading with weights_only=False
for trusted Ultralytics models.
"""

import torch
import functools


def patch_torch_load():
    """Patch torch.load to use weights_only=False for YOLO models"""
    # Store original torch.load
    original_load = torch.load
    
    @functools.wraps(original_load)
    def patched_load(*args, **kwargs):
        # If weights_only is not explicitly set, default to False for YOLO compatibility
        if 'weights_only' not in kwargs:
            kwargs['weights_only'] = False
        return original_load(*args, **kwargs)
    
    # Replace torch.load with patched version
    torch.load = patched_load


# Auto-patch on import
patch_torch_load()
