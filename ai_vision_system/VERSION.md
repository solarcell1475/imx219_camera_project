# Version History

## Version 1.0 (Current/Stable)
- Dual camera support with IMX219 cameras
- YOLO inference (YOLOv8/v9/v10/v11)
- Real-time display with multiple view modes
- Performance monitoring and metrics
- CPU fallback mode (GPU ready but requires PyTorch CUDA wheel)
- Parallel inference support
- Resolution optimization (640x480 inference, 1280x720 display)

**Performance:**
- CPU mode: ~0.5-1 FPS
- GPU mode (when available): 15-30+ FPS expected

## Version 2.0 (Current)
**New Features:**
- **Detection Statistics Tracking**: Tracks object appearance rate, average confidence, total detections
- **GUI Statistics Table**: Real-time table overlay showing:
  - Object class names
  - Appearance rate (percentage of frames detected)
  - Average confidence/probability
  - Total detection count
- **Sortable Statistics**: Sort by appearance rate, confidence, or total detections (keys 1/2/3)
- **Enhanced FPS Optimizations**:
  - Improved adaptive frame skipping algorithm
  - Better inference time calculation
  - Optimized detection statistics updates

**Keyboard Controls (Version 2):**
- `T` - Toggle statistics table on/off
- `1` - Sort statistics by appearance rate
- `2` - Sort statistics by average confidence
- `3` - Sort statistics by total detections
- `R` - Reset all statistics

**Performance Improvements:**
- More aggressive frame skipping for better FPS
- Statistics tracking with minimal overhead
- Optimized table rendering

**Backward Compatibility:**
- All Version 1 features remain intact
- Version 1 configuration files still work
- Statistics table can be disabled if not needed
