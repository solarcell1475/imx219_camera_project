# Installing PyTorch with CUDA Support for Jetson

## Problem
The conda PyTorch installation is CPU-only and doesn't support CUDA on Jetson. You need to install PyTorch with CUDA support to leverage the 67 TOPS GPU.

## System Information

Your system:
- **JetPack:** R36 (release 4.7) - JetPack 6
- **CUDA:** 12.6
- **Python:** 3.10.19 (in `yolo_gpu` conda environment)
- **GPU:** 67 TOPS NVIDIA GPU

## Solution for JetPack 6

### Step 1: Activate the GPU Environment

```bash
conda activate yolo_gpu
```

### Step 2: Install PyTorch with CUDA Support

**For JetPack 6 (R36), PyTorch wheels may need to be obtained from NVIDIA forums:**

1. **Visit NVIDIA Forums:**
   - Go to: https://forums.developer.nvidia.com/t/pytorch-for-jetson/72048
   - Look for PyTorch wheels compatible with JetPack 6 (R36) and Python 3.10

2. **Download the PyTorch wheel:**
   ```bash
   # Example (update URL with actual link from forums):
   cd /tmp
   wget --no-check-certificate <WHEEL_URL> -O torch-2.x.x-cp310-cp310-linux_aarch64.whl
   ```

3. **Install PyTorch:**
   ```bash
   conda activate yolo_gpu
   python -m pip install /tmp/torch-*.whl
   ```

4. **Install torchvision (if available):**
   ```bash
   # Download torchvision wheel from same forum thread
   python -m pip install /tmp/torchvision-*.whl
   ```

### Step 3: Set Up Library Paths

Run the setup script to configure library symlinks:

```bash
cd /home/jetson/Downloads/IMX219_Camera_Project/ai_vision_system
bash scripts/setup_pytorch_libs.sh
```

### Step 4: Verify Installation

```bash
conda activate yolo_gpu
python verify_gpu.py
```

Or manually:
```bash
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

You should see:
- `CUDA available: True`
- `GPU: [Your GPU name]`

## Alternative: If Official Wheels Not Available

If PyTorch wheels for JetPack 6 are not yet available:

1. **Check for JetPack 5.x compatibility:**
   - Some JetPack 5.x wheels may work with JetPack 6
   - May require additional library symlinks (handled by setup script)

2. **Use system Python (if conda environment has issues):**
   ```bash
   # Install in system Python instead
   pip3 install --user /tmp/torch-*.whl
   ```

3. **Wait for official JetPack 6 support:**
   - Monitor NVIDIA forums for official JetPack 6 PyTorch wheels
   - Check NVIDIA Developer Downloads periodically

## Troubleshooting

### Library Not Found Errors

If you see errors like `libcublas.so.10: cannot open shared object file`:

1. Run the library setup script:
   ```bash
   bash scripts/setup_pytorch_libs.sh
   ```

2. Ensure CUDA is in library path:
   ```bash
   export LD_LIBRARY_PATH=/usr/local/cuda-12.6/lib64:$LD_LIBRARY_PATH
   ```

### PyTorch Installed but CUDA Not Available

1. Verify CUDA installation:
   ```bash
   nvcc --version
   ```

2. Check PyTorch installation:
   ```bash
   python -c "import torch; print(torch.__version__)"
   ```

3. Ensure you're using the correct Python environment:
   ```bash
   which python
   conda activate yolo_gpu
   ```

## Expected Performance

With GPU acceleration enabled:
- **Inference speed:** 30-60ms per frame (vs 300-400ms on CPU)
- **FPS:** 15-30+ FPS (vs ~1 FPS on CPU)
- **Speedup:** 10-30x faster than CPU-only

## Next Steps

After PyTorch with CUDA is installed:
1. Install YOLO dependencies: `pip install ultralytics opencv-python`
2. Run verification: `python verify_gpu.py`
3. Test YOLO system: `python main.py`
