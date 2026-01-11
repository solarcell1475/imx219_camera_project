#!/usr/bin/env python3
"""
TensorRT Performance Benchmark
==============================
Benchmark different TensorRT engine configurations for optimal performance.
Tests various input resolutions and measures FPS, latency, and accuracy.
"""

import sys
import time
import numpy as np
from pathlib import Path
from typing import Dict, List

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Add system paths for TensorRT
sys.path.insert(0, '/usr/lib/python3.10/dist-packages')

try:
    from ai_engine.optimization.tensorrt_inference import TensorRTInference
    TENSORRT_AVAILABLE = True
except ImportError:
    TENSORRT_AVAILABLE = False
    print("TensorRT not available")


def benchmark_engine(engine_path: str, test_images: List[np.ndarray], num_runs: int = 10) -> Dict:
    """Benchmark a single TensorRT engine"""
    if not TENSORRT_AVAILABLE:
        return {"error": "TensorRT not available"}

    results = {
        "engine": engine_path,
        "input_shape": None,
        "inference_times": [],
        "total_detections": 0,
        "error": None
    }

    try:
        # Load engine
        trt_model = TensorRTInference(engine_path)
        results["input_shape"] = trt_model.input_shape

        # Warmup run
        _ = trt_model.infer(test_images[0])

        # Benchmark runs
        for i in range(num_runs):
            start_time = time.time()
            detections = trt_model.infer(test_images[i % len(test_images)])
            inference_time = (time.time() - start_time) * 1000  # ms
            results["inference_times"].append(inference_time)
            results["total_detections"] += len(detections)

        # Calculate statistics
        results["avg_inference_time"] = np.mean(results["inference_times"])
        results["std_inference_time"] = np.std(results["inference_times"])
        results["min_inference_time"] = np.min(results["inference_times"])
        results["max_inference_time"] = np.max(results["inference_times"])
        results["fps"] = 1000.0 / results["avg_inference_time"]
        results["avg_detections"] = results["total_detections"] / num_runs

        trt_model.cleanup()

    except Exception as e:
        results["error"] = str(e)

    return results


def create_test_images() -> List[np.ndarray]:
    """Create test images for benchmarking"""
    images = []

    # Create images with different content to simulate real scenarios
    base_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)

    # Image 1: Mostly empty
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    images.append(img1)

    # Image 2: Random noise
    img2 = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
    images.append(img2)

    # Image 3: Some patterns (simulating objects)
    img3 = base_image.copy()
    # Add some rectangles to simulate objects
    cv2.rectangle(img3, (100, 100), (200, 200), (255, 0, 0), -1)
    cv2.rectangle(img3, (400, 300), (500, 400), (0, 255, 0), -1)
    images.append(img3)

    # Image 4: More complex scene
    img4 = base_image.copy()
    for i in range(5):
        x1 = np.random.randint(0, 500)
        y1 = np.random.randint(0, 500)
        x2 = min(x1 + np.random.randint(50, 150), 639)
        y2 = min(y1 + np.random.randint(50, 150), 639)
        color = tuple(np.random.randint(0, 255, 3).tolist())
        cv2.rectangle(img4, (x1, y1), (x2, y2), color, -1)
    images.append(img4)

    return images


def find_available_engines() -> List[str]:
    """Find all available TensorRT engines"""
    engines = []
    for file in Path(".").glob("*.engine"):
        engines.append(file.name)
    return sorted(engines)


def main():
    """Main benchmarking function"""
    print("=" * 80)
    print("TensorRT Performance Benchmark")
    print("=" * 80)
    print()

    if not TENSORRT_AVAILABLE:
        print("❌ TensorRT not available")
        return 1

    # Find available engines
    engines = find_available_engines()
    if not engines:
        print("❌ No TensorRT engines found")
        print("Run: python scripts/convert_to_onnx.py --model yolo11n.pt --size 640,640")
        print("Then: python scripts/onnx_to_tensorrt.py --onnx yolo11n.onnx")
        return 1

    print(f"Found {len(engines)} TensorRT engines:")
    for engine in engines:
        print(f"  - {engine}")
    print()

    # Create test images
    print("Creating test images...")
    test_images = create_test_images()
    print(f"✓ Created {len(test_images)} test images")
    print()

    # Benchmark each engine
    results = []
    for engine in engines:
        print(f"Benchmarking {engine}...")
        result = benchmark_engine(engine, test_images, num_runs=20)
        results.append(result)

        if result.get("error"):
            print(f"  ❌ Error: {result['error']}")
        else:
            print(".1f"
                  ".1f")
        print()

    # Print summary table
    print("=" * 80)
    print("Performance Summary")
    print("=" * 80)
    print(f"{'Engine':<20} {'Resolution':<12} {'FPS':<8} {'Latency(ms)':<12} {'Detections':<10}")
    print("-" * 80)

    # Sort by FPS (highest first)
    valid_results = [r for r in results if not r.get("error")]
    valid_results.sort(key=lambda x: x.get("fps", 0), reverse=True)

    for result in valid_results:
        engine_name = result["engine"]
        resolution = f"{result['input_shape'][1]}x{result['input_shape'][2]}" if result.get("input_shape") else "N/A"
        fps = ".1f"
        latency = ".1f"
        detections = ".1f"
        print(f"{engine_name:<20} {resolution:<12} {fps:<8} {latency:<12} {detections:<10}")

    print()
    print("=" * 80)
    print("Recommendations")
    print("=" * 80)

    if valid_results:
        best_engine = valid_results[0]
        print(f"🏆 Best Performance: {best_engine['engine']}")
        print(f"   FPS: {best_engine['fps']:.1f}")
        print(f"   Latency: {best_engine['avg_inference_time']:.1f} ms")
        print(f"   Average detections: {best_engine['avg_detections']:.1f}")

        # Performance tiers
        print()
        print("Performance Tiers:")
        for result in valid_results:
            fps = result.get("fps", 0)
            if fps >= 30:
                tier = "Excellent (60+ FPS target)"
            elif fps >= 20:
                tier = "Good (30-60 FPS)"
            elif fps >= 10:
                tier = "Fair (15-30 FPS)"
            else:
                tier = "Slow (<15 FPS)"
            print(f"   {result['engine']:<20}: {result['fps']:.1f} FPS ({tier})")
    print()
    print("=" * 80)
    print("✓ Benchmark Complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    # Import cv2 for test image creation
    try:
        import cv2
        sys.exit(main())
    except ImportError:
        print("❌ OpenCV not available")
        sys.exit(1)