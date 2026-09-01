# Robotic Model Scan Cell Architecture

## Objective
Build a local, safety-first scan cell for Gundam/model parts and runners. The system targets measured geometry accuracy of 0.5 mm or better and captures separate RGB data for original-colour texture reconstruction.

## Hardware
- reBot Arm B601-DM: holds a carrier/pallet containing the model part; changes approved viewpoints and performs regrip.
- 3DMakerpro Seal Lite: fixed on a rigid mount; primary geometry capture sensor.
- IMX219 stereo camera: fixed RGB/stereo reference camera; quality control, part localisation and RGB texture capture.
- Jetson Orin Super 8 GB #1: ROS 2, reBot control, MoveIt 2, stage/control integration and safety watchdog.
- Jetson Orin Super 8 GB #2: IMX219 capture, OpenCV/segmentation, image-quality checks and capture logging.
- Mac Studio M5 Ultra 256 GB: Hermes agent, local LLM/VLM, RAG, dataset curation, mesh processing, TRELLIS.2 experiments and LoRA evaluation.
- Motorised stage/rotator: optional controlled yaw motion; initially used only when it improves visibility without disturbing the robot-held part.

## Scan-cell principle
Keep Seal Lite and IMX219 fixed. The scanner/camera frame is the measurement world frame. reBot holds the part carrier and presents it to a validated viewpoint library. This separates sensor stability from object-pose control.

## Geometry and colour are separate passes
1. Geometry pass: Seal Lite captures multi-view geometry after the arm has stopped and settled.
2. RGB pass: IMX219 captures a synchronized, fixed-exposure/fixed-white-balance image set at the same viewpoint IDs.
3. Fusion: the offline pipeline aligns scans, produces a mesh, and projects/calibrates RGB images onto that mesh.

Seal Lite is used as the geometry source. Original-colour quality is a camera/lighting/calibration problem, not a scanner-accuracy claim.

## Safety boundary
- Hardware E-stop and power/safety chain override every process.
- Hermes/LLM may select only approved high-level jobs and may request a safe stop.
- Hermes/LLM must not command motor torque, alter joint limits, release an E-stop, or create unvalidated raw trajectories.
- MoveIt 2, collision checking, controller limits and a watchdog remain between agent intent and robot motion.

## First acceptance target
For a 50–150 mm, non-transparent, low-reflectivity test part:
- Run at least 10 complete captures.
- Measure at least 10 physical reference features with calipers.
- At least 95% of measured feature errors must be <= 0.5 mm.
- Repeatability of the same feature across runs should be <= 0.25 mm.
- Save raw scan data, RGB frames, viewpoint IDs, robot pose, calibration versions, logs and result quality for each run.
