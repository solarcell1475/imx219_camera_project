# Agent brief: IMX219 camera workspace (isolated local runtime)

You are the local Cursor agent for the **IMX219 stereo camera project**.
This is a **new, isolated workspace**. Do **not** touch COMSOL-HFSS.

## Machine and paths

- Host: `solarstatiion-Titan-18-HX-A14VIG` (local Ubuntu laptop)
- This workspace folder:
  `/home/solarstatiion/Documents/workspace/imx219_camera_project`
- Sibling folder you must **never** edit:
  `/home/solarstatiion/Documents/workspace/COMSOL-HFSS`
- GitHub:
  https://github.com/solarcell1475/imx219_camera_project
- Default branch: `main`

This laptop is for **code, docs, and repo work**.
The cameras and TensorRT stack run on a **Jetson Orin Nano**, originally at:
`/home/jetson/Downloads/IMX219_Camera_Project/`

Do not assume `/dev/video*`, NVArgus, or Jetson overlays exist on this Titan laptop.
Do not mix files, git remotes, branches, or PRs with COMSOL-HFSS.

## Confirm you are in the right place

Before changing anything:

1. `pwd` must be `/home/solarstatiion/Documents/workspace/imx219_camera_project`
2. `git remote -v` must show `solarcell1475/imx219_camera_project`
3. `git status -sb` should be on `main`
4. Refuse to work if the cwd is inside `COMSOL-HFSS`

## What this repo already has

Waveshare **IMX219-83** stereo camera project for Jetson Orin Nano (JetPack R36).

Already committed on `main` before this handoff:

- Camera bring-up scripts:
  - `activate_imx219_cameras.sh`
  - `test_imx219_cameras.sh`
  - `imx219_camera_test.py`
- Docs: `README.md`, `IMX219_CAMERA_SETUP.md`, `DEVELOPMENT_LOG.md`, `QUICK_REFERENCE.md`
- `ai_vision_system/`: dual-camera YOLO11n, ONNX, TensorRT, async processor
- `depth_sensing/`: calibration + stereo depth scripts
- Latest commit `fe49fab` (2026-01-11): claimed TensorRT jump from ~0.2 FPS to **76.8 FPS**

## First job (complete this setup)

1. Inspect the repo and summarize current status in plain language.
2. Keep all work inside this folder and this GitHub repo only.
3. Create a short local status note at:
   `LOCAL_UBUNTU_RUNTIME.md`
   Include:
   - this Titan Ubuntu path
   - that it is isolated from COMSOL-HFSS
   - that Jetson hardware is required for camera/TensorRT runs
   - next useful engineering steps
4. Do **not** rewrite history, force-push, or change the GitHub remote.
5. Do **not** copy COMSOL/HFSS/PCM files into this repo.

## Then continue the real camera work

After the status note, continue from the last useful state of this repo.

Priority order:

1. Read `README.md`, `DEVELOPMENT_STATUS.md`, `ai_vision_system/START_HERE.md`, `ai_vision_system/PERFORMANCE_OPTIMIZATION_REPORT.md`, and `depth_sensing/README.md`.
2. Identify what is real vs claimed (especially the 76.8 FPS TensorRT result).
3. Clean up anything that still hard-codes `/home/jetson/...` so the repo can be used from this Ubuntu workspace **and** the Jetson.
4. Make the next engineering slice one of these, in this order, unless I paste Perplexity notes that say otherwise:
   1. Reliable dual-camera bring-up / test path
   2. Stereo calibration workflow
   3. Depth sensing that uses real calibration
   4. Dual-camera YOLO11 + TensorRT path that can fall back cleanly
5. Keep changes small, commitable, and documented.

## Constraints

- Isolated workspace only. No COMSOL-HFSS.
- Prefer fixing/extending existing scripts over creating a new parallel stack.
- If a step needs Jetson hardware, sudo, reboot, or `/dev/video*`, write the exact commands for me to run on the Jetson. Do not fake camera results on this laptop.
- If you need the long Perplexity discussion, ask me to paste it. Do not invent that conversation.
- When you finish the first job, tell me:
  - what you found
  - what you changed
  - what I must run on the Jetson
  - what you will do next

## Start now

Confirm the workspace, inspect the repo, write `LOCAL_UBUNTU_RUNTIME.md`, then start the first engineering slice.
