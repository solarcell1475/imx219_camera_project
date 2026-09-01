# Cursor Hardware Development Handoff

## Mission
Implement a safe, local scan cell that digitises model parts/runners with measurable geometry accuracy of 0.5 mm or better and separately captured RGB colour reference.

The physical rule is non-negotiable:

> reBot holds the part through a carrier. Seal Lite and IMX219 remain fixed and calibrated.

Do not begin with the arm holding the scanner. Stability of the scanner/camera coordinate frame is more valuable than extra motion flexibility for the first milestones.

## Existing hardware
- reBot Arm B601-DM.
- IMX219 stereo camera.
- Jetson Orin Super 8 GB x2, 1 TB SSD each.
- 3DMakerpro Seal Lite.
- Mac Studio M5 Ultra, 256 GB unified memory.
- Motorised stage / rotator options.

## Compute and network topology

```text
Mac Studio M5 Ultra
  Hermes + local LLM/VLM + RAG + data curation + offline mesh/LoRA evaluation
        |
        | Ethernet / high-level job API only
        |
Jetson A (robot edge)                    Jetson B (vision edge)
  ROS 2 + reBot driver                     IMX219 stereo capture
  MoveIt + planning scene                  OpenCV / segmentation
  stage driver                             blur / framing / visibility checks
  safety watchdog                          episode capture/logger
        |                                       |
        +------------ scan cell ---------------+
                         |
             reBot holds carrier/part
             Seal Lite fixed, IMX219 fixed
```

Mac has no direct motor authority. Jetson A owns command execution. Hardware E-stop and protective stopping remain outside all LLM control.

## Safety non-negotiables
- Install and verify an independent physical E-stop before any autonomous movement.
- Begin all work in a restricted, low-speed mode with a clear physical exclusion zone.
- Software must fail closed: sensor, calibration, storage or lifecycle failure means `safe_park`, never an improvised retry.
- Do not allow the LLM to issue torque commands, edit joint limits, release E-stop, or emit unvalidated joint trajectories.
- Use collision checking and approved named poses. The first robot motion work is only `safe_home` and `safe_park`.

## First physical layout
- Rigidly mount Seal Lite. Provide USB cable strain relief.
- Rigidly mount IMX219 stereo camera. Use fixed diffuse LED lighting.
- Position a regrip dock / safe rest inside an accessible but collision-checked arm zone.
- Arm grips a carrier/pallet, not fragile raw model plastic wherever possible.
- Keep arm motion near the base and low acceleration during capture.
- Place a calibration board/target and a colour target in repeatable positions.

## Required software deliverables

### 1. ROS 2 package layout
Create a workspace structure similar to:

```text
ros2_ws/src/
  scan_cell_bringup/
  scan_cell_interfaces/
  scan_cell_robot/
  scan_cell_vision/
  scan_cell_capture/
  scan_cell_safety/
  scan_cell_calibration/
  scan_cell_mock_hardware/
```

Use lifecycle-managed nodes for robot, vision, capture and storage services. Each must expose clear readiness/error state.

### 2. Named poses and approved profiles
Start with named poses only:

```text
safe_home
safe_park
safe_scan_start
V01 ... V12
regrip_dock_a
regrip_dock_b
```

A profile defines only approved viewpoint IDs and constraints. Store profiles in versioned YAML/JSON; do not accept model-generated raw pose values.

### 3. Capture job contract
Implement a job API that accepts an approved `profile_id` and returns `task_id`.

```text
request:  run_scan_job(profile_id, part_id, operator_approved)
response: task_id
```

Every task creates one immutable job folder:

```text
jobs/<task_id>/
  metadata.json
  rgb/left/
  rgb/right/
  raw_scans/
  mesh/
  logs/
  reports/
```

Use the schema in `docs/EPISODE_SCHEMA.md`. Do not commit raw images, scans, ROS bags, private models or secrets to Git.

### 4. `scan_part_in_hand_v0`
Read `docs/SCAN_PART_IN_HAND_V0.md`. Implement as a state machine, not as a free-form LLM conversation.

Minimum state sequence:

```text
PREFLIGHT
  -> VERIFY_PART
  -> SAFE_SCAN_START
  -> MOVE_VIEWPOINT
  -> SETTLE
  -> GEOMETRY_CAPTURE
  -> RGB_CAPTURE
  -> NEXT_VIEWPOINT
  -> QUALITY_CHECK
  -> RESCAN_ONCE | REGRIP_REQUEST | COMPLETE | SAFE_PARK
```

All `FAILED` paths must transition to `SAFE_PARK` or a hardware stop, then log the cause.

## Milestone order

### M0 — Bench foundation
1. Confirm reBot driver/control interface and execute 20 safe home/park cycles.
2. Confirm both IMX219 sensors enumerate reliably after reboot.
3. Capture timestamped stereo frames with locked exposure/white balance.
4. Confirm Seal Lite capture/export workflow, initially manually if it has no stable local SDK.
5. Implement hardware-health and disk-space preflight checks.
6. Verify E-stop and protective stop behavior.

### M1 — Geometry benchmark
1. Select a non-transparent, low-reflectivity 50–150 mm calibration/test part.
2. Capture ten complete geometry runs using fixed scanner and controlled object poses.
3. Measure ten physical features with calipers.
4. Generate a report: 95% of features must be <= 0.5 mm error; repeatability goal <= 0.25 mm.
5. Do not proceed to Hermes autonomy until this benchmark exists.

### M2 — Automated viewpoint workflow
1. Implement 8–12 approved viewpoint motions.
2. At each pose: settle 800 ms, trigger geometry capture, capture RGB stereo frames, append metadata.
3. Add basic coverage/blur/framing checks.
4. Add one approved rescan cycle only.

### M3 — Regrip workflow
Implement regrip docking for occluded geometry. First implementation can require operator confirmation and manual regrip; automation comes later.

### M4 — Hermes integration
Hermes receives only high-level tools such as `run_scan_job`, `get_job_status`, `get_quality_report`, `safe_stop`, and `request_operator_check`. It may query RAG for material/capture facts but cannot bypass M0–M3 safety constraints.

### M5 — Data flywheel
Use only operator-approved success episodes and labelled correction episodes for the dataset queue. LoRA candidates must pass offline evaluation, replay and low-speed dry run before promotion.

## Quick validation commands to add
- `scan_cell_bringup --mock`
- `scan_cell_safety_check`
- `scan_cell_camera_test --left --right`
- `scan_cell_robot_home_test --cycles 20 --speed low`
- `scan_cell_capture_job --profile part-small-v0 --dry-run`
- `scan_cell_report --task-id <id>`

## Documentation to read first
- `docs/ROBOTIC_SCAN_CELL_ARCHITECTURE.md`
- `docs/SCAN_PART_IN_HAND_V0.md`
- `docs/EPISODE_SCHEMA.md`
- `docs/SKILL_RAG_LORA_POLICY.md`
- `docs/ROADMAP_M0_TO_M5.md`

## Definition of Done for the first PR after handoff
- No real autonomous capture motion.
- Mockable ROS 2 state-machine skeleton exists.
- A versioned job folder and metadata writer exist.
- Camera test is reproducible on both Jetsons.
- Robot safe-home/safe-park test is bounded and has an E-stop test procedure.
- Setup instructions are reproducible on a clean Jetson install.
