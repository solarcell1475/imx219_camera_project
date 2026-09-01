# scan_part_in_hand_v0

## Purpose
Safely capture multi-view geometry and RGB reference images of a model part or runner held through a carrier by the reBot B601-DM.

## Preconditions
- Hardware E-stop is released and all safety checks pass.
- Robot controller, camera node, scanner connection and logging/storage are healthy.
- Active calibration versions exist for robot base, tool/carrier, camera and scan world.
- Part is secured in a validated carrier; the part does not flex or occlude critical surfaces excessively.
- An approved viewpoint profile exists for the part size and carrier type.

## Procedure
1. Create `task_id` and capture the complete preflight snapshot.
2. Use IMX219 to verify carrier presence, part integrity and approximate centring.
3. Move reBot to `safe_scan_start` at low speed.
4. For each approved viewpoint ID:
   - Plan with collision checking.
   - Move at controlled speed.
   - Hold and settle for 0.8 seconds.
   - Capture Seal Lite geometry data.
   - Capture timestamped left/right IMX219 RGB frames with locked exposure and white balance.
   - Store robot pose, viewpoint ID and quality indicators.
5. Run coverage and image-quality checks.
6. If coverage is incomplete, run only the approved missing-area viewpoints.
7. If the carrier masks a required surface, place it into the regrip dock, request/operator-confirm regrip, then run the reverse-side profile.
8. Save the episode package and return the arm to `safe_park`.

## Completion checks
- No E-stop/protective-stop event occurred.
- Required viewpoint IDs were captured.
- No critical blur, tracking failure or missing metadata.
- Coverage meets profile threshold, initially 98% of accessible surface.
- Output package contains raw scans, RGB frames, metadata, ROS logs and quality report.

## Failure policy
- Safety anomaly: stop immediately; do not retry autonomously.
- Sensor/lifecycle failure: safe park; log fault; request operator intervention.
- Coverage failure: one approved rescan cycle maximum; then request operator review.
- Collision/path failure: do not invent a new trajectory; use an approved alternate pose or stop.
