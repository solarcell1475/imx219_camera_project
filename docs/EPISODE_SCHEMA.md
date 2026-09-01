# Episode Metadata Schema

Every scan task writes one immutable job directory and one episode record.

```json
{
  "task_id": "scan-YYYYMMDD-HHMMSS-001",
  "skill_id": "scan_part_in_hand_v0",
  "skill_version": "0.1.0",
  "status": "success | partial | failed | safety_stop",
  "operator_approved": true,
  "timestamps": {"started_at": "ISO-8601", "ended_at": "ISO-8601"},
  "hardware": {
    "robot": "reBot B601-DM",
    "scanner": "3DMakerpro Seal Lite",
    "camera": "IMX219 Stereo"
  },
  "calibration": {
    "robot_base_to_scan_world": "CAL-id",
    "carrier_tcp": "TCP-id",
    "camera_to_scan_world": "CAMCAL-id"
  },
  "capture_profile": "part-small-v0",
  "viewpoints": [
    {
      "viewpoint_id": "V01",
      "robot_pose_ref": "pose reference or file",
      "settle_ms": 800,
      "scan_status": "ok | failed",
      "rgb_left": "rgb/left/V01.png",
      "rgb_right": "rgb/right/V01.png",
      "quality": {"blur_score": 0.0, "coverage_increment": 0.0}
    }
  ],
  "results": {
    "coverage_percent": 0.0,
    "geometry_acceptance": "pass | fail | pending_measurement",
    "texture_acceptance": "pass | fail | pending"
  },
  "artifacts": {
    "raw_scans_dir": "raw_scans/",
    "mesh": "mesh/part.obj",
    "ros_bag": "logs/task.bag",
    "quality_report": "reports/quality.json"
  },
  "human_correction": null,
  "safety_events": []
}
```

Never commit raw customer/model image assets, ROS bags, scanner data or secrets to Git by default. Store only schemas, scripts, calibration templates and redacted sample metadata.
