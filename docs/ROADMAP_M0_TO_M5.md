# Robotic Scan Cell Roadmap

## M0 — Bench and safety foundation
- Build rigid fixed mounts for Seal Lite and IMX219.
- Install E-stop, cable strain relief, lighting and regrip docks.
- Bring up reBot, IMX219, scanner connectivity and logging.
- Acceptance: arm can execute 20 safe home/park cycles; stop chain works; camera/scanner capture scripts work.

## M1 — Manual geometry benchmark
- Use a fixed scanner and a simple calibration part.
- Capture at least 10 scans and compare 10 features to caliper measurements.
- Acceptance: 95% of features <= 0.5 mm error.

## M2 — Automated fixed viewpoint capture
- Implement `scan_part_in_hand_v0` with 8–12 approved poses.
- Persist all metadata and quality checks.
- Acceptance: one command produces a complete job directory without unsafe motion.

## M3 — Regrip and full accessible coverage
- Add carrier/dock workflow to expose previously occluded areas.
- Acceptance: two-sided scan is repeatable and arm avoids scanner, lights, camera and dock.

## M4 — Hermes orchestration
- Hermes can select approved jobs, retrieve RAG facts, call high-level actions and produce reports.
- Acceptance: Cantonese request triggers an approved job; failures cause safe stop and report, not autonomous improvisation.

## M5 — Data flywheel and LoRA evaluation
- Curate approved episodes and create a fixed eval suite.
- Train candidate adapters only after data quality gates.
- Acceptance: promoted adapter improves selected planning/classification metrics with zero safety-policy violations in replay and low-speed dry run.
