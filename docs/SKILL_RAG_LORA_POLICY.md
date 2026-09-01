# Skill, RAG and LoRA Policy

## RAG / Memory: facts that can change
Store calibration state, material behaviour, scanner settings, lighting conditions, part observations, scan history, hardware specifications and human corrections.

Examples:
- Black or glossy plastic caused tracking loss under lighting profile L03.
- Carrier C02 blocks the rear lower surface at viewpoint V07.
- The current camera-to-scan-world calibration version is CAL-YYYYMMDD-N.

## Skills: stable procedures
Store validated action structure, tool order, safety gates, success criteria and recovery routes. A skill should not hard-code changing material or calibration facts; it should retrieve those from RAG at runtime.

Examples:
- `scan_part_in_hand_v0`
- `regrip_part_carrier_v0`
- `safe_shutdown_v0`

## LoRA: promoted, repeated behaviour
Only add episodes to a LoRA candidate dataset when they are either human-approved successful runs or clearly labelled failure-to-correction runs. Do not train directly from all logs.

Promotion gates:
1. Deduplicate, anonymise if needed, validate schema and score episode quality.
2. Train a candidate adapter offline.
3. Run fixed regression tasks for tool selection, safety classification, scan-profile selection and failure reporting.
4. Test in simulation/replay and then low-speed dry run.
5. Promote only if it improves target metrics and introduces zero safety-policy violations.
6. Keep the prior adapter version for immediate rollback.

## Skill governance
Each skill has: version, owner, applicable hardware/calibration range, success rate, failure types, last-used date and retirement state. Archive or merge low-use/low-success skills.
