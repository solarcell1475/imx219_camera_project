# Standard Robotic Skill Flow

```text
S1 Receive task
  -> S2 Route to approved skill
  -> S3 Operator confirmation where required
  -> S4 Bring ROS 2 nodes to active lifecycle state
  -> S5 Read environment + calibration snapshot
  -> S6 Readiness failure? safe park + operator intervention
  -> S7 Safety gate (E-stop / protective stop / workspace)
  -> S8 Motion-envelope and collision check
  -> S9 Unsafe? halt and log incident
  -> S10 Generate validated task plan
  -> S11 Controlled execution with watchdog
  -> S12 Verify geometry/RGB/task output
  -> S13 Write immutable episode log
  -> S14 Update RAG facts / candidate data queue
  -> S15 Promote, revise or archive skill only after evaluation
```

This flow is deliberately safety-first: the LLM operates above S10; deterministic motion planning, controllers and hardware safety own all actions below the policy boundary.
