# FRUIT EL-F2 first-attempt input failure

Status: **invalid pre-iteration attempt; first replacement consumed**

The first scheduled trajectory was the predeclared cold-start candidate:
`alpha = 1.25`, injection disabled. It began at 16:15:21 local time on
2026-09-02 and stopped after 1.22 seconds.

The approved `COMMON_LOCAL.yaml` incorrectly pointed `kids.solver.fitreportdir`
to a nonexistent directory labeled `fitreports-unused`. Citlali emitted one
critical missing-fit-report message for each of the 12 networks and then
stopped because the gap-aligned RTC KIDs input contained NaN values.

No map FITS product and no restart checkpoint was created. The partial output
and complete outer log were preserved at:

`/Users/gwilson/work_toltec/local_data/fruit-development/point-123424/fruit-injection-development/el-f2-early-stop-r0.1/attempts/attempt-01-missing-fitreports`

Evidence:

- outer log bytes: 26,400;
- outer log SHA-256:
  `29db532f9d9e2547f8fac8a7d09336d9eae1c1423d25785596f515af51268288`;
- error/critical messages: 12, all missing-fit-report messages;
- maximum resident set size: 269,156,352 bytes; and
- retained attempt tree: 9,968 KiB at inspection time.

This is an environmental input-binding failure, not a scientific result. It
uses the first of the two replacements allowed by the r0.1 packet. The failed
output is excluded from every scientific and performance metric.
