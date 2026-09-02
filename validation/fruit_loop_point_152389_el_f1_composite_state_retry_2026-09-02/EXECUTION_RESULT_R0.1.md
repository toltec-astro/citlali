# SCI-FRUIT EL-F1 r1 composite-state retry result

Result: **valid primary screen; neither non-unity candidate is promising on
this compact case**

The owner-authorized test was
`SCI-FRUIT-EL-F1-R1-COMPOSITE-STATE-RETRY-R0.1`, bound by the exact
`EL_F1_R1_BUNDLE_MANIFEST_R0.1.md` with SHA-256
`574f53c4eeb1a0df818f5682eba750585918f33a6dfdc19427d72cb6a760cb50`.
This is development evidence only. It is not a method qualification or a
comparison with the unavailable exact historical executable.

## State repair established before execution

The separately checkpointed relaxed state now carries only accepted feedback
signal/kernel plus its exact method, alpha, observation, iteration, grouping,
ordered-plane, spatial-WCS, grid, and support identity. The checkpoint-bound
ordinary complete map remains the sole authority for the newest weight and
`MEDRMS` values. The relaxed state replaces only signal/kernel when the next
ordinary map is loaded; no duplicate feedback weight or RMS checkpoint fields
remain.

The experimental checkpoint schema is
`citlali-reduction-restart-checkpoint-v5-el-f1-r1`; the ordinary path remains
on v3. Focused tests cover alpha-one identity, Q-owned weight/RMS including a
literal FITS-decimal RMS value, signal/kernel replacement, fail-closed
identity/support, absence of duplicate fields, and experimental checkpoint
round trip. Direct inspection of the final alpha-1.25 injected checkpoint
shows `fruit_feedback_signal` and `fruit_feedback_kernel` and no
`fruit_feedback_weight` or `fruit_feedback_median_rms` variable.

## Frozen execution

The executable and all inputs were copied and hashed before the first run.
The exact executable SHA-256 was
`a49082dde8f71d6f50edd8c378ad94195496b5eb0e0855b746e189f3442acbcc`.
Its complete identity and the nine frozen input hashes are recorded in
`FROZEN_INPUTS_R0.1.md`. No rebuild occurred during the primary matrix.

All six fresh, single-threaded trajectories completed absolute iterations 0
through 6 on their first attempt. Every trajectory produced seven reduction
directories, used the same frozen executable, and had no error- or
critical-level log message. The alpha-1.25 and alpha-1.50 logs show a three-map
375-by-371 relaxed state at every completed iteration. The complete retained
tree was 2.4 GiB at inspection time. Original source data and original
reduction products were not modified.

| Alpha | Variant | Wall (s) | Maximum RSS (bytes) | Retained (KiB) |
| ---: | --- | ---: | ---: | ---: |
| 1.00 | control | 281.06 | 902037504 | 383596 |
| 1.00 | injected | 281.55 | 933036032 | 384052 |
| 1.25 | control | 281.12 | 945176576 | 430180 |
| 1.25 | injected | 284.57 | 928907264 | 430188 |
| 1.50 | control | 283.41 | 965836800 | 430188 |
| 1.50 | injected | 278.67 | 932200448 | 430188 |

Pair-mean wall time was 281.305 s for alpha 1.00, 282.845 s for alpha 1.25
(`+0.547%`), and 281.040 s for alpha 1.50 (`-0.094%`). These differences are
practical timing parity in this small local screen. Non-unity retained output
was about 12.08% larger because the experimental relaxed signal/kernel state
is checkpointed. `PRIMARY_EXECUTION_R0.1.csv` preserves the exact observations.

The six full logs are retained under the external experiment root's `logs/`
directory. Their SHA-256 values, in alpha/control order, are:

- alpha-1.00 control:
  `8f05a9c59782b38df48a909502df3528817957b6e1484434b5389445e1035d17`;
- alpha-1.00 injected:
  `afc6c9bad427db339371c287093478c7905e87aa39e260a082df4ae6b6c36339`;
- alpha-1.25 control:
  `f7df8b4d9496fe95d695b59a763b93eb9ae6bb7dba1a1789d017553ee691d9d1`;
- alpha-1.25 injected:
  `a8dea8404af3cf92987d2d082660c4c84274ef197583d8314bc4527dbc028780`;
- alpha-1.50 control:
  `37802dccd4d5a8417a009c395175b41d635fade8e6bb03aaea465fa0fb16cf38`;
  and
- alpha-1.50 injected:
  `dbdc96078a77c0da5aaaddda16a345b767f183e19a0f41d9652b1f5a9c3b7b16`.

## Frozen scientific result

The analyzer verified the expected iteration set, realized alpha and injected
source configuration, bitwise control/injected identity before injection,
paired and cross-iteration shape/WCS identity, and identical finite support
for signal, kernel, and weight. Every analyzed image had all 139125 pixels in
common finite support. It then produced the 54 prospective metric rows in
`ITERATION_METRICS_R0.1.csv`.

Both candidates reached the alpha-1.00 iteration-5 central recovery by
iteration 4 in every array. At iteration 6, both also passed the prospective
central-recovery error, major/minor width, centroid, full-map kernel-residual,
and a1400/a2000 annular-residual checks. Their final central recoveries were:

| Array | Alpha 1.00 | Alpha 1.25 | Alpha 1.50 |
| --- | ---: | ---: | ---: |
| a1100 | 0.972210 | 0.972687 | 0.973163 |
| a1400 | 0.978713 | 0.980134 | 0.980446 |
| a2000 | 0.964025 | 0.966833 | 0.967514 |

Neither candidate passed the frozen a1100 annular-residual check at iteration
6:

| Alpha | Annular residual / truth | Ratio to alpha-1.00 | Required |
| ---: | ---: | ---: | --- |
| 1.00 | 0.0005315938 | 1.0000 | reference |
| 1.25 | 0.0006638095 | 1.2487 | at most 1.10 |
| 1.50 | 0.0008197319 | 1.5420 | at most 1.10 |

Alpha 1.50 also showed a transient iteration-2 central-recovery overshoot in
a1100 and a1400 before settling; this was not itself a predeclared failure.
The decisive failure is the a1100 annular residual above. The exact frozen
classification is therefore `not_promising_on_this_compact_case` for both
candidates. Because neither candidate is promising, the predeclared protocol
requires no exact-restart follow-up.

The analysis initially stopped before reading a map because the newly frozen
base file name did not match an older filename heuristic. The evidence reader
was narrowly repaired to load Citlali's authoritative
`citlali_merged_config.yaml` trajectory snapshot and to reject inconsistent
multiple snapshots, with a compatibility fallback for older outputs. Six
focused tests and Ruff passed. No run, metric definition, threshold, numerical
product, or iteration selection changed. The final evidence identities are:

| Artifact | SHA-256 |
| --- | --- |
| `ANALYSIS_MANIFEST_R0.1.yaml` | `1fad74ca70a6b433c99dee77831d0d130cb5af488636089f0b925efb11629970` |
| `analyze_compact_relaxation_screen.py` | `e330f0f25ba93be54ca1c96bcf4a4d2dcf6b3149907fa58673abdde8a869073c` |
| `ITERATION_METRICS_R0.1.csv` | `90aeadeed0535668105517b963287c780e87421128591e02269495ccb6b1a1f9` |
| `SCREEN_RESULT_R0.1.json` | `bdcdec6ec7397f4a107f21997c17a2667694b5730210cba96b912ddbd757306e` |
| `SCREEN_RESULT_R0.1.md` | `4889625bb71ace446258f4399dd698549000d9241c0b8affb4e22712a6e60740` |

## Disposition

Do not promote alpha 1.25 or 1.50 from this compact screen, and do not rerun
the same fixed candidates merely to seek a different outcome. Alpha 1.00
remains the same-build reference, not a newly qualified or historically
superior production method. A later empirical proposal may use the faster
early recovery as evidence when motivating a different recurrence, damping,
or stopping hypothesis, but it requires a separately reviewed packet and must
protect residual structure explicitly.

This result does not launch Gate D or Stage B, qualify a recurrence, change a
production default, select a fallback or stopping rule, establish historical
superiority, or authorize Unity work.

## Repository verification

- focused repaired-state/config/restart C++ tests: 35 passed;
- complete CTest gate: 610 passed, zero failed, one pre-existing disabled;
- baseline and fruit-loop Python tools after the evidence-reader repair: 190
  passed;
- full configuration preflight: 127 Python tests plus all mode-kit,
  compatibility, coverage, schema, authority, and boundary audits passed;
- frozen analyzer focused tests: 6 passed;
- Ruff and `git diff --check`: passed.
