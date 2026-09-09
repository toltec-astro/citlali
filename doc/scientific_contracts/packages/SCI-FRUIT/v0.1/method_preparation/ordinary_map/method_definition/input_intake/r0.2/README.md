# Replacement PTC intake: single-pass mismatch resolved

2026-09-09. Status: **file identity, structure and iteration-state checks pass**.
This clears the previous collection mismatch; scientific input admission and
the weighting experiment's remaining bindings are still pending.

## Program adherence and prior-work recovery

Continue under the [charter](../../r0.4/inputs/program/README.md),
[pilot workflow](../../r0.4/inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](../../r0.4/inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md),
[frozen recovery](../../r0.4/PRIOR_WORK.md) and
[Q02 preparation](../../q02_review/r0.3/README.md). The owner reported replacing
the contents of the same supplied directory and requested another inspection.
This record supersedes the [r0.1 intake](../r0.1/README.md) for current input
state, while preserving its bytes and prior-generation hashes. No A/B, C,
method, population, gate or independent-author permission changes.

## Replacement findings

Directory:
`/Users/gwilson/work_toltec/local_data/beammaps/pointings/reduced/redu01`.

| Observation | PTC bytes | Samples / detectors / declared chunks | Recorded iteration / effective limit |
| --- | ---: | --- | --- |
| 123424 / Neptune | 760,674,540 | 3,628 / 5,905 / 12 | 0 / 1 |
| 129081 / 3c273 | 710,131,596 | 3,628 / 5,512 / 12 | 0 / 1 |

Both full files retain float64 signal and flags, sample-by-detector det_lat and
det_lon, detector/array/network IDs, per-chunk weights and chunk bounds. Signal
units are `mJy/beam`; coordinate units are radians. The directory still contains
22 files. The index timestamp is `2026-09-09.18:09:47`, with no timezone declared.
The two PTC files total 1,470,806,136 bytes. Exact new identities are in
[INPUT_MANIFEST.json](INPUT_MANIFEST.json).

The saved configuration now sets `timestream.fruit_loops.enabled: false`.
It retains `max_iters: 10`, while both output files record
`FRUITLOOPS_ITER=0`, `CONFIG.FRUITLOOPS.MAXITER=1` and an empty model path.
This is consistent with an ordinary single pass. The inactive YAML limit does
not require another export.

To resolve that requested/effective distinction, the manager inspected the
historical source matching the reported version `v4.0.0-62-ge0090e2d`:
commit `e0090e2da68ec118876aa7d3506b4dfb10ae696d`. Its PTC config reader reads
the FRUIT-specific limit only when FRUIT is enabled; `src/citlali/cli/main.cpp`
lines 422–425 explicitly force the effective limit to one when disabled.
The NetCDF writer records that effective value. The manifest binds these exact
source blobs as implementation evidence only. The embedded version is not a
verified executable-content hash or implementation-conformity result.

The legacy `CONFIG.FRUITLOOPS` scalar is 25856 and remains uninterpreted as a
Boolean. The explicit disabled YAML, zero iteration, effective limit one and
matching source handling support this disposition. All other configuration
sections captured correctly by the prior intake remain equal. PTC cleaning
still uses network grouping and rank five; map grouping is a separate fact.

## Preservation, access and remaining work

The owner replaced the old payloads at this path. The old intake still
identifies those prior bytes; it must not be checked against the new files as
though the path were immutable. No claim is made about an old export copy
elsewhere. Codex did not modify, delete or copy any supplied reduction product.

129081's scientific values remain reserved. Only schemas, scalar identity/
configuration headers and small chunk index arrays were inspected, with file
hashing for identity. Signal, flag, coordinate, weight, noise, fitted-source and
map values were not examined. No weighting statistics or comparisons were run.

No run log or provenance sidecar was found in the supplied directory or the
bounded filename search under the local `beammaps/pointings` tree. That remains
a provenance item to recover when available, not a reason to repeat this export
solely because the disabled YAML retains max_iters=10. Exact coordinate-frame
and flag meanings, chunk-support treatment, input/calibration/executable
identities and T2's evidence/uncertainty bindings still need completion before
scientific execution. A/B approval and C's open decision remain unchanged.

## Routine intake-record correction

The prior intake helper reused a variable name while building its manifest,
so r0.1's `saved_config_facts.fruit_loops` contains a navigation path instead of
the intended mapping. Its README correctly records the old enabled/10-iteration
configuration, and its independent PTC headers/hashes support the old mismatch.
This successor records the current FRUIT mapping directly from the replacement
YAML and identifies that malformed historical field without altering r0.1.
No scientific conclusion, input selection or numerical method changes.

Document, source, header and hash checks pass. All prior review/frozen packets
and both opaque review archives remain untouched. No implementation, numerical
test, replay, qualification, Unity access, transfer or push occurred.
