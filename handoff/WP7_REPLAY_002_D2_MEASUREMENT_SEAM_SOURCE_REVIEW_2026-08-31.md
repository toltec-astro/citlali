# WP7-REPLAY-002 D2 Measurement-Seam Source Review

Date: 2026-08-31

Status: **source captured; conformance review failed; no canonical candidate
created**

## Bounded work order

`WP7-REPLAY-002` authorized an immutable four-path capture of the pending D2
measurement prototype, an independent review against canonical D1/D2 and
architecture authority, and reconstruction from canonical base only if that
review passed. It did not authorize import of the preserved producer stack,
production routing, persistent TOD, a common analysis grid, filter design,
factor selection, sample removal, or downsampling.

## Exact identities

- Canonical base reviewed: `4dc7844e59e03cf2d18a9262fe5b75d3ff078681`
- Divergent prototype base: `49fe73e757daa1885cd23127e8441cba47e648d2`
- Preservation branch: `codex/wp7-d2-producer-source-capture`
- Exact four-path capture commit:
  `916fa07600cf6c5e9ea7317a396fdce160a6c419`
- Exact capture tree: `d24aa93521cf4456ba00901e64fa6093cdc266b7`
- Exact capture parent: `49fe73e757daa1885cd23127e8441cba47e648d2`

The capture commit changes exactly:

| Path | SHA-256 |
| --- | --- |
| `tests/CMakeLists.txt` | `45bcb296238a509c368dd9397184a514f59e4b42f95669e39f58b68920ccbd74` |
| `include/citlali/core/pipeline/rtc_filter_d2_measurement.h` | `d7d8ec9f1ac60aac924ff24fb11e52ef1d17dd2876a1eeb4b4082aa11c0fea9b` |
| `tests/rtc_filter_d2_measurement_header.cpp` | `0cbc6981cad7da2c18a61f251a64d9795ce0363e651e0e2020872b5c0b22df65` |
| `tests/test_rtc_filter_d2_measurement.cpp` | `4cab5154b3a370ab0b34c03e24e31dfb66fb7d6b459f6637a9b5eadcaf881841` |

`cmp` verified all four capture paths byte-for-byte against the original dirty
worktree before commit. The original worktree remains at exact base
`49fe73e757...` with the same one modified and three untracked source paths.

## Review result

The source does **not** pass the reconstruction gate.

### R1 — canonical dependency boundary is not satisfied

The proposed public header directly includes
`citlali/core/pipeline/paired_readout.h`. That product does not exist at the
canonical base. Canonical D1 instead contains
`timestream_measured_scan.h`, whose ownership, member model, validity model,
scan/chunk identity, and relation to common-slot state differ materially from
the captured `PairedReadout` interface.

`PairedReadout` was introduced on the divergent line by `6b5220b74` and then
changed together with divergent timing/identity and network-timed RTC work.
Importing it would therefore be a broader producer-architecture replay, not a
selective reconstruction of this four-path measurement seam. Rewriting the
seam against `NativeMeasuredDetectorScan` would be a new architectural choice,
not a mechanical conformance repair. Both actions are outside this work order.

### R2 — the exact captured source does not build its focused target

The command

```text
cmake --build build --target citlali_wp7_timestream_test -j 8
```

compiled the isolated public-header translation unit, then failed while
compiling `tests/test_rtc_filter_d2_measurement.cpp`:

1. line 91 copy-initializes the explicitly constructed
   `NativeObservationScope`; and
2. line 223 calls nonexistent
   `NativeSampleIdentity::packet_counter()` instead of obtaining the counter
   from the native timing authority.

These are repairable source defects, but the dependency/authority finding R1
still blocks canonical reconstruction.

### R3 — source-mask detector and route identity are incomplete

`RtcFilterD2SourceMask::admit` binds an occurrence-axis handle, detector
count, free-form policy string, disposition, and dense byte mask. It does not
bind the exact detector-axis handle or ordered detector identities. A mask
with the right timing pointer and cardinality can therefore be interpreted
under a different detector ordering. The free-form policy string also does
not bind an approved Science, OOF, or Beammap route/profile identity, so the
prototype cannot enforce the required route-specific source-mask state.

### R4 — line evidence is not bound to its native evidence scope

`RtcFilterD2LineMask` has no observation, network, run, occurrence-axis, or
detector-axis binding. A line declaration can consequently be attached to a
foreign network plane. In addition, `applied` admits an empty interval set and
admits intervals with `effective_before_decimation == false` and no operator
evidence. That can represent pending evidence, but it cannot by itself furnish
the realized pre-decimation operator evidence required to advance D2.

### R5 — exact scan/chunk/run and residual-invalidity provenance are absent

The plane retains the observation-level `PairedReadout` and can derive packet
runs, but neither the plane nor its masks bind an exact scan/chunk/run scope.
The residual adds one compact validity byte per cell but no typed or stable
reason for a newly invalid residual. This is insufficient to claim the exact
D1 run/validity provenance promised by the bounded proposal.

## Positive properties retained as design evidence

The capture does contain useful bounded ideas: prefilter numeric values are a
zero-copy view, the residual owns only its derived numeric plane and compact
additional validity, network axes are not resampled or merged, common-grid and
sampling changes are rejected, and memory evidence distinguishes owned from
referenced storage. Those properties are evidence for a future design; they
do not overcome the failed identity and canonical-dependency gates.

## Stop boundary and owner decision required

No branch named `codex/wp7-g4-replay-002` was created, no source was applied to
canonical base, no governance record or mainline file was changed, and no
integration is proposed.

The next move requires a new owner-reviewed architecture choice between:

1. first replaying/reviewing a bounded canonical network-native paired D1
   producer contract, after which a repaired D2 seam could bind to it; or
2. explicitly authorizing a new D2 observer design against canonical
   `NativeMeasuredDetectorScan`, with the changed signal/member, validity,
   route, scan/chunk/run, unit, and detector-axis contracts stated before
   implementation.

Neither choice is implied by this failed review.
