# SCI-FRUIT v0.1 — EL-F8 Penalty-Placement Test r0.1

Decision candidate:
`SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **owner-review proposal; no implementation, build, replay, or analysis
execution is authorized**

## The short version

UID 4460 was already deweighted and then fully flagged. The large arcs appear
because the full flag is applied before shared RTC/PTC processing, where
removing one detector can change the cleaned data from many detectors.

EL-F8 asks one question:

> What changes if the same hard detector decision is applied only when the
> final map is accumulated, after shared cleaning is complete?

This is the most direct test of the owner's concern. It does not reset the
accepted FRUIT model, remove the penalty, soften its numerical strength, or
decide whether UID 4460 is a good detector.

## The one experimental change

Add one opt-in development setting,
`timestream.learning.map_pixel_outlier_detector_exclusion_application`, for
map-diagnostic detector exclusions. Its only admitted values are:

- `pre_cleaning`: the existing default, applying the exclusion before RTC and
  PTC; and
- `pre_mapmaking`: the EL-F8 setting, retaining the detector through RTC/PTC
  and fully excluding it immediately before final map accumulation.

The setting applies generically to records produced by
`mapdiag:raw_obs` with reason
`map_pixel_outlier_detector_dominance`. It must not name UID 4460 in production
logic. Busy-residual detector penalties, network penalties, sample masks,
weights, feedback selection, recurrence, injection, filters, and all other
processing remain unchanged. The default remains `pre_cleaning`.

The accepted feedback model follows the existing subtraction, residual-only
cleaning, and add-back path in both modes. EL-F8 does not reset or replace any
feedback map.

## Exact starting states

Use complete copies of the frozen EL-F5 iteration-4 directories:

- `C4`: no-injection control; checkpoint SHA-256
  `a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`;
- `A4`: off-source injected trajectory containing the UID 4460 record;
  checkpoint SHA-256
  `2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c`.

The injection remains 100 mJy/beam in every array at map-world
`(AZOFFSET, ELOFFSET) = (0, -60)` arcsec. Every replay advances exactly once,
from absolute iteration 4 to 5.

Before staging, the `A4` checkpoint hash above must be verified against the
external source. If it is not exact, stop and correct this packet rather than
substituting another state.

## Four short replays

Build one EL-F8 development executable after the opt-in setting and its tests
pass. Then run, in this order:

1. `C5-current`: `C4` with the existing `pre_cleaning` behavior;
2. `A5-current`: `A4` with the existing `pre_cleaning` behavior;
3. `C5-map`: `C4` with `pre_mapmaking` behavior; and
4. `A5-map`: `A4` with `pre_mapmaking` behavior.

The first two are compatibility gates for the new executable. Their map planes
must reproduce the existing EL-F5 `C5` and `A5` products bitwise. Every
scientific checkpoint value must also match, except the explicitly recorded
creator-version and new configuration-provenance fields. If either gate fails,
stop before the mapmaking-placement runs are interpreted.

The original EL-F5--F7 products remain read-only. All replays use isolated
copies and output directories.

## What will be measured

For all arrays, retain signal, kernel, and weight maps plus complete checkpoint
state. Report compact-source recovery, centroid, width, kernel projection,
kernel residual, real-Neptune-region RMS, annular RMS, detector participation,
and learned-state differences using the existing EL-F7 definitions.

For a1400, define:

\[
T_{\rm current}=A5-C5,
\qquad
T_{\rm map}=A5_{\rm map}-C5_{\rm map},
\]

and reuse the existing EL-F6 no-UID-4460 map `N5` to define

\[
D_{\rm current}=A5-N5,
\qquad
D_{\rm map}=A5_{\rm map}-N5,
\]

\[
Q=D_{\rm current}-D_{\rm map}=A5-A5_{\rm map}.
\]

`D_current` is the known effect of excluding UID 4460 before cleaning.
`D_map` is its effect when excluded only from map accumulation. `Q` is the
additional effect of moving that exclusion upstream of shared cleaning.

This direct interpretation is restricted to a1400 because UID 4460 is the only
carried map-diagnostic detector penalty in that array. The candidate setting
also moves two existing a2000 map-diagnostic exclusions in both paired
branches; a2000 remains an important side-effect measurement but is not
described as a UID-4460 decomposition. The a1100 busy-residual penalty retains
its current placement.

At the four original trigger pixels, report the final value, weight, leading
contributor, and leave-one-out evidence when available. Also verify from the
application log that UID 4460 has 676 participating raw samples during shared
cleaning in `A5-map` and is fully excluded before map accumulation.

## How the result will be read

- If the large Neptune/annular arcs move mainly into `Q` while `D_map` is much
  smaller and more local, early exclusion is the amplification mechanism.
- If `D_map` retains comparable broad structure, the detector's direct mapped
  contribution remains the leading explanation.
- A mixed result means both mechanisms matter.

No new numerical pass/fail threshold will be invented after the maps are
opened. The complete component maps and signed cross terms will be retained.
The result is descriptive and cannot by itself select a production policy.

## Validity and stop rules

- Source parent: Git commit `831bdf69bdf89ea1facd37630b5eaacdfa176b1f` on
  `codex/sci-fruit-v0.1-empirical-lane`.
- New local root:
  `/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1`.
- One configured thread and `--grppiex seq`.
- Exact copied checkpoint, configuration, WCS/grid, units, normalization,
  finite-support, and injection checks precede subtraction.
- The new executable must pass focused tests, all enabled CTest cases, the
  baseline/FRUIT Python tests, and the complete configuration preflight before
  it is frozen.
- At most one replacement per replay, only for an environmental interruption.
- At most 1 hour and 64 GiB per replay; 6 hours and 8 GiB retained in
  aggregate.
- Stop after the four registered replays and decomposition. Do not add a soft
  factor, confirmation counter, threshold change, new source, extra iteration,
  or second observation.

## Claim limits

EL-F8 is a local mechanism test on one exposed pointing trajectory. It cannot
judge UID 4460, validate every hard exclusion, qualify a detector policy,
select a recurrence, compare with historical Citlali, launch Gate D or Stage
B, or authorize production or Unity use. A favorable result would justify a
separate candidate-policy and genuine-failure-retention test; it would not
make `pre_mapmaking` the default.

## Owner choices

### Choice A — Approve the placement decomposition (recommended)

Approve
`SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1` exactly against its
bundle manifest. This authorizes only the bounded opt-in development setting,
tests, local build, four one-iteration replays, analysis, and result record
described above.

### Choice B — Accept the mechanism explanation only

Record the read-only explanation, but do not implement or run EL-F8.

### Choice C — Revise the test

Return a new packet with a different intervention, controls, measurements, or
bounds. Nothing in Choice A is authorized.

General permission to continue is direction to prepare this packet, not exact
approval of Choice A.
