# POINT 152389: earlier offsets and whole-network failures

Owner-requested follow-up, 2026-09-22. Read-only scientific diagnosis against
source `912787f62862132295e0dddab660a6fc398e28c7` on
`codex/map-successor-first-forward-001`. Original MAP products are unchanged.

## Offset comparison

The earlier table is `v1/point/pointings/reduced/redu00/152389/raw/ppt_commissioning_pointing_152389_citlali.ecsv`.
Its axes are azimuth/elevation offsets. The delivered successor MAP axes are
FK5 J2000 tangent east/north. Comparing their numerical pairs directly is invalid.

All pairs below are **azimuth/elevation arcseconds**. The successor columns
are diagnostic regrids at 1 arcsec per pixel, using the exact retained PTC
signals, eligibility, networks and uniform coefficients. They are not rerun
reductions or replacement science products.

| Array | Earlier published fit | Successor coordinates rotated per sample | Same successor signals with earlier APT geometry and earlier altaz pointing calculation |
|---|---:|---:|---:|
| a1100 | (+5.083, +0.117) | (+7.038, −0.504) | (+5.216, +0.301) |
| a1400 | (+5.051, −0.204) | (+6.979, −0.407) | (+5.268, −0.289) |
| a2000 | (+4.789, −0.373) | (+6.660, −0.634) | (+4.732, −0.380) |

The final column differs from the earlier published positions by 0.227, 0.234,
and 0.057 arcsec respectively. The larger difference is mostly explained by
changed coordinate inputs, especially the detector-position APT. This does not
establish that either calibration table or pointing convention is scientifically
preferable, or attribute the remaining subarcsecond differences to one cause.

A stepwise control using the earlier APT but retaining the successor boresight
trajectory gives (+4.545,+0.102), (+4.769,−0.569), (+3.756,−0.264) arcsec.
Thus the APT change and the change between celestial and legacy altaz boresight
calculations both matter. A simple constant rotation alone is insufficient.
The current selected APT is the v2 canonical matched artifact; the earlier run
uses `v1/point/apts/apt_152389_matched.ecsv`. Their geometry differs even when
joined by the same observation/network/raw channel. Near the central source,
median new-minus-old rotated detector displacements are approximately
(+2.35,−0.56), (+2.17,−0.29), (+2.89,−0.34) arcsec. This comparison does not
assume that the old and new matching decisions select the same physical detector.
It measures the geometry assigned to the same recorded signal channel.

The original v1/v2 telescope files are byte-identical, SHA-256
`2aec5f1f2be8c365127fcf275a02eb0c780eeda9494cd225e50e50c4386a9392`.
The earlier run's config selects a recomputed telescope file, SHA-256
`377a81c754ee8b6516f31d7825323ff590eba3e504ac7144bd5b75e1dfaf3632`.
Its raw altaz fields used here are identical to the original file; its
SourceRaAct/SourceDecAct and parallactic angle differ. Both reductions configure
zero additional azimuth/elevation pointing offsets and use the same source center.
No pointing offset was added to force agreement.

Validation of the diagnostic:

- Reconstructed every retained successor AST coordinate from its exact native
  schedule, selected v2 geometry and existing AST equations. Maximum discrepancy
  is 1.405e-10 arcsec. The new coordinates were also retained directly in the MAP
  bundle; this comparison checks the diagnostic's reconstruction.
- Preserved all 13,453,121 / 6,719,154 / 1,402,249 contributions. None was omitted
  for missing earlier detector geometry. Unavailable successor samples remain
  unavailable; nw2/nw11 are not restored by this diagnostic.
- Fit the earlier maps with the same unweighted elliptical Gaussian + constant
  used for the successor description. Results (+5.078,+0.111),
  (+5.064,−0.205), (+4.803,−0.377) arcsec closely reproduce their published fits.
  The roughly 2 arcsec difference is not a fitter-coordinate indexing error.
- Diagnostic regrids use 1 arcsec nearest-pixel bins and finite hit support;
  the delivered MAP retains 2 arcsec pixels and its 0.5 normalization support cut.
  Signals, cleaning, calibration, network membership and coefficient family are
  frozen, but these regrids do not reproduce the legacy reduction's weights,
  cleaning, membership, gridding kernel or flux scale.

`coordinate-comparison.json` contains full precision and intermediate results;
`regrid.py` reproduces them. `offset-comparison.png` shows the positions with each
panel normalized to its central peak. This is a centroid investigation, not
flux-transfer or PSF qualification.

## Networks 2 and 11

Both network workers failed at `RtcJumpExclusionPlan::consider`, before RTC Apply,
CAL, PTC or their downstream VAL decisions. The exception is:

`RTC admitted transition has no existing scan support`

| Network | Selected science rows (zero-based, half-open) | Native tail outside those scans | Admitted unbound events |
|---|---|---|---|
| 2 | [33,7289) | [7289,7699) | 2: channel 246 [7448,7451), channel 405 [7297,7302) |
| 11 | [32,7288) | [7288,7699) | 15: channels 0,1,2,3,4,5,7,8,9,11,12,13 around [7366,7371); channel 17 [7368,7370); channels 483,485 [7369,7371) |

Six forced 10-second processing intervals cover the selected science domain.
RTC learns events over the full native input, including the roughly 3.36-second
tail. The first chronological tail event is about 0.066 seconds beyond the last
science interval for nw2 and 0.639 seconds for nw11. These are *admitted transition
supports*, not merely candidate seeds or preliminary recovery windows. Whether a
specific admitted event is a physical instrument jump is not independently
established by this audit.

The jump planner iterates each admitted coordinate and requires at least one
scan overlap. It has no representation for the distinction between **known
outside the requested processing domain** and **scan relation unavailable**.
The first non-overlap throws. The CLI catches that exception at the network
worker boundary and returns failure, so no completed network product reaches
CAL/PTC/MAP. Network 2 first throws on channel 246 (detector order), not on the
chronologically earlier channel 405 event; network 11 first throws on channel 0.

There are 19 admitted groups on 18 detectors in nw2, and 74 groups on 29 detectors
in nw11. Only the above 2 and 15 groups lack scan overlap. The exception is not a
VAL judgment that all detectors are bad. Existing jump treatment can exclude an
individual detector for a whole observation after three admitted groups; that
rule is distinct from aborting every detector in its network and must not be
silently altered when repairing this failure.

A diagnostic-only native replay of nw2 and nw11 reproduced both failures. A
read-only export was inserted immediately before the original failing call to
record final admitted coordinates and overlapping scans. Event exports and native
time arrays were bit-identical to the original failed run for both networks.
The source instrumentation is preserved in `diagnostic-export.patch`; the original
tracked source was restored byte-for-byte and rebuilt. The diagnostic binary,
input manifest, complete failed replay and measured RUN record are retained.
Its version includes `912787f62-dirty`; the explicit patch defines that dirt.
`network-failure-summary.json` lists all final coordinates and identity hashes.

## Required repair direction and remaining scope

This is an implementation/completion gap relative to the intended cause-preserving,
named-use design. SCI-VAL-REQ-001/002/004/007/008 distinguish producer facts,
unknown versus inapplicable states, and consumer-owned use decisions. RTC must
publish the facts and exact support; VAL evaluates uses under the owner's profile.
A caught network exception is neither a substitute for those facts nor a reason
to declare the unaffected detectors unusable.

The bounded repair should carry the complete processing-domain relation into the
RTC jump plan, preserve known outside-domain events and their causes, and keep
unaffected detector/scan uses available. It must retain the current per-detector
jump-count rule and propagate any relevant FIR/state/donor influence before
admitting downstream samples. In particular, ignoring a tail event merely because
its center is outside science support is insufficient: nw2's nearer event could
intersect context, and the exact realized influence must be checked. Genuinely
missing relations must remain distinguishable from known non-overlap. Do not
swallow the exception, fabricate a scan, turn unknown into good, or label the
whole network bad.

No repair or new admission was performed in this audit. Its scope is the owner's
request to compare and understand. Repair and a new full-network reduction remain
open; the original MAP still excludes nw2/nw11, and nw3 scan 5 retains its separate
ALS convergence failure. J2000, the current APT selection, fixed nominal beams,
no spatial mask and uniform coefficients remain unchanged.

## Governance and evidence

Read ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE and
REVIEW_AND_CONFORMANCE. This is a read-only scientific audit/evidence follow-up on
the existing MAP branch; no application WIP slot or additional branch is created.
Local AppleClang/Homebrew and `/Users/gwilson/tolteca/bin/python` only; no Unity,
Spack validation, push, integration, activation or cleanup. Gates are diagnostic
reconstruction, fit reproduction, exact native failure replay, restoration of
tracked source and visual inspection. Previous broad tests are historical MAP
implementation evidence, not newly rerun audit gates. Exactly two expected
network exceptions occurred in the failure replay. Independent review is recorded
separately against the final evidence commit.

Durable audit root: `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-offset-network-audit-20260922`.
35 files are indexed by `FILES_SHA256SUMS`, SHA-256
`8b9bdc7799d71f052dd0a95807f3b9c8804df38c3101bb47ffd69945fbf785e6`.
