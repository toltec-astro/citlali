# OG pointing compatibility verification

**Physical conventions are preserved; complete interface/numerical compatibility
is not yet established.** Reuse the working OG/TolTECA correction machinery.
There is no reason here to design a new pointing correction or refit the saved
maps. The current successor still lacks the conventional PPT publication handoff.
This audit also demonstrates a single-PPT input rejection and a subsecond epoch
conversion difference to close explicitly before claiming full compatibility.

## Exact identities and scope

- Original fork comparator:376e002238b1f49aeced8a3f33e8742db141634b.
- Preserved OG `gw_dev`:ffc6b9070f4744f9778f3db71cdc468846d1da89.
- Frozen reference POINT:c31a60a0b74a7149d03d542966d6e35b77b8091c;
  reference SCIENCE:187df04b21e942701cf41e6d9c50883922fd65aa.
  The supplied archived project is V2 refactor operational evidence, not an
  execution of the current OG checkout. Its recorded version agrees with these
  [baseline identities](../../doc/APPLICATION_BASELINES.md).
- Operational TolTECA v2:2791e6a1e6349ad1d3ac549a648f41cbc51b98c7.
- Combined candidate:b090da0458271c618568802fde1cf5c39f5dd5d2,
  application source:c83375dc030095821999a0e0ff435f13bf82b914.

Read-only source comparison covers original fork, preserved OG and frozen V2
geometry/PPT/output code, current configured POINT and weighted native path, and
the actual operational TolTECA resolver. No source fit, build, reduction, input
restoration, calibration issuance, Unity operation or scientific-policy change.

## Verified convention and implementation chain

| Stage | Established behavior | Successor finding |
|---|---|---|
| POINT measurement | AltAz `x_t` = tangent azimuth displacement; `y_t` = elevation displacement, both arcsec | Accepted native centroid carries these same axes/units/signs; still diagnostic JSON |
| Cross-array correction estimate | TolTECA takes the simple mean of PPT x/y, not a new noise-weighted array estimator | Reuse exact pinned resolver; detector noise weighting does not change this downstream rule |
| User/Paddle reconciliation | Add POINT minus SCIENCE User+Paddle offsets to the measured displacement, then negate once | Existing TolTECA source is the authority; do not negate in the PPT adapter or add offsets a second time in native Citlali |
| Azimuth projection | Raw azimuth separation has cosine of corrected elevation | Same helper arithmetic; supplied tangent-az correction has **no additional cosine factor** |
| Detector geometry | Rotate APT x/y by corrected elevation, add supplied az/alt correction | Same scientific arithmetic in weighted successor |
| Output orientation | Positive tangent az is toward decreasing saved FITS columns; elevation increases rows | Same physical orientation despite different internal storage conventions |
| Constant/bracket correction | One value constant; two values linear in declared time support; legacy also permits observation-span support | Native model supports constant/linear; configured boundary is narrower and epoch precision differs (below) |
| Correction versus measurement | PPT describes measured displacement; downstream composes correction | Successor deliberately emits `pointing_correction:null`, no PPT; adapter/qualification still missing |

Write U for UserOff+PaddleOff converted from radians to arcseconds. Operational
TolTECA computes, separately for az and elevation:

`correction = -(mean(PPT measured displacement) + U_POINT - U_SCIENCE)`.

It chooses the closest pre/post PPT by observation number, deduplicates an exact
observation match, and attaches each PPT's recorded MJD. This is bracket selection,
not a flux-calibration or new interpolation formula. Source:
`tolteca/reduce/engines/citlali.py:505–588` at the pinned TolTECA revision.

For the physical coordinate transform, let A/E be telescope az/elevation, As/Es
the source angles, CA/CE the telescope correction series, dx/dy the APT arcsecond
geometry, pa/pe the supplied arcsecond corrections, k=arcsec-to-radian and e=E−CE:

`az_tangent = cos(e)*(A-As) - CA + k*(cos(e)*dx - sin(e)*dy + pa)`

`el_tangent = E-Es-CE + k*(sin(e)*dx + cos(e)*dy + pe)`.

OG performs its azimuth unwrap before these operations. The successor evaluates
the existing transforms at its assigned native/RTC times. User/Paddle header
fields are carried separately for upstream composition; they are not independently
re-added by this detector transform. The two uses of corrected elevation are for
boresight and detector rotation, not duplicate subtraction from one coordinate.

OG source anchors: `src/citlali/core/engine/telescope.cpp:237–252`;
`include/citlali/core/utils/pointing.h:30–58`; POINT table scaling in
`include/citlali/core/engine/pointing.h:613–614`. The original fork and preserved
OG agree on these expressions; frozen V2 retains them in its extracted owners.
Successor anchors: `src/citlali/core/pipeline/ast_rtc_coordinates.cpp:25–56`;
`src/citlali/cli/point_weighted_execution.h:105–128,261,355–356`;
`include/citlali/core/utils/pointing.h:16–83`.

### Why internal column signs differ without changing the answer

OG deposits positive az into increasing internal columns, fits that internal map,
and converts fitted column displacement to positive arcseconds. Its FITS writer
then reverses columns (`fits_io.h:68–73`), with negative CDELT1.
The successor deposits directly into a negative-CDELT1 grid and fits physical
coordinates reconstructed from that signed WCS. Both persisted images therefore
represent the same sign. OG `jinc_mm.h:533–539`, `map.cpp:97–100` and successor
`ast_rtc_coordinates.cpp:75–87`, `forward_map.cpp:49–55`,
`point_empirical.py:45–48` establish the chain. This is coordinate convention
agreement, not equality of JINC and containing-pixel map estimators or fit results.

## Executed checks using saved products and synthetic fixtures

`check_resolver.py` extracts and executes the **actual pinned** TolTECA resolver,
not a rewritten approximation. Five synthetic cases cover nonzero unequal
User/Paddle offsets, positive/negative displacements, reversed PPT order, single
pre/post pointings and exact-observation deduplication.

Actual archived PPTs152389/152391/152393 and saved SCIENCE telescope headers then
reproduce every endpoint and MJD in `redu03/astrometry_provenance.yaml` exactly:

| SCIENCE | Az correction endpoints, arcsec | Elevation endpoints, arcsec | Largest difference from saved values |
|---|---|---|---|
| 152390 | −8.403992188685542, −10.438279151916504 | +1.257425770999287, −0.37166276574134827 | 0 |
| 152392 | −8.755760388390684, −9.092263221740723 | +0.16599909439852567, +0.6797776222229004 | 0 |

Six existing combined-build synthetic C++ tests passed: three ForwardMap tests
for corrected elevation/telescope corrections, zero/rotation sign and TAN-grid
orientation; three native-carrier tests for constant support, explicit-MJD support
and observation-span support. Exact output is in `native_test_results.json`.
This is supplemental AppleClang/Homebrew evidence using the existing build;
no observation processing or rebuild. Existing synthetic signed-centroid Python
fixtures were inspected but not rerun because they fit authored maps.

## Demonstrated gaps; no application repair performed in this audit

1. **One-PPT correction input is currently rejected by configured POINT.** The
   actual TolTECA resolver emits one positive MJD with one az/alt value. The old
   reader accepts that as a constant correction. New `_calibration` rejects it
   because it requires two positive epochs. Running that exact helper on both
   single-PPT fixtures reproduces the rejection; the same constant without its
   date is accepted, and the two-PPT actual-epoch pair passes. Preserve the known
   constant meaning and date provenance in the adapter; do not change TolTECA to
   invent a second date. Source: `configured_point_config.py:230–266`, retained
   `pointing_offsets_config_read.h:116–120`. New configuration also rejects the
   old two-value observation-span form without positive epochs. This is a narrower
   supported input set, not a change in the physical offset formula.
2. **Fractional-second interpolation is not exactly the old implementation.** OG
   `modified_julian_date_to_unix` returns `long long`, truncating seconds. New CAL
   ingress uses double `(mjd-40587)*86400`. For the saved152390 bracket, the largest
   same-time difference across overlapping support is0.000965411″ az and0.000773116″
   el; for152392 it is0.000189330″/0.000289073″. These are arithmetic bounds, not
   observed map shifts or new scientific tolerances. Epochs differ by0.178–0.987s.
   The existing explicit-MJD unit test exercises the older truncating support
   helper with integer-day epochs, so it does not test this CAL-ingress difference.
   Add fractional-epoch parity cases and explicitly resolve shared conversion
   behavior; do not silently claim bitwise preservation or silently retime data.
3. **Native versus common-grid evaluation is intentionally a different schedule.**
   Matching formulas does not make coordinates at different sample times identical.
   A same-time fixture is required to isolate correction compatibility; end-to-end
   observational claims remain separate.
4. **Successor POINT-to-SCIENCE publication is still unconnected.** Native accepted
   centroids are measurements; no conventional PPT is published, and corrections
   remain null. The new PPT adapter must preserve the OG meanings and account for
   corrections already applied during its POINT reduction. No current full fixture
   composes resolver→new CAL ingress→AST→map/measurement→PPT against the reference.

## Decision for the connected implementation plan

Treat OG axes, units, sign, User/Paddle composition, bracket selection and correction
application as recovered working behavior to preserve. They are not new scientific
choices. Add the demonstrated constant-date/configuration and fractional-epoch
cases to the first compatibility batch. Reuse existing concrete correction owners
with one conversion boundary; avoid a second Python/native implementation of the
correction formula. Evaluate fractional-time handling explicitly against OG,
separately from the authorized native scheduling change.

Do not reinterpret these findings as approval of photometric uncertainties or
SCIENCE source policy. The fitter/mapmaker changed, so identical offset conventions
do not guarantee identical measured centroids. Correct publication and complete
application must still be verified when the connected path exists. The wing study
remains paused.
