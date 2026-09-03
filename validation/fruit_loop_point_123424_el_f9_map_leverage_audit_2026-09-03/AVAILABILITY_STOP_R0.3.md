# FRUIT EL-F9 registered availability stop r0.3

Status: **the exact leverage branch is stopped; the read-only flagging audit may continue**

## Why the r0.2 interpretation is superseded

`PRE_ANALYSIS_SEMANTICS_R0.2.md` identified `weight_formal_I` as an
additive mapmaking coefficient. That statement is incorrect for the JINC
products fixed by this test. It is preserved as the chronological record of
the pre-measurement check, but it must not be used to interpret EL-F9.

For one JINC map pixel, the checked-out implementation accumulates

\[
S=\sum_i K_i w_i d_i,\qquad
G=\sum_i K_i w_i,\qquad
V=\sum_i K_i^2 w_i,
\]

then publishes

\[
M=S/G,\qquad W_{\rm formal}=G^2/V.
\]

Here `K_i` is the signed JINC response, `w_i` is the detector weight, and
`d_i` is the processed sample. `G` is the in-memory `grid_weight`; `V` is the
pre-normalization `weight` accumulator. The normalization replaces `weight`
with `G^2/V`, releases `grid_weight`, and later copies the finalized
coefficient into `weight_formal`. Neither `G` nor `V` is published in the
retained FITS products.

Therefore

\[
W_{\rm formal,all}-W_{\rm formal,without}
\]

is not UID 4460's additive contribution and need not be nonnegative. The
registered `W4460`, `L4460`, `M4460`, and `C4460` constructions are not
admissible for these products.

## Observed stop evidence

After registration r0.3, the paired a1400 maps showed:

- primary `METHOD = jinc`;
- 94,634 positive `weight_formal_I` pixels in N5 and 94,635 in A5-map;
- 2,019 materially negative and 2,448 materially positive values of
  `weight_formal_I(N5) - weight_formal_I(A5-map)`;
- a difference range of -0.00040876871744439935 to
  +0.005278011435943697, against a floating-point classification bound of
  1.4210854715202004e-14; and
- no N5-positive/A5-map-zero support loss.

The empirical-to-formal coefficient ratios also differ between the pair
(median 1.292566567724443 for N5 and 1.2925953335741982 for A5-map), so the
published `weight_I` planes cannot repair the missing JINC state.

The paired scan-5 a1400 detector-statistic planes are bitwise identical for
all 445 APT-accepted detectors. The signed final-coefficient difference is
therefore not evidence for a changed a1400 PTC detector weight; it is
consistent with the nonlinear JINC coefficient and support finalization.

## Consequence

The exact per-pixel fractional leverage, withheld contribution, hit count,
unique-detector count, and processed-signal contrast are unavailable from
the retained pair. In accordance with `TEST_DEFINITION.md`, EL-F9 will not
substitute an inverse-variance proxy and will not run Citlali to fill the
gap.

The remaining read-only audit is still valid. It may report the signed formal
coefficient difference strictly as a non-leverage diagnostic, inventory the
missing products, trace APT/RTC/PTC/map-diagnostic behavior, and test the four
trigger pixels against the already registered direct map component.

Implementation evidence used for this correction is non-authoritative:

- `include/citlali/core/mapmaking/jinc_mm.h` (JINC accumulation);
- `src/citlali/core/mapmaking/map.cpp` (normalization and release);
- `include/citlali/core/mapmaking/map.h` (in-memory state); and
- `include/citlali/core/pipeline/map_image_output_helpers.h` (published
  planes).
