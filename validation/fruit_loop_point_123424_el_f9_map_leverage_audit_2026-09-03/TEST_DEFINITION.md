# FRUIT EL-F9 map-leverage and flagging audit

Test ID: `SCI-FRUIT-EL-F9-MAP-LEVERAGE-FLAGGING-AUDIT-R0.1`

Status: **registered before new measurement; owner-authorized read-only audit**

## Question

Why can the already deweighted UID 4460 have a large, scan-shaped direct map
effect in EL-F8, and which part of that behavior is or is not represented by
the existing APT, detector-weight, sample-flag, and map-diagnostic evidence?

## Fixed comparison

Use only the completed iteration-5 a1400 products from:

- `N5`, the EL-F6 injected replay with the carried UID 4460 record removed;
  and
- `A5-map`, the EL-F8 injected replay in which the same carried record is
  applied only before map accumulation.

These two products share the same incoming injected iteration-4 state and the
same RTC/PTC participation. Their relevant final-map difference is the
registered direct map component. No reduction will be rerun.

Let `M_all` and `W_all` be the N5 signal and weight maps, in which UID 4460
participates in map accumulation. Let `M_without` and `W_without` be the
A5-map signal and weight maps, in which its proposed samples are withheld.
Define on common finite pixels:

\[
W_{4460}=W_{all}-W_{without},
\qquad
L_{4460}=W_{4460}/W_{all}.
\]

`L4460` is the exact fractional map-weight leverage implied by the paired
products if the registered pairwise identity and map-weight semantics are
confirmed. It is not automatically a hit fraction or detector count.

Where `W4460` is materially positive, reconstruct the map value represented
by the withheld contribution:

\[
M_{4460}=
\frac{M_{all}W_{all}-M_{without}W_{without}}{W_{4460}},
\]

and its contrast with the retained map,

\[
C_{4460}=M_{4460}-M_{without}.
\]

The exact identity

\[
M_{without}-M_{all}=-L_{4460}C_{4460}
\]

must close within a declared floating-point bound before either `L4460` or
`C4460` is interpreted. Pixels whose subtracted weight is indistinguishable
from floating-point roundoff are excluded from the reconstructed-contribution
calculation and reported separately.

## Fixed measurements

For the complete finite map, the existing injected-source aperture, the
existing fitted-Neptune aperture, and the existing 40--120 arcsec annulus with
Neptune excluded, report:

- the number and fraction of pixels with positive UID 4460 weight support;
- distributions and percentiles of `W_all`, `W4460`, `L4460`, `M4460`,
  `C4460`, and the registered `D_map = M_without-M_all`;
- the median and 90th/95th/99th percentiles of leverage and absolute contrast
  among the top 10, 5, and 1 percent of `abs(D_map)` on UID support;
- rank and linear correlations of `abs(D_map)` with leverage and absolute
  contrast, reported descriptively rather than used as pass/fail gates;
- binned `D_map` RMS versus leverage quartile and versus `W_all` quartile;
- the same quantities at the four original map-diagnostic trigger pixels,
  when those pixels have UID support in the final-map pair; and
- any exact hit count, unique-detector count, sample-flag reason, or
  detector-resolved residual statistic present in retained outputs.

The final item is availability-gated. If retained products do not contain an
exact quantity, record it as unavailable; do not reconstruct a hit count or
unique-detector count from inverse variance, and do not call `1/L4460` a
detector count.

## Flagging and weighting trace

Audit, without changing, the checked-out implementation and retained evidence
for:

1. APT admission and pre-existing detector flags for UID 4460;
2. its detector-weight factor and rank among accepted a1400 detectors;
3. prior sample-local flags and how many mapmaking samples were already
   unavailable before the carried hard record;
4. the map-pixel selection threshold, leading-contributor assignment,
   repeated-detector threshold, and use or non-use of leave-one-out
   significance;
5. the distinction between moderate weighting, sample-local flagging, and the
   later factor-zero scan-local record; and
6. whether ordinary detector-level summaries contain a statistic that is
   explicitly sensitive to spatially concentrated map leverage.

Implementation is evidence about present behavior, not scientific authority.
The audit must separate observed values, source-defined semantics, and
interpretive inference.

## Interpretation

No numerical dominance threshold or candidate safeguard is selected. The
result will say whether the observed direct effect is associated with high
fractional map leverage, high withheld-contribution contrast, or both, and
whether the present detector-level checks explicitly represent that
map-local combination. Any causal or generic claim beyond the exact paired
iteration-5 products is prohibited.

## Validity and stop rules

- Verify the frozen EL-F8 and EL-F6 result identities before analysis.
- Require matching signal units, weight units, WCS/grid, shapes,
  normalization, and finite support.
- Confirm from source and provenance that subtracting the paired weight maps
  has the stated numerator/denominator meaning.
- Preserve all external reduction and input products byte-for-byte.
- Analysis output is limited to small tables, JSON/YAML provenance, plots,
  and an owner-facing report in this validation directory.
- Stop and report unavailable evidence if exact per-detector leverage cannot
  be recovered from the retained pair. Do not run Citlali to fill a gap.

This audit cannot judge UID 4460, validate a flagging rule, select a safeguard
or recurrence, change production, qualify FRUIT, launch Gate D or Stage B, or
authorize Unity activity.
