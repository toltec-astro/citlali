# FRUIT EL-F9 map-leverage and flagging audit

Result: **the flagging trace is complete; exact map leverage is unavailable
from the retained JINC products**

Test ID: `SCI-FRUIT-EL-F9-MAP-LEVERAGE-FLAGGING-AUDIT-R0.1`

## Plain answer

UID 4460 was neither invisible to the pipeline nor obviously dominant in the
ordinary detector-level summaries.

- The APT accepted it, and all three recorded APT flags are zero.
- Earlier FRUIT evidence had already reduced its accumulated detector-weight
  factor to `0.75867`, near the 36th percentile from the low-weight end.
- In the iteration-5 scan examined here, its final PTC weight is at the 79th
  percentile from the low end and its flagged fraction is effectively in the
  middle of a large tie. Those two quantities do not look like hard-rejection
  cases.
- Its residual RMS and standard deviation are high, at the 91st and 93rd
  percentiles, and its median is near the 1st percentile. The detector is
  therefore not ordinary in every scalar summary, but none of these recorded
  values directly measures a spatially concentrated map imprint.

The map diagnostic then takes a qualitatively different action. Four
globally extreme off-source pixels identify UID 4460 as their leading
scan-local contributor. Four is exactly the configured repeat threshold, so
the next iteration applies a factor-zero record to the whole detector/scan:
305 processed samples are withheld from mapmaking, including 271 newly
flagged samples and 34 that were already flagged.

Most importantly, the four pixels that selected that action have exactly
zero direct map difference between N5 and A5-map. The large arcs occur
elsewhere after the full set of scan-5 samples is withheld. The trigger
location and the response location are therefore spatially decoupled.

## What the published maps can and cannot tell us

The prospective EL-F9 definition assumed that subtracting paired formal
weight planes would recover UID 4460's map weight. Source inspection and the
registered numerical check disproved that assumption for JINC.

At each pixel, JINC accumulates

\[
S=\sum_i K_iw_id_i,\qquad G=\sum_i K_iw_i,\qquad
V=\sum_iK_i^2w_i,
\]

then publishes

\[
M=S/G,\qquad W_{\rm formal}=G^2/V.
\]

`weight_formal_I` is therefore a nonlinear finalized coefficient, not an
additive sum of detector weights. The unpublished `G` and `V` planes are both
required to remove one detector exactly.

The observed difference
`weight_formal_I(N5) - weight_formal_I(A5-map)` is materially negative in
2,019 pixels and positive in 2,448 pixels, ranging from
`-4.08769e-4` to `+5.27801e-3`, while the floating-point classification bound
is `1.42109e-14`. That signed behavior is expected from `G^2/V`, but it proves
that this difference is not an admissible UID-weight or leverage map. The
empirically rescaled `weight_I` plane also cannot repair the loss because its
scale differs slightly between the two products.

Consequently, the retained products cannot determine whether the large
direct arcs are driven mainly by unusually high local fractional leverage,
unusually high processed-signal contrast, or a combination of both. No proxy
was substituted and Citlali was not rerun.

## Recorded detector evidence

The comparison cohort is the 445 accepted a1400 detectors in zero-based scan
5. Percentiles use the midpoint of ties and run from the low end.

| Quantity | UID 4460 | Percentile |
|---|---:|---:|
| PTC detector weight | `5.69737e-5` | 79.21 |
| final flagged fraction | `0.111475` | 44.94 |
| PTC residual RMS | `652.056` | 91.35 |
| PTC residual standard deviation | `652.654` | 92.92 |
| PTC residual median | `-102.366` | 1.01 |
| PTC inverse-variance window median | `1.83256e-5` | 20.99 |
| RTC inverse-variance window median | `1.07099e-4` | 79.66 |
| RTC local-despike samples added in this scan | `0` | midpoint of zero tie |
| detector notches applied in this scan | `0` | midpoint of zero tie |

The N5 and A5-map scan-5 values of RMS, standard deviation, median, flagged
fraction, and detector weight are identical for all 445 accepted a1400
detectors. This is consistent with their common RTC/PTC participation and
confirms that the direct map comparison is not hiding a changed a1400 PTC
weight.

Four historical UID 4460 RTC sample-mask intervals occur in scans 0--3; none
occurs in scan 5. They show that prior processing had seen localized issues,
but do not explain the scan-5 hard action.

## The four trigger pixels

| Row, col | Map value (mJy/beam) | Effective samples | UID leave-one-out z | Direct next-map effect (mJy/beam) |
|---|---:|---:|---:|---:|
| 142, 280 | 135.715 | 230.729 | 1.95356 | 0 |
| 144, 280 | 116.716 | 219.500 | 1.69687 | 0 |
| 142, 281 | 126.721 | 216.525 | 2.12348 | 0 |
| 144, 281 | 111.562 | 215.322 | 1.78382 | 0 |

The map-level outlier selection is stringent, but the detector-specific
leave-one-out values shown above are only `1.70`--`2.12`. The checked-out rule
records leave-one-out evidence but does not require it to cross a separate
detector-specific threshold before counting the leading UID. Once the repeat
count reaches four, it emits the factor-zero scan-local record.

## Why the apparent flagging paradox occurs

The evidence supports this bounded interpretation:

1. ordinary APT, RTC, and PTC logic does record some unusual behavior, but
   the detector's weight and flagged fraction are not extreme;
2. no retained detector scalar directly asks whether one detector has a
   concentrated or geometrically coherent effect in a small set of map
   pixels;
3. the map rule notices four extreme pixels, but the decision boundary is a
   repeated *leading contributor* count rather than a minimum
   detector-specific leave-one-out effect; and
4. crossing that count changes the action discontinuously from no hard record
   to withholding the whole scan-local detector contribution.

This explains how one detector can be accepted and moderately weighted yet
still participate in a large hard-exclusion response. It does **not** prove
that UID 4460 is scientifically bad, that its direct samples alone created
the original extreme map pixels, or that another observation would behave
the same way.

EL-F8 supersedes the earlier expectation that upstream shared cleaning would
be the main amplifier: upstream placement is material, but the direct
mapmaking component is larger in the registered Neptune and annular regions.
EL-F9 further shows that the direct component is not localized at the four
trigger pixels. The observed scan-shaped response and the signed published
coefficient diagnostic are consistent with the JINC response footprint, but
the missing additive component planes prevent a quantitative leverage-versus-
contrast attribution.

## Missing evidence and the next useful test

The retained FITS files publish signal, empirical and formal coefficients,
noise variance, kernel, and signal-to-noise planes. `mapdiag.nc` retains
aggregate map summaries. None retains the pixel-resolved JINC numerator
`S`, grid denominator `G`, variance accumulator `V`, their UID 4460
components, exact hit count, unique-detector count, or the final processed
sample position/value/flag stream.

The next useful step is a separately reviewed **diagnostic-only JINC
accounting replay**. It should preserve the existing numerical path and write
small, targeted component products for the fixed UID/scan:

- total and UID-specific `S`, `G`, and `V` planes;
- exact sample/hit and unique-detector support needed to interpret local
  redundancy;
- the final per-sample positions, processed values, weights, and flag reasons
  for the target UID/scan; and
- the support threshold and mask used during finalization.

A compatibility replay must show that enabling those diagnostics leaves the
science maps bitwise unchanged. The component arithmetic should then
reconstruct the already retained A5-map result. Only after that closure would
it be honest to compare fractional leverage with processed-signal contrast or
design a soft/map-local safeguard.

This recommendation is not authorization to add diagnostics or run another
reduction.

## Evidence status and limits

Observed values come from the hash-registered EL-F8/EL-F6 products, matched
APT, diagnostic NetCDF files, and learning tables. JINC and learning-rule
semantics come from the checked-out implementation and are non-authoritative
historical/implementation evidence. The explanation above is a bounded
inference from both classes.

The result applies only to observation 123424, a1400, UID 4460, zero-based
scan 5, and the fixed iteration-4-to-5 pair. It does not select a detector
policy, threshold, soft factor, recurrence, production default, qualified
FRUIT method, Gate D launch, Stage B activity, or Unity action.
