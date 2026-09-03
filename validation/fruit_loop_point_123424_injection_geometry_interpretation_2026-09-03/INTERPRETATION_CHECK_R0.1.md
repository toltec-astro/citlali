# FRUIT observation-123424 injection-geometry interpretation check

Status: **read-only interpretation supplement; no experiment or algorithm
change**

Date: `2026-09-03`

This check answers how the real astronomical source and synthetic source were
combined in EL-F2, EL-F3, and EL-F4. It uses the frozen configurations,
implementation, and completed development products as evidence. Those sources
describe the executed experiment but are not scientific authority or method
qualification.

## Finding

Observation 123424 is a pointing observation of **Neptune**. The synthetic
source was placed at the nominal pointing-map origin. The fitted Neptune and
synthetic-source centroids were **not coincident**, but they occupied the same
central source region and their responses overlapped. This was not an
off-source or blank-sky injection.

At the iteration-4 boundary relevant to the UID 4460 event, the measured
geometry was:

| Array | Fitted Neptune offset (az, el; arcsec) | Synthetic-kernel offset (az, el; arcsec) | Separation (arcsec) | Kernel FWHM major/minor (arcsec) | Separation / major FWHM |
|---|---:|---:|---:|---:|---:|
| a1100 | (+13.031, -5.574) | (-0.020, -0.038) | 14.177 | 6.183 / 6.135 | 2.29 |
| a1400 | (+12.556, -5.333) | (+0.053, +0.038) | 13.608 | 7.442 / 7.138 | 1.83 |
| a2000 | (+15.259, -9.433) | (-0.146, -0.197) | 17.962 | 9.917 / 9.321 | 1.81 |

The coordinates above use the FITS `AZOFFSET`/`ELOFFSET` convention. The map
has 1-arcsec pixels and a WCS origin of `(0, 0)`. The a1400 synthetic transfer
fit at iteration 4 is within `0.046` arcsec of its same-iteration kernel fit,
confirming that the injected source itself is at the map origin.

The a1400 Neptune fit has a `12.6` by `7.18` arcsec FWHM at this boundary.
Consequently the two centroids are separated, rather than being the same
source, but the synthetic source lies on the central response surrounding the
much brighter real source. Both centroids are also inside the configured
30-arcsec map-diagnostic source radius.

## Amplitudes

The frozen input describes Neptune with the following photometry metadata:
`2329`, `2800`, and `1923` mJy for a1100, a1400, and a2000. The synthetic
source is exactly `100 mJy/beam` in every array before processing, beginning
at absolute iteration 1.

The configured Neptune values and fitted map amplitudes are not interchangeable:
the latter include the realized reduction response and fitting limitations.
At iteration 4 they were:

| Array | Configured Neptune value (mJy) | Control pointing-fit amplitude (mJy/beam) | Control-map value at injection origin (mJy/beam) | Injected truth (mJy/beam) | Fitted transfer amplitude (mJy/beam) | Kernel-normalized recovery |
|---|---:|---:|---:|---:|---:|---:|
| a1100 | 2329 | 5742.3 | 878.9 | 100 | 70.4 | 0.851657 |
| a1400 | 2800 | 1766.4 | 476.8 | 100 | 76.1 | 0.890451 |
| a2000 | 1923 | 138.8 | -28.7 | 100 | 55.8 | 0.724708 |

The weak and unstable a2000 pointing fit is not a reliable measurement of
Neptune's physical flux. More generally, none of the pointing-fit amplitudes
above should replace the configured photometry as source truth.

For the decisive a1400 case, the control map already contained
`476.849 mJy/beam` at the nominal injection origin at iteration 4. The
injected map contained `549.035 mJy/beam`, a pixel difference of
`72.186 mJy/beam`. Thus the synthetic response was measured on top of a
substantial pre-existing central-field signal. At iteration 5 the analogous
values were `514.446`, `579.435`, and `64.989 mJy/beam`.

## What the recovery estimator measures

Both branches contain the original Neptune observation. Only the injected
branch adds `100 mJy/beam` times the pristine unit point-source kernel to each
detector timestream. The addition occurs before subtraction of the preceding
FRUIT model and before PTC cleaning. It is repeated in every enabled iteration.

For each iteration the analysis forms

`T_k = signal_I(injected, k) - signal_I(control, k)`.

It fits a Gaussian to `T_k` around the known map origin and divides the fitted
amplitude by `100 mJy/beam` and by the fitted peak of the same-iteration
injected kernel. It also reports a full-map least-squares projection of `T_k`
onto that kernel. The Gaussian search and fit use the central 25-arcsec region,
which includes both the injection origin and Neptune's fitted centroid.

This subtraction removes the real source only to the extent that the two
reductions respond identically to it. FRUIT processing, learned masks and
penalties, detector weights, and PTC cleaning are nonlinear or stateful.
Therefore `T_k` is the **causal difference made by adding the synthetic source
to this Neptune pointing observation**, not a guaranteed isolated-source map.
Changes induced at Neptune's position or elsewhere remain in the transfer
map and can affect its fit and residual metrics.

## Relation to UID 4460

The four a1400 map pixels that caused UID 4460 to cross the hard four-pixel
threshold are not central-source pixels. After applying the recorded
internal-to-FITS column reversal, their offsets are approximately
`(+102 to +103, -35 to -33)` arcsec. They are `107.2`--`108.8` arcsec from
the synthetic source and `93.6`--`95.2` arcsec from the fitted Neptune
centroid.

At those four pixels, the injected-minus-control differences at iteration 4
are only `0.50`--`0.58 mJy/beam`. Nevertheless, contributor tracing changed
UID 4460 from three qualifying pixels in the control to four in the injected
branch, exactly at the configured hard threshold. Applying that new
factor-zero penalty on iteration 5 caused the previously demonstrated a1400
response collapse.

The penalty therefore did not directly reject the combined central peak. It
was a nonlocal, scan-5 threshold response to a small change in an off-source
structure. Because the injection passes through the observation's actual
detectors, scan path, cleaning, weighting, and iterative feedback, the present
evidence cannot distinguish a generic FRUIT source/penalty interaction from a
Neptune-field, injection-location, detector, or scan-geometry interaction.

## Revised interpretation of EL-F2, EL-F3, and EL-F4

- **EL-F2 remains a valid result for its exact paired experiment**, but it is
  evidence for incremental recovery of a map-centered synthetic source in a
  Neptune pointing field. It is not evidence for an isolated compact source
  on blank background.
- **EL-F3's causal conclusion remains valid but narrower than a generic
  source-protection claim.** Removing the carried UID 4460 penalty caused full
  reversal of the incremental a1400 response loss in this exact composite
  field and scan geometry. It does not establish how the same penalty policy
  behaves for an isolated or off-source injection.
- **EL-F4 still rejects the wholesale feedback-bypass policy.** Its primary
  rescue reproduces the same geometry-specific EL-F3 event, while the
  protected regressions show that suppressing all feedback-supported penalty
  evidence is too broad. EL-F4 does not by itself establish a generic
  astronomical-source/penalty mechanism.
- Observation 152389 also used a map-centered injection in a pointing field
  containing a real source; its fitted real-source centroid is about
  `7.3`--`7.7` arcsec from the synthetic kernel. It is an independent
  observation, but not an independent blank-sky injection geometry.

## Recommendation before EL-F5

This changes the design priority. Do not use the existing centered injections
alone to derive a supposedly general EL-F5 penalty-attribution rule.

The next experimental proposal should first include a **prospectively placed
off-source injection control in observation 123424**. The location must be
chosen from coverage and geometry only, before examining outcomes; lie outside
the real source and 30-arcsec source-protection region by several realized
beam widths; retain adequate and comparable coverage; and be represented by a
properly projected point-source kernel in the detector timestream rather than
by shifting a finished map.

A rigorous location comparison will require a newly frozen executable because
the current diagnostic seam has no offset parameter. That future harness
change must remain default-off and prove bitwise identity when disabled. A
fresh same-build control, centered-injection replication, and off-source
injection should be run under the unchanged complete-map penalty policy before
another bypass or attribution policy is tested. An optional second off-source
position in an orthogonal scan direction would help separate location from
scan-direction effects.

If the centered run reproduces the UID 4460 event while the off-source run
does not, the mechanism should be treated as location/geometry-specific until
shown otherwise. If a comparable off-source penalty and response loss occurs,
that would strengthen the case for a broader FRUIT penalty interaction. Only
then should EL-F5 test a prospectively defined discriminator or soft-penalty
policy.

No new injection location, threshold, run, implementation, recurrence,
fallback, production change, Gate D, or Stage B work is authorized by this
interpretation check.

## Evidence inspected

- frozen observation-123424 base and alpha-1.25 injection configurations;
- `include/citlali/core/pipeline/fruit_loop_injected_source_test.h` and
  `include/citlali/core/engine/detail/pointing_run_impl.h`;
- `tools/fruit_loops/compare_injected_source_pair.py`;
- EL-F2 iteration metrics and control/injected FITS products;
- pointing-fit ECSV products identifying Neptune and its fitted centroids;
- EL-F3 causal-diagnostic and counterfactual records; and
- EL-F4 design evidence, penalty inventory, and complete-map control/injected
  products.
