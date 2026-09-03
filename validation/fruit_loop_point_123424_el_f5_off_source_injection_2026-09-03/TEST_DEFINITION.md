# FRUIT EL-F5 off-source injection location control

Status: **registered before implementation or execution**

Test ID: `SCI-FRUIT-EL-F5-OFF-SOURCE-INJECTION-R0.1`

## Question

Does the injection-specific UID 4460 penalty association and subsequent
a1400 response loss exposed by the centered observation-123424 experiment
also occur when the same synthetic point source is placed away from Neptune
in the same observation?

This is a location-control experiment. It is not a blank-field test, an
isolated-source response measurement, a new detector judgment, a penalty-
policy proposal, or a FRUIT-method qualification.

## Prospectively fixed location

The synthetic source is fixed at FITS map-world offsets

`(AZOFFSET, ELOFFSET) = (0, -60) arcsec`.

The location was selected on 2026-09-03 using only the already completed EL-F4
complete-map **control weight** planes. No off-source injected result existed.
It lies outside the configured 30-arcsec central source region and is about
56 arcsec from the fitted a1400 Neptune position at the relevant iteration-4
boundary.

In a 21 by 21 pixel neighborhood around the location, the median positive
weight divided by the full-map positive-weight 95th percentile spans these
ranges over control iterations 0--5:

| Array | Minimum | Maximum |
|---|---:|---:|
| a1100 | 0.759 | 0.788 |
| a1400 | 0.728 | 0.751 |
| a2000 | 0.765 | 0.851 |

These weight-only facts establish adequate three-array coverage. The location
was not optimized using a detector-penalty position or an injected outcome.

## Diagnostic implementation contract

The existing default-disabled diagnostic injection receives two finite
configuration values, `az_offset_arcsec` and `el_offset_arcsec`, both defaulting
to zero. When the injection is active and either value is nonzero, the normal
unit point-source kernel is generated at that map-world position and follows
the same raw-timestream filtering, downsampling, PTC processing, mapmaking,
and per-iteration injection ordering as the centered kernel.

The source must not be created by shifting a finished map. Missing parameters,
explicit zero offsets, an inactive pre-start iteration, and a disabled
injection must retain the existing centered-kernel path exactly. The diagnostic
remains unavailable when kernel generation is disabled.

The offset identity must be recorded in effective configuration, processed-
timestream provenance, NetCDF configuration variables, output FITS headers,
and diagnostic logs. This changes no FRUIT recurrence, learned-state rule,
penalty threshold, mask, weight, filter, or production default.

## Required compatibility evidence before execution

1. A focused unit test must demonstrate exact sample-for-sample equality
   between the legacy centered kernel/injection path and the generalized path
   with explicit zero offsets.
2. A focused unit test must demonstrate the declared displacement and preserve
   the pristine kernel during signal injection.
3. The complete CTest suite and required configuration preflight must pass.
4. The new no-injection control must reproduce every corresponding EL-F4
   observation-123424 complete-map signal, kernel, and weight image bitwise at
   iterations 0--5 before the off-source result is inspected.

Failure of any item invalidates the experiment. It does not trigger retuning.

## Fresh-run matrix

Both trajectories start from the same raw observation-123424 inputs with one
configured thread, `grppiex: seq`, the same newly frozen executable,
`alpha=1.25`, unchanged complete-map detector-penalty evidence, saved
diagnostics/checkpoints, and absolute iterations 0--5.

| Order | Variant | Injection position | Amplitude |
|---:|---|---|---|
| 1 | no-injection control | not applicable | none |
| 2 | off-source injected | `(0, -60)` arcsec | 100 mJy/beam in each array |

The existing EL-F4 centered complete-map injected trajectory remains the
centered comparison. It is not subtracted from the new off-source trajectory.
A fresh centered trajectory is permitted only if a compatibility check fails
or if the off-source result leaves a specific ambiguity that direct
replication can resolve. Such a trajectory must be registered before it is
run.

## Measurements and interpretation

For each iteration and array, form the paired response

`T_k = signal_I(off-source injected, k) - signal_I(new control, k)`.

Fit and residual measurements are centered on the declared injection position
and normalized by the same-iteration off-source `kernel_I` and the 100
mJy/beam injected truth. Record the complete injected-minus-control response,
not just its local fit.

At the iteration-4 boundary, compare the paired learned detector-penalty
inventories, including the exact scan-5 a1400 UID 4460 record and every other
injection-specific factor-zero detector penalty. Report the iteration-4 to
iteration-5 response change and compare it descriptively with the existing
centered evidence.

The valid interpretations are:

- **same event replicated off source:** the same injection-specific UID 4460
  record and associated response-loss direction recur, strengthening but not
  proving a location-general mechanism;
- **centered event not replicated:** the UID 4460 difference is absent and no
  comparable injection-specific hard-penalty association appears, supporting
  a location/geometry-specific interpretation for the current evidence;
- **different penalty association:** another injection-specific penalty and
  response change appears, requiring a separately registered causal test
  before attribution; or
- **inconclusive:** compatibility, coverage, transfer-fit, execution, or
  product-completeness requirements fail.

Even a replicated association is not automatically causal and does not make
the injected increment an independently calibrated sky product.

## Stop rule and claim limits

Stop after the two trajectories and registered comparison. Do not tune the
location, amplitude, thresholds, masks, weights, filters, or recurrence after
opening results. Do not automatically run NGC4449, subtract Neptune, repeat a
centered trajectory, implement a new penalty policy, or launch qualification.

