# SCI-CAL-001 AM 12.2 successor adoption-study protocol

## Status and scope

This protocol is frozen on 2026-08-01 before the successor calculations are
launched. It implements the bounded evaluation authorized by
`OWNER_DIRECTION_AND_PROVENANCE_CORRECTION_2026-08-01.md`. No result exists at
registration time, and this file does not itself adopt a model or authorize an
operational domain.

The study asks one question: does a separately versioned, band-integrated AM
12.2 operator add no more than 1% fractional extinction-correction error over
the intended q0--q75 TolTEC regime while satisfying the required structural
gates? It does not attempt to regenerate or identify the historical generic-q
lineage and does not evaluate observational absolute calibration.

## Frozen support and physical convention

The exact study support is:

```text
tau0  = 0
tau25 = 0.0504874104674104401
tau50 = 0.0883393725904400573
tau75 = 0.158313198574890929
20 deg <= elevation <= 80 deg
```

The three arithmetic-opacity holdouts are:

```text
midpoint(tau0,  tau25) = 0.02524370523370522005
midpoint(tau25, tau50) = 0.0694133915289252487
midpoint(tau50, tau75) = 0.12332628558266549315
```

No q95 value participates. There is no extrapolation, clamping, or nearest-q
selection outside the closed support.

`tau225` is zenith optical depth. The frozen modified-secant coordinate is

```text
secz = 1 / sin(elevation)
X(elevation) = secz * (1 - 0.0012 * (secz**2 - 1))
X(80 deg) = 1.01538872688246729
tau225 = -log(T225 at elevation 80 deg) / X(80 deg)
```

The sample correction has top-of-atmosphere pivot `X_ref=0`. Direct AM
transmission at the sample elevation is truth. An implementation expressed as
zenith-equivalent band opacity must apply the complete sample `X(elevation)`;
it may not apply zenith opacity as line-of-sight opacity.

## Immutable AM inputs

The calculation family is the copied AM 12.2 source/input suite described in
`copied_am_manifest.json` and `REGENERATION_SPEC.md`:

- copied source payload aggregate SHA-256
  `0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8`;
- copied Linux AM 12.2 executable SHA-256
  `3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c`
  as suite-custody evidence only; and
- independently built GCC-15 AM 12.2 evaluation executable SHA-256
  `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`.

The only permitted AMC inputs are:

| Profile | Role | SHA-256 |
| --- | --- | --- |
| `LMT_DJF_5.amc` | low-opacity holdout | `fcb3b70f44cad98cf0586fede9dcd3b2e35f3cb45023d0485c782c108b25b474` |
| `LMT_DJF_25.amc` | training and holdout | `aeeeeb48bef422f2d9392b5d7a3d62ab1887fd9e7c10322d5246d914841ba866` |
| `LMT_DJF_50.amc` | training and holdout | `d7c256d04d922beb51c9f8ab715e5be1a962252580eff2d08ba1be4d206eb5b0` |
| `LMT_DJF_75.amc` | training and holdout | `b63503c7f4170404d18f3797735b64fb947ce73eed35f0315155d0a29d499721` |
| `LMT_annual_25.amc` | high-opacity-interval stress holdout | `a9524553a5808a549eb18046a9ed6f8bd67ca1e29ccd1c91e05b351b64ea23e6` |
| `LMT_MAM_25.amc` | high-opacity-interval stress holdout | `82ac1e2a49a528244c1571daadcc8d42bd6d13c0ba8a7b5d2f81d10ebc13caee` |

Profile bytes, pressure/temperature/H2O/O3 layers, AM source, and bandpass
bytes are immutable. The only atmosphere-generation parameter varied is the
existing AMC `Nscale troposphere h2o` argument. Every scale search and run
must record its exact input digest, scale as a round-trip decimal and binary64
hex value, argv, working directory, executable and execution-context digests,
return code, warnings, parsed-row count, and raw-output SHA-256.

Use the copied 0--500 GHz, 10-MHz grid. The existing resolution study bounds
its center-frequency effect well below 0.1%; changing the range or resolution
would create a different evaluation version.

AM return code 1 is acceptable for this study only under the bounded existing
warning contract: all 50,001 numeric rows are present, the only AM warnings
are the known unresolved-narrow-line records and summary, and there are no
cache-mutation warnings, unknown warning classes, or error lines. Such a run
is warning-bearing evidence, not a clean software success. Any violation
invalidates the study.

## Frozen anchor constructions

Every nonzero anchor must match the exact repair-base 80-degree transmission
literal (`0.9500275`, `0.9142065`, or `0.8515054`) used to derive its frozen
`tau225` coordinate. Reuse the canonical 48-bisection parsed-transmission
plateau method from diagnostic P1. The scale and both inside/outside plateau
bounds are recorded; the realized parsed T225 must equal its target literal
exactly. No fit uses a TolTEC-band residual.

### Model lane A — fixed DJF25 profile

Use `LMT_DJF_25.amc` at all three nonzero training anchors. The preregistered
H2O scales are:

| Anchor | Scale |
| --- | ---: |
| q25 | `1.04228717356798550` |
| q50 | `2.02963214820032256` |
| q75 | `3.72800975456677941` |

This lane holds the pressure, temperature, and non-H2O profile structure fixed
and makes `tau225` the only atmospheric state coordinate.

### Model lane B — conditioned DJF nodes

Use the DJF percentile profile nearest its named opacity anchor and apply only
the small scale adjustment required for the exact target:

| Anchor | Profile | Scale |
| --- | --- | ---: |
| q25 | `LMT_DJF_25.amc` | `1.04228717356798550` |
| q50 | `LMT_DJF_50.amc` | `1.02851665282420113` |
| q75 | `LMT_DJF_75.amc` | `1.01048455031671569` |

No profile selection or scale may change after holdout results are inspected.

At each training anchor, generate direct AM truth at even elevations
`20,22,...,80` degrees. The clear anchor is analytic:
`tau225=0`, line-of-sight optical depth zero, and transmission one at every
frequency and elevation.

## Spectral convention and passbands

The primary passbands are TolTECA package-data version
`v1.0.0-tolteca_package_data` at repository commit
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`:

| Band | Artifact | SHA-256 |
| --- | --- | --- |
| a1100 | `tolteca/data/cal/toltec_passband/data/a1100_passband.ecsv` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| a1400 | `tolteca/data/cal/toltec_passband/data/a1400_passband.ecsv` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| a2000 | `tolteca/data/cal/toltec_passband/data/a2000_passband.ecsv` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

Use their `f` column in GHz and dimensionless `throughput` column exactly.
Linear interpolation of AM transmission onto a passband node is permitted
only between adjacent 10-MHz AM nodes; no spectral extrapolation is allowed.
Integrate with the composite trapezoid rule on the immutable passband nodes.

The read-only FTS challengers are mapped by their recorded TolTEC network:

| Band | FTS artifact | SHA-256 |
| --- | --- | --- |
| a1100 | `FTS_5934_N0.npz` | `b72e9be7a4637adbfb5f2a6e131741a4a7b151effc03dea49410d0e56a5df74c` |
| a1400 | `FTS_9932_N9.npz` | `da440896f537545871ac0d026b5149aeb5ba356e2f613c934567ef20fde0fd36` |
| a2000 | `FTS_9903_N11.npz` | `b3d5a0b1a332d40b4cdc436cb4327e849afbce42cfd766108aacf6016e775e65` |

For the challenger central curve, use `fc` as GHz, retain nodes within the
corresponding primary ECSV frequency support, set negative `sc` samples to
zero, and integrate the resulting nonnegative curve by the composite
trapezoid rule. Do not smooth, baseline-subtract, use the uncertainty envelopes
as weights, or extrapolate. Preserve the signed raw arrays and report the
clipped-node count and clipped integral fraction. FTS results are sensitivity
challenges, not replacement passbands or historical-lineage evidence.

For each passband, evaluate four source spectra:

```text
S_alpha(nu) = (nu / nu_pivot)**alpha
alpha in {-1, 0, 2, 4}
nu_pivot = 272.73, 214.29, 150.00 GHz for a1100, a1400, a2000
```

The pivot cancels from normalized transmission but is frozen for reproducible
weights. Define band-effective transmission, line-of-sight optical depth, and
extinction correction as

```text
T_eff = integral R(nu) S_alpha(nu) T_AM(nu) dnu
        / integral R(nu) S_alpha(nu) dnu
lambda = -log(T_eff)
C = exp(lambda) = 1 / T_eff
```

Both numerator and denominator must be finite, the denominator must be
strictly positive, and `0 < T_eff <= 1`. Results retain passband identity,
spectral index, and integration rule; an alpha-specific result may not be
silently applied to a different source convention.

## Continuous-operator candidates

Construct each candidate independently for model lane A and model lane B,
each passband identity, each band, and each spectral index.

At every nonzero opacity anchor, represent `lambda(elevation)` with a
shape-preserving PCHIP through the 31 even-elevation raw nodes. This common
elevation representation is exact at its training nodes and is evaluated only
inside 20--80 degrees. Compare two opacity representations:

1. `am12_piecewise_linear_los_tau_eval_v0`: linear from the analytic clear
   anchor to q25 and piecewise linear between q25--q50 and q50--q75 in
   line-of-sight optical depth at the requested elevation.
2. `am12_pchip_los_tau_eval_v0`: the same owner-approved linear clear-to-q25
   segment, joined continuously at q25 to a shape-preserving PCHIP through
   q25, q50, and q75 in line-of-sight optical depth.

The implementation must pin the interpolation library/version and constructor
options and serialize sufficient coefficient or knot state for independent
reproduction. No fit degree, smoothing parameter, atmospheric nuisance
parameter, or post-result tuning is permitted.

## Held-out truth

No holdout contributes a training value. For each opacity interval, condition
every listed immutable profile independently to the exact arithmetic midpoint
and run AM directly at odd elevations `21,23,...,79` degrees:

| Opacity interval | Midpoint profiles |
| --- | --- |
| q0--q25 | `LMT_DJF_5`, `LMT_DJF_25` |
| q25--q50 | `LMT_DJF_25`, `LMT_DJF_50` |
| q50--q75 | `LMT_DJF_50`, `LMT_DJF_75`, `LMT_annual_25`, `LMT_MAM_25` |

The scale solve uses only 225-GHz/elevation-80 transmission. It must be frozen
before any band-integrated odd-elevation output is inspected. The requested
midpoint coordinate, achieved parsed T225, achieved coordinate, scale plateau,
and coordinate residual must all be reported. The achieved coordinate must be
within one half of the final represented AM line-of-sight-tau decimal place
after conversion by `X(80 deg)`; otherwise that holdout is invalid and the
coverage gate fails. It may not be shifted to a more convenient coordinate.

These are joint opacity-and-elevation holdouts. In addition, report training
anchor residuals at the 31 even nodes; they are structural checks, not
independent fidelity evidence.

## Exact gates

All tolerances below are frozen before execution. A skipped, missing, or
out-of-support required row is not a pass.

### G0 — provenance and execution integrity

- Every source, AMC, passband, executable, script, and generated artifact
  matches a recorded SHA-256.
- Every expected run has one immutable execution sidecar and complete numeric
  output under the bounded warning contract.
- Locale, host, compiler/build identity, argv, environment, cache topology,
  and all package/library versions are recorded.
- Replaying artifact construction from validated raw cache produces identical
  committed tables, reports, manifests, and digests.

Any failure invalidates the study; later numerical gates are not interpreted.

### G1 — anchor conditioning and exact reproduction

- Every q25/q50/q75 training run has parsed 80-degree T225 exactly equal to
  its frozen target literal.
- At every even elevation, candidate evaluation at each training `tau225`
  reproduces its own band-integrated anchor `lambda` to absolute error
  `<= 1e-12`.
- For `0 <= tau225 <= tau25`, both candidates reproduce
  `lambda=(tau225/tau25)*lambda_q25` to absolute error `<= 1e-12`, including
  exact zero and q25 endpoints.

### G2 — finite domain and positivity

On all structural grids, anchors, and holdouts, `lambda` and `C` are finite,
`0 < T_eff <= 1`, `lambda >= 0`, and `C >= 1`. Numerical tolerance may excuse
only `lambda` values down to `-1e-12`, which must be reported and must not be
clipped before evaluation.

### G3 — continuity

Evaluate both one-sided binary64 neighbors of q25 and q50 plus offsets of
`1e-12` times the full opacity span. At every band, alpha, passband, and
elevation on a 0.1-degree grid, the maximum relative correction discontinuity
must be `<= 1e-10`. Analytic construction identity and numerical results are
both recorded.

### G4 — physical monotonicity

On a tau grid containing all anchors, all midpoints, and 1001 uniformly spaced
domain points, and an elevation grid from 20--80 degrees in 0.1-degree steps:

- `lambda` must be nondecreasing with increasing `tau225`; and
- `lambda` must be nonincreasing with increasing elevation.

The step tolerance is `1e-12` in line-of-sight optical depth. Report counts,
largest wrong-way step, and largest resulting correction excursion. There is
no q95-feature waiver inside this q0--q75 study.

### G5 — fail-closed support

Negative/non-finite opacity, non-finite elevation, elevations immediately
below 20 or above 80 degrees, opacity immediately above q75, invalid
passband/alpha identity, nonpositive passband normalization, and missing
brackets must return an explicit invalid/unsupported result. Clamping,
extrapolation, unity substitution, NaN publication, or partial correction is
a failure.

### G6 — primary numerical representation fidelity

For every TolTECA-ECSV holdout row, define

```text
fractional correction error = abs(exp(lambda_candidate - lambda_truth) - 1)
```

The maximum must be `<= 0.01` separately for every model lane, operator
candidate, band, and alpha, and over their combined required holdout set.
Report maximum, p95, RMS, median, and the exact profile/interval/scale/band/
alpha/elevation location. The maximum is the pass/fail statistic; summaries
cannot average away a failure.

### G7 — passband challenger disposition

Repeat G1--G6 independently with each mapped FTS challenger. In addition,
compare FTS and primary-ECSV truth corrections at identical physical runs.
If their maximum relative correction difference exceeds `0.01` in any band or
alpha, the AM representation may still be numerically valid, but an adoption
recommendation stops for an explicit owner passband choice. The discrepancy
may not be charged to interpolation error or averaged across bands.

### G8 — evidence coverage

Every three opacity intervals, all eight registered midpoint profile cases,
all 30 odd elevations, three bands, four alphas, and both primary and
challenger passbands must be present. Required row counts and a missing-key
anti-join are machine reported. Missing coverage fails rather than narrowing
the domain post hoc.

## Ranking and decision rule

A candidate is eligible only if G0--G6 and G8 pass and G7 does not require an
unresolved passband decision. Rank eligible model-lane/operator pairs by the
maximum primary-ECSV holdout correction error across every band, alpha,
profile, and odd elevation. A smaller maximum ranks first. If maxima differ by
no more than `1e-4` in fractional correction, prefer the fixed-DJF25 model lane
and then the piecewise-linear opacity operator because they introduce fewer
model identities and less interpolation structure. Report all component
metrics; do not create an unregistered composite score.

If no candidate is eligible, the AM 12.2 successor does not yet warrant
adoption over the declared domain. If one is eligible, the numerical package
may recommend that exact versioned model recipe, passband convention, alpha
contract, operator, and q0--q75/20--80 support for owner selection. A passing
study is not itself selection or application authorization.

## Separate observational gate and dependency

The provisional 1% threshold is representation fidelity, not physical
photometric accuracy. Before production, a separately preregistered,
human-run exact-repair-SHA campaign must evaluate approximately 5--10%
absolute flux accuracy, approximately 5% repeatability, and residual trends
with opacity and airmass. Calibrator, Beammap-extinction, beam, bandpass,
selector, and common airmass systematics remain explicit.

`SCI-CAL-001-XAUD-001` also remains open. Real aligned elevation is eligible
only with ordered sample identity, timing-gap/interpolation origin, duration,
and original-versus-synthesized status. This protocol neither resolves nor
works around that dependency.

## Required outputs and stop

Preserve a task-specific runner, immutable input manifest, execution-context
manifest, per-run sidecars/digests, scale table, raw-grid inventory, operator
coefficient/knot artifact, full holdout table, structural and fidelity metric
tables, concise decision report, machine decision object, and `SHA256SUMS`.
Verification must operate cache-only and launch no AM process.

Stop after committing the coherent study evidence. Do not modify Citlali
application code, contact Unity, launch repair implementation, launch the CAL
re-audit, or edit the coordination registry.
