# SCI-CAL-001 atmosphere regeneration specification v1

## Status and identity

This is a reproducible **partial-recovery specification**, not a claim that the original atmosphere calculations have been regenerated. It is bound to repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and repair-line evidence head `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`.

Complete q25, q50, and q75 raw grids are locally recoverable and digest identified. The q95 raw grid, original `am` executable/configuration, atmospheric profiles, and profile-construction provenance are absent. Therefore the raw-generation stage is fail-closed and has no executable command until the items in `owner_input_request.json` are supplied.

`regeneration_manifest.json` is the machine-readable state record. It validates against `atmosphere_regeneration_manifest.schema.json`; unresolved scientific facts are explicit identifiers rather than guessed values.

## Recovered raw artifacts

| Model | Local artifact | SHA-256 | TolTECA registry MD5 | Status |
| --- | --- | --- | --- | --- |
| q25 | `amLMT25.npz` | `6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b` | `008d7fa69aff187a9edf419f3d961b4c` | complete local bytes |
| q50 | `amLMT50.npz` | `1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81` | `6ec393672be8af4dfa06a3f4cf9aa32e` | complete local bytes |
| q75 | `amLMT75.npz` | `adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e` | `d6cf4bb27008179ec491864388deac58` | complete local bytes |
| q95 | TolTECA datafile ID 461 | unknown until supplied | `0ca7b331823237767d26016d19bffb3d` | missing; endpoint not contacted |

The recovered NPZ schema is:

- `el`: float64, shape `[31]`, degrees, exactly 20 through 80 in steps of 2;
- `atmFreq`: float64, shape `[50001,31]`, GHz, each column exactly 0 through 500 in steps of 0.01;
- `atmTtx`: float64, shape `[50001,31]`, dimensionless atmospheric transmission as named by the source artifact;
- `atmTRJ`: float64, shape `[50001,31]`, Rayleigh-Jeans atmosphere temperature in kelvin.

The exact transmission reference plane and the original `am` directives that generated these arrays remain requested facts. The recovered values must not be relabeled from a contemporary `am` manual without historical configuration evidence.

## Exact recovered legacy derivation

For each available q model and each raw elevation node, select the rows at exactly 225.00 GHz and at the recovered nominal band frequency:

| Band | Nominal sample frequency |
| --- | ---: |
| a1100 | 272.73 GHz |
| a1400 | 214.29 GHz |
| a2000 | 150.00 GHz |

Define

```text
R_q,b(e) = atmTtx_q(nu_b,e) / atmTtx_q(225 GHz,e).
```

Fit `R_q,b(e)` with `numpy.polyfit(elevation_radians, R, 6)` over all 31 nodes and round each coefficient to eight decimal places. This exactly reproduces all 63 repair-base q25/q50/q75 coefficient literals. Consequently the legacy coefficient lineage is monochromatic; none of the available TolTEC passband arrays participates in this fit.

The selector coordinate is zenith `tau225`. For recovered anchors it is verified as

```text
tau225_q = -log(atmTtx_q(225 GHz, 80 deg)) / X(80 deg).
```

Here `X(80 deg)=1.01538872688246729e+00` under the repair-base modified-secant formula. The resulting values are `5.04874104674104401e-02`, `8.83393725904400573e-02`, and `1.58313198574890929e-01` for q25, q50, and q75. The repair-base q95 coordinate `3.04868387190534607e-01` is derived the same way from source literal `0.7337698`; it remains unverified against q95 raw bytes.

To reproduce and verify the recovered fit from the repository root:

```sh
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity
$HOME/tolteca/bin/python validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity --check
```

The first command writes normalized deterministic tables only; it does not modify the sibling repository or the NPZ inputs.

## Full raw-grid regeneration protocol

Once the owner-supplied package exists, perform these steps in order:

1. Validate its manifest against `atmosphere_regeneration_manifest.schema.json`. Reject null or unresolved generator, profile, site/geometry, grid-directive, and q95 fields.
2. Verify every input byte stream against both its manifest SHA-256 and the package `SHA256SUMS`. Preserve the original bytes without normalization.
3. Record the exact OS/container, compiler, `am` source or executable digest, dependency versions, locale, floating-point type, argv array, environment variables, working directory, and stdout/stderr.
4. Execute only the supplied versioned argv. Do not replace profiles, site defaults, spectral settings, or output directives from memory or current software.
5. Write each raw output atomically, record byte count and SHA-256, and inventory member names, dtype, shape, units, coordinate ordering, and non-finite policy.
6. Compare regenerated q25/q50/q75 bytes and values with the recovered artifacts before inspecting q95. Differences are evidence to report, not values to coerce.
7. Normalize all four models into the same long-table fields preserved in `recovered_raw_nominal_grid.csv`; retain the immutable raw outputs separately.
8. Freeze operator candidates and held-out nodes before reading independent intermediate-run results.

The historical text says 10--80 degrees in 10-degree steps, while the recovered arrays and plotted nodes are 20--80 degrees in 2-degree steps. The recovered arrays govern the reproduced coefficient calculation. `GRID-001` requires provenance to resolve the historical wording discrepancy for a full generator rerun.

## Versioned continuous-operator candidates

For a sample with valid zenith `tau225` and aligned elevation `e`, evaluate the q-anchor band transmission with the top-of-atmosphere pivot:

```text
X(e) = sec(z) * (1 - 0.0012 * (sec(z)^2 - 1))
T_q,b(e) = R_q,b(e) * exp(-X(e) * tau225_q)
L_q,b(e) = -log(T_q,b(e))
C_q,b(e) = exp(L_q,b(e)).
```

The full sample airmass is applied to zenith `tau225`; no Beammap-relative or reference-airmass subtraction is permitted. A continuous candidate interpolates the complete line-of-sight optical depth `L`, then returns `T=exp(-L)` and correction `C=exp(L)`.

All candidates share the owner-approved low-opacity definition exactly:

```text
L_b(tau,e) = (tau/tau_q25) * L_q25,b(e),  0 <= tau <= tau_q25.
```

Above q25, the preregistered candidates are:

- `piecewise_linear_los_tau_v0`: adjacent-anchor affine interpolation in zenith `tau225`; primary candidate because it adds no unmeasured curvature;
- `pchip_los_tau_v0`: shape-preserving PCHIP through q25/q50/q75/q95; challenger;
- `cubic_through_anchors_los_tau_v0`: unconstrained exact-anchor stress test, not recommended for selection.

Requests outside a declared opacity/elevation domain, non-finite inputs, invalid aligned-elevation eligibility, or absent model support fail closed. There is no extrapolation.

## Selection and validation gates

Candidate evaluation must report, by band and over the complete owner-declared domain:

1. exact-anchor line-of-sight-tau error and low-opacity identity error;
2. finite values, `L>=0`, `0<T<=1`, and `C>=1`;
3. continuity at every anchor, including one-sided binary64 `nextafter` probes;
4. nondecreasing `L` with increasing zenith opacity;
5. nonincreasing `L` with increasing elevation, with the known q95/a2000 feature reported separately rather than hidden;
6. fractional extinction-correction error `abs(exp(L_candidate-L_truth)-1)` at raw anchors and preregistered independent intermediate/held-out runs;
7. maximum, 95th percentile, median, RMS, signed range, and error location.

At most one-percent correction error is a provisional numerical representation-fidelity gate. It is not a claim of per-sample physical photometric accuracy. Software correctness, representation fidelity, 5--10% absolute observational accuracy, and approximately 5% observation-to-observation repeatability are separate gates. More samples do not remove shared calibrator, Beammap-extinction, selector, or airmass systematics.

No operational domain is declared by this package. The current 30--80 degree dense grid is diagnostic, and the recovered raw arrays alone support only 20--80 degrees. q95, intermediate runs, and owner domain approval are required before selection.

## Aligned-elevation dependency

`SCI-CAL-001-XAUD-001` remains an open, held-for-CAL-re-audit dependency. Atmosphere evaluation may consume aligned elevation only with explicit ordered sample identity, timing gap/interpolation origin, acquired duration, and original-versus-synthesized eligibility. This constraint changes neither the atmosphere equations nor this regeneration scope.
