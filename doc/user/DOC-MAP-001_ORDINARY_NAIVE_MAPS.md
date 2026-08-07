# DOC-MAP-001: Ordinary Naive Science Maps — Phase A

Status: `validated_bounded`

Applies to: ordinary `naive`, array-grouped, Stokes-I observation and coadd
maps under `sci-map-001-f010-v1`

Production disposition: `existing_use_only`

## What And Why

An ordinary Citlali science map is a package of aligned planes, not a signal
image with optional bookkeeping. The package records the normalized sky
estimate, its realized response, the coefficient used to normalize it, and
separate facts describing geometric opportunity, admitted data, exposure,
numerical support, science-policy support, and final validity.

These distinctions matter because a finite pixel is not necessarily usable, a
well-covered pixel is not necessarily valid, and a coefficient with inverse-
squared signal units is not automatically an inverse variance. The package
lets an astronomer identify the pixels and companion products that the
pipeline actually authorizes for interpretation without reconstructing those
decisions from historical `coverage` names.

An observation map combines eligible detector samples. A coadd map combines
complete admitted observation-map packages. Both use the ordinary positive-
coefficient normalization described by
[SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001](../science/SCI-METHOD-WEIGHTED-MAP-NORMALIZATION-001.md).

## Product Package

All applicable planes share the map shape and WCS. Required companions and
their identities are recorded in realized provenance.

| Plane | Unit | What it measures |
| --- | --- | --- |
| `signal_I` | `mJy/beam` | Ordinary positive-coefficient normalized Stokes-I map value |
| `weight_I` | `1/(mJy/beam)^2` | Realized gridding and normalization coefficient; nonprecision by default |
| `kernel_I` | `mJy/beam` | Response/kernel companion accumulated and normalized with the signal |
| `geometric_hits_I` | count | Finite in-bounds detector/sample projections before upstream eligibility and estimator selection |
| `contributing_hits_I` | count | Terms admitted by the ordinary estimator contribution rule |
| `coadd_observation_count_I` | count | Admitted observation maps contributing to the coadd pixel; absent from observation maps |
| `upstream_eligible_exposure_I` | detector s | Projected detector-seconds eligible before estimator and normalization retention |
| `retained_exposure_I` | detector s | Detector-seconds retained after contribution and normalization-support decisions |
| `normalization_support_I` | binary | Whether numerical normalization/population is authorized |
| `science_policy_support_I` | binary | Whether the separate full-cut science-support policy passes |
| `science_valid_I` | binary | Authoritative raw science validity |

Two compatibility names remain:

- `coverage_I` is exactly `retained_exposure_I`. It is detector-seconds, not
  wall-clock time, precision, confidence, support, or validity.
- `coverage_bool_I` is the deprecated exact alias of
  `science_policy_support_I`. It is not authoritative validity.

## Which Pixels Are Scientifically Valid?

Use `science_valid_I`. A raw pixel is valid only when all of the following are
true:

1. numerical normalization is supported;
2. the separate science-support policy passes;
3. `signal_I` and every declared required companion are finite; and
4. the complete product identity was admitted.

Do not substitute `weight_I`, either exposure plane, either support plane,
either compatibility alias, or a generic finiteness test. A downstream
operator may produce a finite value where the raw map was invalid, but it
cannot promote that location to raw science validity. Invalid map edges are
not scientific pixels and need only be prevented from contaminating valid
regions.

## Observation Coaddition

An observation enters a coadd atomically as one ordered package. Array and
Stokes identity, units, response, required companions, full-precision WCS,
shape, support/validity policy, and provenance must all match before any coadd
state changes.

The only supported placement in this contract is centered integer embedding
on the same common grid. The observation and coadd shapes must differ by even
numbers of rows and columns, and the full-precision WCS must identify the same
world coordinate after the corresponding integer reference-pixel offset.
There is no fractional shift, interpolation, reprojection, source recentering,
or mean/null-mode subtraction: the signal-centering operator is `L = I`.

Signal, kernel, declared realizations, retained exposure, and observation
count use the same admitted membership and embedding. A package mismatch or an
unexpected non-finite required contribution fails before partial admission.

## Configuration That Changes Meaning

- `mapmaking.method` must resolve to ordinary `naive` for this contract.
- Grouping must resolve to array grouping and the component must be Stokes I.
- `coverage_cut` participates in distinct normalization-support and
  science-policy-support thresholds. The exact realized thresholds and
  algorithm identities are recorded in the lossless sidecar.
- An enabled global empirical rescaling changes the realized coefficient stage
  and must be recorded. It does not by itself establish precision.
- If map filtering follows, the validated raw package remains an immutable
  parent identified by one `RAWPDGST`. Filtered-product interpretation is not
  defined by this Phase-A guide.

Requested configuration alone is insufficient for interpretation. Use the
effective and realized provenance associated with the reduction.

## Supported And Unsupported Interpretations

Within `science_valid_I`, the package supports interpretation of the ordinary
normalized map and its matched response/kernel under the recorded units,
calibration, WCS, and coefficient state. Coadds additionally support the
recorded integer-aligned combination of admitted observation packages.

This contract does **not** establish:

- that `weight_I` is inverse variance or that `1 / weight_I` is map variance;
- a pixel uncertainty, covariance model, GLS solution, or statistical
  significance;
- a JINC contribution rule or JINC versions of the F010 planes;
- detector- or network-grouped F010 products;
- general reprojection, interpolation, or fractional astrometric correction;
- filtered, convolved, source-finding, or fruit-loop product semantics; or
- upstream production eligibility.

Calibration, response, astrometry/WCS, coefficient/covariance, and upstream
eligibility conclusions remain conditioned on `SCI-CAL-001`,
`SCI-ALIGN-001`/`SCI-AST-001`, `SCI-PTC-001`, and `SCI-VAL-001`. This is why
the bounded MAP contract is accepted while production remains
`existing_use_only`.

## Provenance And Validation

The lossless sidecar is authoritative for full-precision bundle identity,
WCS admission, realized thresholds, coefficient stage, membership, offsets,
required companions, and raw-parent/product digests. FITS headers provide
required user-facing identity and convenience values but do not replace that
sidecar authority.

Authoritative references:

- [ADR 0009](../adr/0009-science-map-bundle-admission-and-validity.md)
- [Scientific conventions](../SCIENTIFIC_CONVENTIONS.md#science-map-bundle-identity-and-coaddition)
- [Executable product contract](../../validation/product_contracts.json)
- [Application-integration decision](../../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md)
- [Focused contract tests](../../tests/test_science_map_contract.cpp)
- [Independent equation/truth tests](../../tests/test_science_map_truth_suite.cpp)
- [Production FITS tests](../../tests/test_science_map_fits_products.cpp)
