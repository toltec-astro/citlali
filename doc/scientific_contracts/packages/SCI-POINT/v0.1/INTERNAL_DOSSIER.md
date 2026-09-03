# SCI-POINT Internal Implementation-Informed Dossier

Status: quarantined Stage A material; prohibited from implementation-blind
Stage B authorship

Date: `2026-09-02`

This dossier records what the manager recovered from implementation,
configuration, schemas, historical audits, validation, and operational
workflows. It helps avoid reinvention and exposes gaps. It is not scientific
authority, a conformance finding, a validation result, or an author input.

## Exact Citlali Recovery Base

Commit: `0b977a90a0bae6a68dadcf7824c9b7a0c80a7f45`

### Fitting path

- `include/citlali/core/engine/detail/pointing_fit_maps_impl.h`
- `include/citlali/core/utils/fitting.h`
- `include/citlali/core/utils/gauss_models.h`
- `src/citlali/core/utils/gauss_models.cpp`
- `include/citlali/core/config/post_processing_config.h`
- `include/citlali/core/pipeline/source_fitting_config_policy.h`

Observed behavior:

- one `mapFitter::pointing` call per array map;
- six-parameter zero-background elliptical Gaussian;
- weighted-peak initialization, central-radius option, global fallback;
- bounding box plus amplitude/FWHM/angle parameter bounds;
- pixel sigma `1/sqrt(weight)` for the Pointing mode;
- independent fit validity and marginal formal uncertainty; and
- sequential fits because the deployed Ceres covariance path is not used.

The scientific adequacy of those choices is unassessed here.

### Plan, product, and lifecycle path

- `include/citlali/core/pipeline/pointing_execution_plan.h`
- `include/citlali/core/engine/detail/pointing_setup_impl.h`
- `include/citlali/core/engine/detail/pointing_output_impl.h`
- `include/citlali/core/engine/pointing.h`
- `validation/product_contracts.json`

Observed behavior:

- raw-observation and filtered-observation stages;
- attempts expected once per array map when fitting is enabled;
- ECSV schema metadata `citlali-pointing-fit-v2`;
- row key is array identity;
- amplitude, centroid, FWHM axes, angle, marginal errors, and three diagnostic
  fields;
- fit values also appear in map FITS metadata; and
- Pointing reduction products include maps and other diagnostics that are not
  automatically POINT scientific outputs.

### Pointing/FRUIT and mode coupling

- `include/citlali/core/config/pointing_config.h`
- `include/citlali/core/engine/detail/pointing_fruitloop_impl.h`
- `include/citlali/core/engine/detail/pointing_map_population_impl.h`

Observed behavior includes `standard` and `psf_preserve` source strategies,
FRUIT source-center controls, and Pointing-specific recurrence plumbing. These
implementation names do not authorize OOF inference or a POINT-owned FRUIT
method. SCI-FRUIT retains recurrence authority; SCI-OOF will own OOF science.

## Current Product Vocabulary

The current fit-table fields are:

`array`, `amp`, `amp_err`, `x_t`, `x_t_err`, `y_t`, `y_t_err`, `a_fwhm`,
`a_fwhm_err`, `b_fwhm`, `b_fwhm_err`, `angle`, `angle_err`, `sig2noise`,
`peak_over_full_map_rms`, and `fit_sig2noise`.

This is a recovered implementation vocabulary, not an admitted scientific
alias set. The author packet uses `fitted_amplitude_over_full_map_rms` as the
canonical descriptive identity, retains `sig2noise` only as a legacy alias,
and does not admit `peak_over_full_map_rms` unless the approved compatibility
method establishes the required positive-peak interpretation.

`doc/SCIENTIFIC_CONVENTIONS.md` already defines the three diagnostic meanings.
The product schema notes a known Astropy column-unit debt. Stage A records the
scientific quantities without prescribing a file migration.

## Current Operational Consumers

### TolTECA

Authority reference inspected: `origin/main` at
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`, file
`tolteca/reduce/engines/citlali.py`.

Observed `_resolve_pointing_offsets` behavior:

1. choose Pointing tables before/after a target by observation number;
2. compute the arithmetic mean of all table-row `x_t` and `y_t` values;
3. account for telescope user and paddle offsets;
4. change sign from measured displacement to correction;
5. attach the Pointing-table MJD; and
6. publish an astrometry configuration record for later application.

This is the recovered operational wheel. Stage A does not determine whether
array admission, averaging, selection-by-obsnum, sign composition, or offset
adjustment belongs normatively to POINT or to a downstream producer.

### TolProj

Repository head inspected:
`539be7b5fcb0be573610f9e8783c3e6df8bfb5b1`.

Observed workflow roles:

- identifies and prepares pointing observations and exact APT parents;
- constructs pointing reductions;
- consumes recovered per-array Pointing amplitudes and reference-source fluxes
  in a downstream flux-scale calibration workflow; and
- preserves immutable matched APT parents while issuing distinct calibrated
  child products.

These are workflow and CAL/TolProj responsibilities, not POINT estimator
authority.

## Historical Evidence

- `doc/POINTING_COMPACT_EQUIVALENCE_2026-06-30.md`
- `doc/POST_PROCESSING_CONFIG_AUTHORITY.md`
- `handoff/HANDOFF_2026-06-18_POINTING_SOURCE_AWARE.md`
- relevant Pointing entries in `doc/REFACTOR_STATUS.md`
- Pointing product schemas and fit-metric tests under `validation/` and `tests/`
- historical `SCI-MODE-001-XAUD-001--003` and
  `SCI-SRC-001-XAUD-001` handoffs

These sources may inform later conformance, migration, and validation work.
They are excluded from Stage B scientific authorship.

## Dossier Conclusion

The current system is not missing a Pointing fitter. The missing work is a
scientific contract that names its estimand, parent, support, uncertainty,
products, acceptance, aggregation, and downstream boundary without silently
endorsing incidental code or redesigning mature behavior.
