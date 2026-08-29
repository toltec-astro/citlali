# SCI-NOI v0.1 — STD Numerator, Scale, And Claim Table

Artifact identity: `SCI-NOI_STD_NUMERATOR_SCALE_CLAIM v0.1/r0.1`

Status: proposed sanitized Stage A scientific input; exact bytes await owner
approval

| Field | Required exact declaration | Unavailable or prohibited inference |
| --- | --- | --- |
| Numerator | Immutable exact MAP or JINC product, estimand, operation, generation, unit/beam, response reference, WCS, support, validity, and lifecycle | A bare plane, filename, STOKES token, or equal shape is not a numerator identity |
| Scale parent | Exact authorized UNC product/method/generation and whether it supplies standard deviation, standard error, projected uncertainty, calibrated scale, or another quantity | Variance, covariance, inverse variance, precision, and consumer weight are not direct denominators |
| Scale transformation | Every exact square root, projection, calibration, or other operation needed to create the direct scale | A unit-compatible number is not sufficient without the authorized transformation |
| Direct denominator | Authorized finite strictly positive scale in the exact numerator signal unit | Zero, negative, nonfinite, unavailable, or incompatible scale makes STD unavailable on that domain |
| Compatibility | Estimator, response reference, parent relation, unit/beam, WCS, support, validity, lifecycle, and product generation | No interpolation, domain extension, or parent substitution by analogy |
| Dependence | Exact dependence between numerator and estimated scale and its consequence | Standardization does not create independence or a pivotal statistic |
| Local behavior | Exact outside-support, missing, zero, negative, nonfinite, unavailable, and conflict behavior | Invalid division is unavailable, never numeric zero or infinity |
| Output | `empirical_scale_standardized_signal`, exact support/lifecycle/provenance, and unit `1` | Dimensionless does not mean probability, significance, or calibration |
| Claim | “standardized by the stated empirical scale” | No Gaussian-z, Student-t, N-sigma, false-alarm, detection, completeness, purity, or catalog claim |

For compatible numerator `q` and direct empirical scale `sigma_q`,

```text
empirical_scale_standardized_signal = q / sigma_q
unit(empirical_scale_standardized_signal) = 1.
```

The output is dimensionless but remains only standardized by the exact stated
empirical scale. Gaussianity, pivotal behavior, Student-t behavior, a z-score,
N-sigma significance, detection probability, false-alarm rate, completeness,
purity, and catalog eligibility require separately authorized null, selection,
search, multiplicity, decision, and validation authority.

The Stage A numerator/scale pair remains an owner decision. A normalized MAP
signal or a numerically realized `jinc_map` is a candidate numerator; a
standard deviation, standard error, projected uncertainty, calibrated scale,
or another transformed positive signal-unit UNC quantity is a candidate scale.
No candidate pairing becomes available by appearing in this table.
