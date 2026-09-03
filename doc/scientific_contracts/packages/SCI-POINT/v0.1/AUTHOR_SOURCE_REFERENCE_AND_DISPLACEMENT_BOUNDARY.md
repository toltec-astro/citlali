# SCI-POINT Source Reference And Displacement Boundary

Identity: `SCI-SOURCE-REFERENCE_TO_SCI-POINT v0.1/r0.3`

## Four Distinct Objects

1. authoritative expected source position;
2. parent-map WCS/reference origin;
3. requested and effective algorithmic search center; and
4. fitted source centroid.

They may coincide numerically but never share identity by implication.

## Expected-Position Record

The record shall bind source identity, target/catalog/ephemeris authority,
apparent-position convention, applicable time or epoch relation, frame,
refraction and aberration convention where applicable, morphology/reference
origin, position uncertainty, validity interval/domain, lifecycle, provenance,
and typed availability/failure.

## Measurement

In the exact declared ordered AltAz tangent basis,

`Delta_POINT = mu_fitted - mu_expected`.

If the tangent-coordinate origin is exactly the expected source position,
`mu_expected = (0,0)` and `Delta_POINT = mu_fitted`. POINT owns this
measurement sign. The pointing-support producer owns any later
measurement-to-correction sign and telescope-offset composition.

The search center affects search and initialization only. Changing it cannot
change the measurement zero point. A fallback peak, finite fit, or high
diagnostic ratio does not establish association with the known source.

If the expected-position authority, exact tangent transform, or source
association is unavailable, the source-attributed displacement is unavailable
even when a numerical Gaussian fit exists.

The boundary also carries the typed applicability facts `known`, `isolated`,
`bright`, and `approximately_centered`, each with value/state, exact authority
or method, domain, cause, validity, lifecycle, and provenance. Missing tests or
thresholds remain unavailable and are not inferred from observing-mode labels.
