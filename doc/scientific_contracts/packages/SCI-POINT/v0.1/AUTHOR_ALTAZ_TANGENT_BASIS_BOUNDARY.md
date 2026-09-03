# SCI-POINT AltAz Tangent-Basis Boundary

Identity: `SCI-AST_TO_SCI-POINT_COORDINATE_BOUNDARY v0.1/r0.3`

The exact boundary instance shall bind:

- tangent-plane reference direction;
- two named ordered basis vectors `(e_1,e_2)`;
- angular units, handedness, and positive direction of each axis;
- relation to parent WCS, pixel coordinates, and physical tangent metric;
- applicable observation time and coordinate-state identity;
- map-to-tangent transformation, including Jacobian where relevant;
- validity/support, uncertainty, lifecycle, provenance, unavailable state, and
  failure.

No axis label alone establishes raw delta-Az, cross-elevation, elevation, or
another tangent component. Memory order, display orientation, and FITS array
order are not basis or sign authority.

For parent-map coordinate `p_q`, the exact boundary supplies physical tangent
coordinate `u_q = T(p_q; state)`. The source model is evaluated using the
physical tangent-plane metric on `u_q`, not storage-pixel distance, unless the
boundary explicitly proves equivalence over the admitted fit support.

When a fit occurs in another coordinate basis, displacement and uncertainty
transformation require the exact transformation and all covariance terms
needed by it. Transforming marginals alone through a non-axis-aligned Jacobian
does not produce exact AltAz marginal errors.
