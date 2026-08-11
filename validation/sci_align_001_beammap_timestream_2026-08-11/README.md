# SCI-ALIGN-001 direct Beammap timestream fit

This bounded diagnostic applies the validated pointing-observation `t + tau`
fit concept to a Beammap full-PTC product.  It is intentionally not an
unchanged invocation of the pointing tool: detector-grouped Beammap maps use
telescope-plus-pointing coordinates and suppress physical APT offsets.

The adapter therefore evaluates telescope-plus-pointing at `t + tau` against
each detector's standard-APT `x_t_raw/y_t_raw` center.  Per-detector widths and
angles also come from that same standard APT.  Left/right APTs are prohibited.
The 100-detector cohort is selected without directional information by the
same network-balanced standard-only rules used for the split-map hero review.

The baseline stage reports point estimates for a static model, common lag,
scan-axis hysteresis, and their joint model, plus the lag objective profile.
Uncertainty and corpus inference are deliberately deferred until this adapter
passes synthetic recovery and one checksum-frozen real Beammap comparison.

The sign comparison is explicit.  The retained map diagnostic reports
`(right centroid - left centroid)/(right rate - left rate)`.  A signal modeled
at the complete coordinate `t + tau` produces the negative of that map slope,
so the direct `tau` is compared with the negative of the existing map value.
This is a sign conversion, not a factor of two.

The required input is a repaired full-PTC `*_ptc_timestream.nc`, not the sparse
per-detector source-crossing sidecar.  The sparse sidecar does not retain the
complete telescope trajectory needed for exact within-scan `t + tau`
interpolation.

Revision 2 records a numerical repair prompted by the first real anchor, not
by its fitted timing value.  With native Beammap residuals near `1e-6`, every
L-BFGS-B fit stopped at iteration zero because the unscaled objective was near
`8e-15`; an explicit tau grid simultaneously showed that the returned start
was not the objective minimum.  All objectives are now divided by one fixed,
pre-fit PTC-weighted signal-energy scale.  This dimensionless scaling cannot
change an optimum or model ordering and is reported in the result manifest.

The same anchor then exposed a second numerical failure before interpretation:
the spatial coordinates moved, but `tau` remained exactly at a multistart seed
even though the independent objective profile was lower elsewhere.  The
scalar `1e-4` finite-difference step meant `1e-4 ms` for `tau`, which is too
small for the float-valued Beammap signal.  Revision 3 uses the pointing
fitter's already-validated unit-aware `0.01 ms` tau step while retaining
`1e-4 arcsec` for spatial and hysteresis coordinates.
