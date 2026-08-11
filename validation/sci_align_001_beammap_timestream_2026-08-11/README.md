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
