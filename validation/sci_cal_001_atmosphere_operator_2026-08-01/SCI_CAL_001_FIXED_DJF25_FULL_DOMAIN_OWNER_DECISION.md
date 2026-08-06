# SCI-CAL-001 fixed-DJF25 full-domain operator decision

Owner-directed contract: one `am12_fixed_djf25_piecewise_linear_los_tau_v1`
operator over `0 <= tau225 <= .25` and `25 <= elevation_deg <= 80`, with no
selector or operator switch at `.15`. `LMT_DJF_25.amc` is a declared model
approximation, not inferred atmospheric truth. It is fail-closed outside
support, uses full sample airmass and `X_ref=0`, and requires finite positive
transmission.

The owner accepted the fixed-DJF25 `.15` comparison: 84 TolTECA-v1
band/alpha/elevation cases; maximum correction difference `0.0338295%`, p95
`0.0216573%`, RMS `0.0099135%`; a1100/a1400/a2000 maxima
`0.0338295%/0.0085375%/0.0144061%`. Value continuity is guaranteed by the
single shared `.15` anchor; derivative continuity is not a gate.

The machine contract binds the exact low fixed-DJF25 node artifact, TAU025
cache manifests, profile identity, anchors, interpolation, regimes, and
limitation that this is representation evidence—not observational calibration
accuracy, production adoption, or a runtime profile-selection rule.
