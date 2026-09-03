# SCI-POINT Response And Bias States

Identity: `SCI-POINT_RESPONSE_BIAS v0.1/r0.3`

POINT is nonlinear when search, fallback, support, or active constraints depend
on data. Three authorities remain distinct:

1. `POINT-FIXED-BRANCH-RESPONSE-STATE`: conditional on one exact seed/fallback
   branch, fit support, fit-weight state, and active-constraint set;
2. `POINT-FULL-PROCEDURE-RESPONSE-STATE`: reruns central search, peak initialization,
   fallback, fit-domain resolution, constraint activation, and optimization
   under perturbation; and
3. `POINT-OBSERVATIONAL-BIAS-ACCURACY-STATE`: separately established empirical
   behavior.

`POINT-SOURCE-ASSOCIATION-STATE` is separate from all three. These roles may
share one storage container only if identity, availability, cause, owner,
method, domain, and lifecycle remain lossless. Availability of any one does
not imply another.

Parent-map response is input state, not automatically the response of the fit
procedure. A local Jacobian is not full-procedure response. Until separately
authorized, both POINT response roles and observational bias/accuracy are
typed unavailable; their absence does not authorize zero response error or
zero centroid bias.
