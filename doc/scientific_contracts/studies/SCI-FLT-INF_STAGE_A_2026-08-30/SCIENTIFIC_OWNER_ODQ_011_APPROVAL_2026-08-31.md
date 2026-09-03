# SCI-FLT-INF-ODQ-011 scientific-owner approval

Decision identity: `SCI-FLT-INF-ODQ-011`

Decision date: `2026-08-31`

Scientific owner: G. Wilson

Status: **Option 1 approved; ODQ-011 closed**

## Approved selection and fallback policy

Base v0.1 has no automatic method selector and no fallback. It either realizes
the exact requested matched-template method or records an explicit unavailable
or failed state.

Missing, invalid, incompatible, or failed parent, template, PSD/`Q`, learning,
support, normalization, response requirement, or approximation qualification
does not authorize constant-spectrum substitution, convolution/low-pass
substitution, another template or coefficient, relaxed support, another
approximation, destriping, mode removal, or retention of the requested method
identity over a different operation.

A user may explicitly request a separately authorized method before execution.
That is ordinary method selection, not fallback. Any future automatic selector
is a separately contracted policy that binds its request, candidates,
selection rule and inputs, realized method, state, response, uncertainty, and
failure. Its output retains the realized underlying method identity.

Data-thresholded spectral selection and the inactive destriping family are
excluded from this package and deferred to their own Stage A recovery. They
require separate estimand, selector, response, covariance, support, NOI,
lifecycle, and failure authority.

This decision rejects historical silent substitution and authorizes no
alternative numerical route or Stage B launch.
