# SCI-POINT Diagnostic Formula Table

Identity: `SCI-POINT_DIAGNOSTIC_FORMULAS v0.1/r0.3`

| Canonical role | Conditional formula | Exact required authority | Current state | Claim ceiling |
| --- | --- | --- | --- | --- |
| `fitted_amplitude_over_full_map_rms` | `A_hat / RMS_full` | exact `POINT-FULL-MAP-RMS-METHOD v0.1`: parent/generation, full-map rows, about-zero versus centered rule, weighting, source-region inclusion, support/edge, non-finite behavior, zero denominator, unit cancellation, lifecycle/provenance, amplitude-parent relation | `unavailable_pending_separate_owner_approval` | descriptive dynamic range only |
| alias `sig2noise` | exact alias of preceding role | same record and identity | unavailable | not statistical S/N |
| `fitted_amplitude_over_formal_amplitude_error` | declared signed or absolute `A_hat / sigma_A` | exact amplitude role, approved formal-error method, positive finite denominator, bound-censored/unavailable behavior | unavailable pending formal-error method and sign convention | formal standardization only |
| alias `fit_sig2noise` | exact alias of preceding role | same record and identity | unavailable | not statistical S/N |

Neither diagnostic is statistical significance, detection probability,
false-alarm statistic, completeness, or purity. Zero or non-finite denominator
never yields an infinite claim; it yields the exact unavailable or failed state
defined by the future method record.

`peak_over_full_map_rms` is not an admitted alias. It may be introduced only
if the approved compatibility method establishes that fitted amplitude has the
relevant positive-peak meaning for the exact source model and parent route.
