# SCI-CAL-001 AM 12.2 frequency-resolution diagnostic

The SHA-256-bound native AM 12.2 build was evaluated on the preregistered 140--280 GHz grids. This is a numerical-resolution diagnostic, not an operator or domain authorization.

| Grid step | Maximum correction difference from 1 MHz |
| ---: | ---: |
| 10 MHz | 0.000340% |
| 5 MHz | 0.000360% |
| 2 MHz | 0.000360% |
| 1 MHz | 0.000000% |

The 10-MHz 140--280 GHz center values do not exactly match the copied 0--500 GHz AM 12.2 products; the maximum range-change correction difference is 0.000340%. The preregistered 0.1% bounded resolution diagnostic passes.

All 16 raw combined outputs and deterministic execution sidecars remain in the external cache. Their exact and normalized SHA-256 values, AM identity, argv, return status, and all warning classes are retained in the manifest and metrics table. Cache/unknown warnings and error lines are zero; the unresolved-line summary remains explicit.

A pass does not suppress AM's unresolved-line warning or convert an exit-status-1 run into a clean software-success claim. The copied 10-MHz grid remains immutable lineage evidence.

This result does not recover the registered legacy q95 artifact, authorize a successor model family, or establish 5--10% absolute flux accuracy or approximately 5% repeatability.
