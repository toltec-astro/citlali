# Verification arithmetic warning

The first independent verification completed all assertions but emitted NumPy
divide/overflow/invalid warnings from the scalar BLAS dot product in the
directional derivative check. Both the analytic and finite-difference values
were finite and agreed to better than 7e-11 relative error. The imported base
already documents the local Accelerate floating-status issue.

Preserve `VERIFICATION_FIRST_ATTEMPT.log` and `.json`. Replace only the
verification dot product by an elementwise multiply-and-sum with explicit
finite-value assertions. Rerun verification, without optimizer calls, input
timestream reads or cleaning. The objective, candidate, registered sources,
measurement rules, thresholds, population and numerical results are unchanged.
This is the routine verification repair allowed by AGENTS.md and the owner.
