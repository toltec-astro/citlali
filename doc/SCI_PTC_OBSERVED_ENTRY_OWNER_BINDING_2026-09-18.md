# SCI-PTC observed-entry realization — owner direction, 2026-09-18

The owner explicitly resolves the pending missing-data estimator choice:
observed-entry fixed-rank least squares is the ordinary development default;
flag-aware pairwise covariance is an explicitly selectable comparison. This
additive realization binds SCI-PTC v0.1/r0.5, its OD002 and requirements
008/013/014/019/026/029, and the existing SCI-VAL PTC named-use profiles to the
actual successor CAL product and its exact VAL generation. It does not edit
frozen source or infer scientific policy from legacy behavior.

`ptc-cal-observed-entry-use-2026-09-18-v1` realizes basis_fit_admission@1,
operator_application@1, output_retention@1 and
response_companion@1. The approved CAL singleton-opacity amendment is retained.
No separate fixed-template loading_fit_admission@1 operation is requested;
ALS loading updates are internal to the basis estimator and do not claim
that frozen-model profile. Inputs are actual CAL x in mJy/nominal-beam, with unchanged classification,
RTC/ALIGN/AST lineage and immutable parent. Original donor-influenced neighbors
remain eligible when CAL admits them. Direct replacement/exclusion centers
never contribute to the mean, fit or application coefficients. Unavailable
CAL is not zero-valued data. r is retained through its parent and remains inert.
The caller binds one explicit network within one array and the existing
processing-scan generation, intersected with physical native runs. No scan
context borrowing, new time grid, adaptive membership, rank clipping, refit
rejection, source mask, coefficient-family selection, MAP or FRUIT is added.

## Definite numerical realization

One arithmetic mean per detector over its finite fit-eligible segment values;
identity scaling; binary metric. Means are removed and never restored.
The minimized objective is the sum of squared centered residuals over observed
entries only. Missing entries are stored as zero arithmetic sentinels but
never enter the objective, means, normal matrices or residual decisions.

Both methods initialize with the recovered local pairwise convention:
C_ij = sum_t(mask_ti mask_tj z_ti z_tj)/(n_ij-1). Each z uses its detector mean,
not a pair-specific mean. Every pair must have at least two observations.
Insufficient overlap is an explicit unavailable fit, including for the ALS
initializer; it is not a zero covariance or complete-case fallback. Pairwise
covariance may be indefinite. The requested largest modes must be positive
above the declared relative tolerance. A selected/unselected cutoff gap at or
below that same scale is unresolved degeneracy and fails explicitly, rather
than selecting an arbitrary tied subspace. For ALS this is an explicitly
unavailable initializer, not proof the final observed-entry problem is
nonunique; alternate initialization remains outside this increment. Rotations wholly within the selected
subspace are a gauge and do not change the operator. Unselected negative eigenvalues are
not clipped into an invented covariance qualification.

ALS starts with that eigenspace, solves time coefficients on observed detector
rows, solves detector loadings on their observed times, orthonormalizes the
basis by QR, and recomputes time coefficients. Rank-deficient *learning* rows
use the declared minimum-norm pseudoinverse and still contribute their observed
entries to the objective. A deficient detector-loading update fails the fit;
no detector is silently removed. The frozen *application* always requires all
requested modes identifiable at that time. Learning and Apply rank rules are
distinct. Basis-dependent factorizations are reused across equal masks only
while the basis is unchanged, and discarded after each basis update.

Relative normal-matrix eigenvalue tolerance is 1e-10. Default convergence is
two successive relative objective decreases <=1e-5 (or a squared-residual
floor of 1e-24 times centered input energy), with at most 100 iterations.
A cap is nonconvergence. Nonfinite or materially increasing objectives,
insufficient support, lost rank and decomposition failures remain visible.
The objective history, initialization, stopping reason and iteration count
are recorded. No automatic method substitution occurs. These are engineering
numerical settings, not source-transfer or astronomical qualification.

Both learned bases use exactly the same frozen detector-right masked least
squares Apply. A supplied response on the exact CAL grid goes through that
same local linear operator without subtracting the signal mean or relearning.
Unavailable upstream complete response/covariance remains unavailable, not
zero and not evidence that a finite signal is scientifically qualified.

## Increment and verification

This is the one spine on `codex/timestream-successor-ptc-001`, worktree
`/private/tmp/citlali-timestream-successor-cal-001`, canonical base
`35b05ba91c761646dd08340906c7e2e05e8d2587`. The immutable CAL-input prerequisite
is `52288eb835d5dee674dc6e059b269b0bad7c36c8`. Owner authority includes connected
implementation, focused/regression checks, bounded cost and convergence probes,
independent exact-SHA review, and ordinary default activation. The owner
performs all pushes. The frozen pre-successor comparison remains unchanged.

Runtime Learn produces centered, exactly admitted group evidence and a fitted
basis; Consider freezes that evidence, rank, metric, eligibility and CAL/VAL
parent into a plan; Apply publishes cleaned occurrences and operation-local
causes in a new VAL generation. Numerical ALS iterations do not count as
pipeline Learn/Consider/Apply iterations. Development Learn/Consider/Apply
remains the engineering workflow as well.

Local AppleClang/Homebrew development checks are supplemental, not a new
Unity/Spack qualification. Tests cover complete-data SVD, incomplete rows,
excluded NaN/Inf donor storage, failures without fallback, exact CAL/VAL and
response binding. Twelve detectors remain a regression case; a real network
and independently generated 400-detector overlapping/staggered masks measure
scale. Rank probes independently fit each candidate on one preserved CAL
input. A tighter fit compares cleaned models/subspaces, not eigenvector signs.
