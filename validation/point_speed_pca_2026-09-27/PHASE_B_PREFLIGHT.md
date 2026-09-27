# Phase B method and execution preflight

Phase A completed at e28f6630fffaba77f78fc2c2cb883c7345ad92f4:14 trajectories,
78 feedback calls,14 independent original parents and14 endpoints. Every strict
regression passed. Its measured timing/stage table was committed at
b7746e3ca6f1615039a2b9d4a3aabe2df62a2a7f before either method implementation.
The primary bottleneck is not proven to be PCA: rank10 feedback native wall79.55s
included8.75s PTC fits and46.49s unattributed native adapter overhead. Instrumented
normal I/O and existing worker/thread policy remain unchanged.

## Actual reference and preserved fallback

PtcPrepared subtracts each detector's arithmetic mean over binary eligible
entries within its original network/physical segment. It stores zero sentinels
outside eligibility, never measurements. For centered z the current objective is
sum_(t,d eligible) [z_td - a_t dot b_d]^2. Alternating least squares updates
loadings and coefficients using the existing mask-pattern grouped normal solves;
the basis is orthonormalized after each loading update. Two sufficiently small
relative decreases (or the existing tiny-loss condition) stop the solve, with
Jmax100, relative objective1e-5 and rank tolerance1e-10 unchanged. Per-residual
preparation, application and diagnostic r semantics remain unchanged.

Cold initialization and the existing selectable pairwise_covariance method use
joint centered products divided by (pair count−1), require count>1, positive
selected modes above rank tolerance and a resolved cutoff. They remain intact.
This is already an alternating observed-entry method, not a newly introduced EM
algorithm. Literature context is Bailey2012 (https://arxiv.org/abs/1208.4122),
Delchambre2014 (https://arxiv.org/abs/1412.4533), and Halko et al2011
(https://arxiv.org/abs/0909.4061); the explicit owner formula, not package defaults,
defines this experiment. No randomized/truncated third solver is implemented.

## B1 warm observed-entry ALS

M0 uses the ordinary cold solver. Every positive feedback cycle seeds only the
basis from the immediately previous converged state; current residual means,
coefficients, objective and all ALS updates/stopping are recomputed. No previous
means or basis-dependent coefficient factors are reused. PtcWarmStart validates
method, requested rank, exact solver tolerances/limit, shape, orthonormality,
finite basis, exact binary eligibility and the full scope string. Scope contains
prospective trajectory identity, signal branch, network/segment, bound source
PTC receipt (including detector ordering/slots/CAL identity), and eligibility hash.
The native adapter authenticates the predecessor receipt/cycle/branch and basis
file hash. First REF feedback uses only its own raw-derived cold M0; injected M0
is cold on its own perturbed parent. No cross-run/branch/network/segment seed.
Incompatibility fails explicitly; there is no structural or scientific cold retry.

## B2 masked-covariance-count-v1

A separate native PtcMethod computes C_ij=sum(a_ti a_tj z_ti z_tj)/sum(a_ti a_tj),
with every joint overlap from the unchanged binary learning domain. It requires
at least max(rank+1,2) pairs for every matrix entry in the full admitted detector
population, including diagonal entries. Excluded nonfinite values never enter
multiplication. No complete-row restriction, detector deletion, imputation,
whitening, normalization, smoothing, clipping, shrinkage or weighting conversion.
Native symmetric Eigen decomposition selects the largest positive algebraic
modes. Signed full spectrum, ordered-pair overlap histogram, negative/positive
mass, most-negative/largest-positive ratio and relative eigen residual are saved.
Residual threshold is100*machine_epsilon*detector_count; the matrix is never
repaired. Fewer than rank positive eigenvalues, unsupported pairs, nonfinite values
or failed decomposition/residual is unavailable. Existing validity-aware coefficient
application and its full-rank/support guards remain in charge. This cleaner is
approximate and is not asserted to minimize the observed-entry SSE.

## Execution and gates

Exactly three serial original-input REF attempts per valid method at rank10;
the predetermined first REF supplies science/own noise/shared immutable parent
for its three signed/broad controls. An invalid first REF blocks remaining repeats
and injections; no repeat rescues a scientific failure. Each map is checked on its
full support/normalization/WCS before quality comparison. Original x/r, RTC/CAL,
AST/VAL, MAP/BASE and geometry stay fixed. Checkpoint endpoints N1/3/5 run offline
after timing, never become feedback, and count in the32 planned endpoint budget.
Injection response/ideal measurements at1/3/5/10 use existing owners; N0 response
is retained too. No new native reference injection campaign.

Focused native tests cover complete PCA equivalence, incomplete rows, masked NaN,
indefinite spectrum, insufficient pair support, insufficient positive rank, current
warm solving and invalid scope/mask/tolerance/nonconverged seeds. Python tests cover
short termination, failure readiness, attempts, blocked dependents and nonzero
failure propagation. Matching clean local binary and independent exact-SHA review
are required before Phase B. Representative Unity/Spack remains unperformed.

Phase C is not selected yet. At most one numerically valid support-preserving
candidate may extend to rank15 after an end-to-end speed/quality assessment;
inconsequential gains or inferior/unsupported candidates do not force a winner.

Local supplemental gates:60 focused Python tests;37 focused native CTest cases;207 baseline tests plus137 subtests; full configuration preflight pass. CLI build passed. An initially mistyped optional test target failed after the CLI succeeded; corrected target citlali_ptc_numerics_test built and its tests passed. This consumed no science attempts.
