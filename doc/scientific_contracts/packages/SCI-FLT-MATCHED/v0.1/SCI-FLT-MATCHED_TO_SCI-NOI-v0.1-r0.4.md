# SCI-FLT-MATCHED_TO_SCI-NOI v0.1/r0.4

Status: second-review compatibility-boundary repair draft; no NOI realization,
empirical covariance equivalence, or uncertainty route is claimed available

Producer of transformation and compatibility predicate: `SCI-FLT-MATCHED`

Producer of realization population and membership meaning: `SCI-NOI`

For a signal realization `y` and candidate NOI member `z`, FLT evaluates
`K_NOI(z,y)` before Apply using only declared input facts: parent class and
quantity; WCS/grid/shape/index/order/pixel-center convention; units and
calibration convention; template identity; resolved state generation;
`D_loc`, final-support and missingness semantics; NOI population and membership
authority; numerical profile; and failure policy. Compatibility never depends
on whether the transformed result later succeeds.

For an admitted member FLT supplies the immutable resolved state and exact
fixed Apply action: template, output anchors, `W_p`, `E_p`, `d_p`, `L`,
`S_apply`, validity, numerical-profile identity, failure policy, and generation.
Every admitted member receives exactly that action; no Learn or Resolve step
runs on a member. Missing required payload makes the affected member or anchor
incompatible or unavailable; it never adapts support or normalization.

NOI owns member generation, ensemble population, dependence, finite-sample
uncertainty meaning, lifecycle, and provenance. Fixed-state parity alone does
not prove that an NOI target covariance equals `C_parent`, represents physical
noise, or instantiates U1. Any covariance comparison binds the complete frozen
condition `h=(g,theta)`, fixed selector `P_C`, and one common finite numerical
domain. Failed draws are not censored or pairwise deleted. Empirical covariance
evidence separately binds its sampling model, draw dependence or ergodic
premise, finite moments, estimator, normalization, convergence criterion/mode,
uncertainty construction, and coverage.

An NOI-informed state update creates an immutable successor FLT generation.
Per-member relearning, failed `K_NOI`, mixed generation, or changed support or
normalization makes the member/anchor or NOI companion unavailable or failed
without changing the signal estimand.
