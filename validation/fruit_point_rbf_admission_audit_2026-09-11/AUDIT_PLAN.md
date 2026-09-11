# Retained-product RBF admission audit

## Program adherence and prior-work recovery

This is the small first-step audit requested in the
[owner reassessment](OWNER_REASSESSMENT.md), under the
[program charter](../../doc/scientific_contracts/README.md). Adopt the completed
[RBF protocol](../fruit_point_rbf_feedback_2026-09-11/PROTOCOL.md), frozen
source/input identities, and retained results. Keep both rejection decisions.
This audit reads saved products only: no new parent read, reduction, learning,
feedback estimator, injection, RBF tuning, or 129081 evaluation. The original
packet is immutable. A separate starlet proposal is not execution approval.

Record this diagnostic before inspecting its scores. It cannot become a new
acceptance gate or retroactively change the frozen experiment.

For every R1/R2 case, array and pass, reconstruct the two score decisions and
cosine decision from receipts. Inspect maps for first nonzero next model, first
nonzero applied model, equality to the previous next model, and equality of
repeated outputs with zero applied model. Report one-based map/pass numbers.
The next model inferred from map 1 first affects map 2, never map 1.

For known-source first maps, use the actual paired processed response
T=M_source,1−M_null,1 as an explicitly oracle template. It includes the realized
learning/processing response; it is not the unprocessed injection. Its dependence
on the realized nuisance and nonlinear learning makes this a favorable diagnostic,
not an independent morphology estimate or proof of generic detectability.

Use three fixed template components on the existing core (r<=15 arcsec), shoulder
(15<r<=35), and tail (35<r<=60) about the declared injection translation. Subtract
the same outer-O least-squares plane from each input map, and multiply maps and
template by sqrt(Q/median(Q on S)) to approximate coverage-related variance.
This is diagnostic normalization only, not a map coefficient or detector weight.
Each component is normalized to unit pixel L2. Project both the source and its
paired null onto these signed templates; no positivity clipping.

Estimate the 3x3 covariance of those projections from the other retained null
seed, using template translations by multiples of 20 arcsec, from −80 to +80
in each coordinate. Require the entire nonzero template footprint on S and the
median footprint Q within a factor of two of the target footprint. Use all
eligible positions, with at least 16 required. These overlapping placements
are not independent trials. The reference seed is 20260912 for first-seed
sources and 20260911 for the second compact source. Never train this covariance
on a source map. Retain the eligible positions, coverage ratios and projection
vectors. This modest stationary-after-coverage approximation is not a qualified
pixel covariance or detector-noise model.

Let C be the sample covariance, C*=0.9 C+0.1 diag(C), and p the three norms of
processed template components. Compute w=C*^-1 p/(p^T C*^-1 p), fitted amplitude
w^T(f−mean_null), and empirical amplitude scale sqrt(w^T C* w). Require finite
positive variance and condition number <=1e8; otherwise mark unavailable.
The optimality claim is limited to these three template projections under C*.
This includes covariance between their noise projections; it is not a globally
optimal full-image matched filter. Record raw and shrunk covariances.

Report source score, paired-null score, their difference, and the reference
null score range in empirical amplitude-scale units. The paired amplitude
must equal one by construction. Five empirical scales is a descriptive marker,
not a calibrated significance, confidence interval, or method-admission rule.
Only two independent nuisance realizations exist: this diagnostic cannot
establish a false-alarm rate or universally settle standalone detectability.

The audit also explicitly distinguishes basis error, regularization error,
whole-model agreement, null rejection and output recovery. Low cosine alone
does not identify overfitting, and 378 passes are not 378 independent tests.
