# Exact proposal for the U/N4U mapping screen

Q02 r0.6, 2026-09-10. **Pending owner decision
SCI-FRUIT-Q02-MAPPING-SCREEN-R0.6.** The [review opening](README.md) governs.
This is an empirical mapping experiment, not a FRUIT method or a successor
scientific contract. Approval would supply the precise experimental permissions
below without claiming frozen-parent implementation conformity.

## 1. Recovery and changes requiring this decision

| Recovered source | Disposition in this proposal |
| --- | --- |
| Frozen Q01 and Q02-A/B substance | Adopt unchanged; no numerical successor/source adoption or C decision inferred. |
| Approved r0.2 U/N design | Retain paired same-population comparison, causal training, required metrics, bounded losses and later independent replication. Propose the explicit N4U successor below to its no-fallback family. |
| r0.3 T1 numerical proposal | Retain all nine stochastic laws, counts, seeds, order, units, estimands and margins. Replace the old minimum-64/no-fallback rule with section 2 and replace the affected failure fixtures. Add the 27 lower-bound tails specified in section 3 for demonstrated-failure labels. These numbers were proposals, not earlier execution approval. |
| r0.3 T2 and r0.4 pooled-training proposal | Supersede the leading per-chunk/64 and observation-wide alternatives with four-chunk N4U. Replace the proposed 1.5-fold discovery noise-contrast selection with reporting both fixed observations regardless of contrast. No claim of two certified noise regimes follows. |
| Approved and completed r0.5 precision diagnostic | Cite all windows, tails and failures; choose K=4 only as this proposed mapping candidate. Retain the exact input identities, populations, adapter, guards and arithmetic scatter. Do not modify or complete its unavailable results retrospectively. |
| Earlier T2 acceptability aspiration | Narrow this first real-data screen to observed-change evidence and fixed-state response checks. Real-data noninferiority, physical source recovery and confidence coverage remain unavailable. This claim/gate change is explicit for owner review. |
| Frozen MAP r0.7.1 | Cite its conditional containing-pixel N/Q arithmetic and exact support predicates, from the copies in ../../r0.4/inputs/authority/map. This decision would admit c=0.1 and the test populations only in this isolated harness. It would not create an ordinary production route. |

The completed diagnostic found K4 arithmetic gaps only in 152389 column 91,
window 2 (a1100), and column 4472, window 0 (a1400): each has zero training and
136 evaluation occurrences. Those are evidence, not a detector-specific rule.
The fallback applies to any group meeting its predeclared condition. Precision
estimates from the old incomplete normalization cannot be attached to N4U.

## 2. One exact comparator and causal state

For T2, use K=4 consecutive original chunks, anchored at zero, with windows
0–3, 4–7, 8–11. A group g is (observation, detector column, window). Retain
all three arrays separately and all twelve chunks. Training J_g is the union
of the same eligible, source-excluded first halves as r0.5; evaluation I0 is
the same eligible second-half population with no source exclusion. All accepted
coordinates have containing pixels in the geometry below, so no new grid cut
changes the diagnostic population. Let e_g count that group's I0 occurrences.
Only e_g>0 groups are required; empty arrays are unavailable.

U assigns gamma_g=1. For the N4U comparator, compute in float64:

```text
n_g  = |J_g|
mu_g = sum(b_i over J_g) / n_g
v_g  = sum((b_i - mu_g)^2 over J_g) / n_g
w_g  = 1/v_g
H    = required groups with n_g >= 2, finite v_g > 0, finite w_g > 0
L    = other required groups
wbar_H = sum(e_g w_g over H) / sum(e_g over H)
gamma_g = w_g/wbar_H   for g in H
gamma_g = 1            for g in L
```

H and its normalization include all windows of one observation/array, not
one window at a time. All groups remain in the map. The mean gamma on H is
one, and adding e_g copies of one on L leaves the full I0 mean one. Equivalently,
missing raw weights are explicitly assigned wbar_H before full normalization.
This is a fallback to uniform relative influence, not an estimate of a missing
physical variance. Do not normalize the surviving groups and then discard L.
If H is empty, N4U is unavailable for that array; U may be retained with the
paired result unavailable. If normalization or a positive normalized value
cannot be represented finitely, stop as arithmetic unavailability; do not
silently broaden the fallback to hide an overflow or underflow.

Missing/fill/nonfinite training occurrences follow the input rule, then n_g
is evaluated. Exactly two samples may define a coefficient; record sparse
counts. Zero/nonfinite scatter, too little training, or an unrepresentable raw reciprocal
receives the explicit fallback reason. There is no shrinkage, clipping, maximum
weight, 64-sample minimum, 20% cutoff, precision-based veto, learned guard,
calibration correction, covariance inverse or detector removal. A strictly
positive near-zero scatter remains influential if representable. Its danger
is measured through concentration and map behavior, not repaired after seeing
results. Invalid input identity/schema/coordinates do not become fallback cases.

Record the full generation before consuming evaluation signal values for map
statistics: training/normalization membership, n, mu, v, raw w, H/L reasons,
e_g, gamma and source hashes. Geometry, flags and finite-value predicates may
be read first to define I0. All training already exists before the offline map
action, although it need not precede each evaluation occurrence's acquisition.
Neither an evaluation noise value, map outcome, injected truth nor stored PTC
weight chooses gamma. Shared PTC learning makes split rows dependent. Fix the
generation for all T2 full/window/response maps; do not re-estimate on evaluation
halves or give a window map a new normalization.

T1 instantiates the same law with one group per detector and its declared
256-sample training. Its signal/null branches estimate their own generation
before their respective map actions, preserving W05's contamination test.
There is no temporal-window claim for that synthetic single-segment case.

## 3. Exact T1 campaign

Incorporate the **T1-only** sections of
[FIRST_SCREEN_BINDINGS.md r0.3](../r0.3/FIRST_SCREEN_BINDINGS.md): 9x9 grid,
16 detectors, visit order, training positions, source formulas, all nine rows
W00 through W05, independent trials, seed/stream/draw identities, support and
estimands. Incorporate r0.4's
[width, tolerance and 215-bound clarifications](../r0.4/EXECUTION_PROPOSAL.md).
Section 2 above replaces only minimum-64/no-fallback behavior, including its
contradictory old deterministic failure sentence. Do not inherit the old T2
pooling or execution split. No stochastic case, seed or source amplitude changes.

There are 9 x 1,024 x 2 input states x 2 policies = **36,864 stochastic maps**.
Retain the original 215 scalar bounds and tail allocation 0.05/256: 207 case
bounds and eight analytic-check tails. Add 27 specified tails for demonstrated
failure, making **242 scalar bounds** in this successor. Continuous Student-t
bounds remain Monte Carlo approximations, not exact finite-sample guarantees.
Keep 5% excess RMS (squared factor 1.1025), each arm's 2% amplitude and width,
0.02w centroid, 0.01 gross-excursion difference, full required support and the
analytic U checks. No unused tail allocation may be reassigned after results.

The exact bound count is 26 per case times nine, plus eight analytic-check tails:

- Two tails for the paired noise contrast M_U-1.1025 M_N (one new lower tail).
- Four amplitude tails: both signs of each arm's mean amplitude bias.
- Eight centroid tails: both signs, two axes, two arms.
- Eight width tails: both signs, two axes, two arms.
- Four exact binomial tails: lower and upper excursion probability in each arm
  (two new tails). Upper difference is upper(U)-lower(N); lower difference is
  lower(U)-upper(N).

A pass requires the upper noise/difference bounds and both signs of each
absolute-bias bound to satisfy their margins, plus direct support and cost
gates. For a demonstrated failure, the lower noise/difference bound must exceed
its margin, or an absolute-bias interval must lie wholly outside the allowed
range. The two existing opposite-sign tails already supply that interval; do
not count duplicate tails. An upper bound exceeding a margin alone is
inconclusive, not demonstrated loss. Missing required metrics give an
inconclusive/unavailable case, never a reduced complete-case trial sample.
Direct required support loss fails the support criterion and leaves dependent
science metrics unavailable. Retain all rows after a legitimate scientific
failure. The eight analytic-check tails still test whether the independent U
benchmark lies within its prescribed interval; their failure blocks scientific
interpretation of the harness.

Also report source-response residual power, signed response, signal-map error,
null map means and x/y adjacent-pixel correlations. They keep leakage/error
visible but acquire no extra decision bound. A similarly biased U/N pair fails
the absolute source criteria. No T1 result establishes a physical beam, NOI
family, actual PTC transfer or real-data noise regime. A synthetic heterogeneity
loss can be real even when the corresponding TolTEC regime is unestablished.

## 4. Exact T2 empirical permission, geometry and map count

Use only the two discovery files, exact hashes/bytes and adapter in
[r0.5 protocol section 2](../r0.5/WEIGHT_PRECISION_PROTOCOL.md). That section's
restriction on evaluation values is replaced, for this campaign only, by
permission to compute the maps and metrics specified here. Reverify identities
before scientific access; mismatch stops with no substitute. Admit the same
legacy mJy/beam quantity, float32/float64 storage, embedded column/UID/network/
array identities, flag-zero and masked/nonfinite predicates, recovered 3,628
row edges and altaz relation. Preserve its 1e-15 rad adapter tolerance and
actual elapsed-time interpretation. These are delivered PTC outputs with fixed
learned state, not newly certified calibrated sky samples. Original raw/build
provenance gaps remain disclosed. No APT noise column, stored weight, alternate
map, new observation, upstream relearning or reduction is used.

Work in the adapter's x,y tangent coordinates converted to arcseconds. Define
h=2 arcsec, integer indices j=floor(x/h+1/2), k=floor(y/h+1/2), centers (hj,hk),
and half-open cells [hj-h/2,hj+h/2) x [hk-h/2,hk+h/2). This is a declared legacy
experimental plane, not an assertion of frozen AST projection conformity.
Each observation/array grid is the smallest integer rectangle containing all
its I0 indices **and** every pixel center with radius <=3R. The latter are
required even if unvisited. Keep all I0 occurrences, including those outside
3R. More than 2,000,000 pixels in any grid stops the campaign; do not crop.

R is exactly 28.59742792952955 arcsec for 123424 and 27.38359774550894 arcsec
for 152389, centered on (0,0). Define fixed pixel-center regions:

```text
C:  r <= R              central/source comparison
O1: R < r <= 2R         inner background comparison
O2: 2R < r <= 3R        outer background comparison
O = O1 union O2
D = C union O          full intended comparison region
w = R/3               template width for this diagnostic only
```

These choices precede weighted outcomes. Neither source-free truth in O nor
this w as an observed beam is asserted. No fitted centroid recenters a region.
For each map accumulate numerator sum(gamma_i b_i), Q=sum(gamma_i), occurrence
counts and sum(gamma_i^2). On its complete grid sort finite Q>0, use zero-based
index floor((floor(0.75N)+N)/2) and Qstar at that index (zero for an empty set).
The exact test-only policy is c=0.1, with S_norm: Q>0 finite and Q>=0.01 Qstar;
S_sci: Q>0 finite and Q>=0.1 Qstar. Finite numerator and quotient are also
required. Store normalization-support values separately; scientific metrics
consume S_sci only. Recompute each arm/map's native thresholds on its full
grid; never substitute the intersection as native support. Empty rows remain
unavailable, not zero. Q is not a measured inverse map variance.

For each of six observation/array cases, produce two full I0 maps, and two maps
for each of the three K4 evaluation windows: **48 native maps**. The window maps
use the same grid and generation, and their own native Q/support. They are
three ordered pieces of one observation, not independent statistical trials.

On each full I0 map only, check one pixel-defined source template:
T_p=exp(-4 ln(2) r_p^2/w^2) for p in C, zero elsewhere. Add +T_p and -T_p in
mJy/beam to each occurrence at its containing pixel, in memory, after PTC.
Use alpha=1 mJy/beam, fixed flags/geometry and fixed gamma; training is not
perturbed. These **24 additional maps** give **72 T2 maps** in total. Save
Delta_plus=map(b+alpha T)-map(b) and Delta_minus=map(b-alpha T)-map(b).
Expected fixed-state response is +/-alpha T on the existing supported rows.
This is a signed mapping-response and support check; its unit response follows
from the pixel-defined input and N/Q arithmetic. It is not evidence for recovery
of a physical source, subpixel morphology, full coefficient-estimation response,
PTC relearning or FRUIT. If perturbed arithmetic overflows, retain failure;
do not reclassify the original finite occurrence or rerun with a smaller alpha.
No physical injection/population permission in CAL/PTC is inferred.

## 5. T2 measurements, alerts and uncertainty limits

The statistical objects are the six fixed observation/array cases. They are
not six independent replicates: arrays share atmosphere/observation state, and
windows share reduction and coefficient estimation. No resampling or synthetic
real-noise replacement is included. There are **no real-data confidence bounds**
in this campaign. Save point estimates with support and provenance, not errors
from dividing a spatial standard deviation by sqrt(pixel count).

For every full and window map, compute each metric on its complete prescribed
region only when all that region's pixels are scientifically available. Also
report paired metrics on the arm intersection as explicitly diagnostic, with
missing counts beside them; intersection metrics cannot satisfy a full-region
criterion. Never fill missing science pixels or drop an array. Use equal pixel
weights in these diagnostics, not Q as a precision weight.

| Measurement | Exact definition and predeclared alert on full maps |
| --- | --- |
| Background amplitude and structure | For O, O1 and O2 separately report mean b, mean b^2 and mean (b-mean b)^2. If M=mean b^2, flag M_U>1.1025 M_N as an observed uniform-loss alert. Report the ratio when M_N>0; otherwise it is unavailable, including 0/0. Smaller values are not proof of less noise. |
| Native source amplitude | Subtract each map's mean over O from its C pixels, obtaining s. Let A=dot(T,s)/dot(T,T) on C, in mJy/beam. Report both A and A_N-A_U. Alert abs(A_N/A_U-1)>0.02 when A_U>0; a nonpositive or unavailable amplitude in either arm is an unresolved source result. This is agreement with U, not truth or a physical integrated flux. |
| Native morphology | On the same signed, background-subtracted s, use sum(s r)/sum(s) for x/y centroid and sqrt(sum(s (r-centroid)^2)/sum(s)) for each width. No positivity clipping. Nonpositive sum or second moment makes that metric unavailable. Alert any centroid-axis U/N difference >0.02w or abs(width_N/width_U-1)>0.02. No agreement certifies that either morphology is correct. |
| Residual leakage | Save full U-N maps; O1/O2 means and powers above; and mean(b_p b_q)-mean(b_p)mean(b_q) over horizontally and vertically adjacent available pixel pairs within each ring. Report pair counts and covariance normalized by the two endpoint standard deviations if positive. Ring-power alerts use the same 1.1025 factor. These structures may contain atmosphere, source wings or calibration effects. No false-source probability or model-admission rate is available. |
| Signed fixed-state response | Report maximum absolute Delta_plus-alpha T and Delta_minus+alpha T over full native S_sci, inside and outside C separately. Require <=1e-9 mJy/beam and identical native support. This is an arithmetic gate, not a measured physical-recovery margin. |
| Support | Report every region's full size, native available count in each arm, common count and lost-pixel locations/causes, for S_norm and S_sci separately. Any missing S_sci pixel in D blocks a complete full-region result. U native coverage in D more than 0.01 below N is a support-loss alert. Retain both facts even if one is redundant. |
| Training and detector influence | Per group retain n, v, gamma and fallback cause. Per pixel and array report count, Q, sum(gamma^2), max detector coefficient share, and fallback count/share of count and Q. Record central C and O shares separately, plus each fallback group's contribution. Q^2/sum(gamma^2) is concentration only, not independent sample count or exposure. |
| Temporal behavior | Report the above native metrics for the three ordered window maps, within-arm differences window1-window0, window2-window0 and window2-window1 on D, plus per-pixel max(window maps)-min(window maps) for each arm wherever all three are available. Missing support remains visible. No independent-window uncertainty, stationarity pass or FRUIT convergence is inferred. |
| Cost | Measure weight estimation and U/N mapping time separately under matched resource settings; record total wall time and aggregate peak memory. Retain cost even for a scientific failure. |

The 20% precision diagnostic remains linked, with its full span sensitivity,
tails and changing scatter. Do not recompute it with a favorable span or call
fallback coefficients known precisely. For arrays where no fallback applies,
the old gamma population is unchanged, but the old approximate error still
is not map covariance. For fallback arrays the old normalized uncertainty is
unavailable for the new policy; estimating that uncertainty is outside this
campaign. Required physical source recovery, full-procedure response, real
noise covariance and their errors remain unavailable. T1 supplies known-signal
conditional evidence, not substitutes for those T2 quantities. Stability and
convergence of FRUIT, admission/revocation and terminal selection are not
applicable to mapping alone and remain required for any later T3 decision.

Report training dispersion before evaluation maps as unweighted across H groups:
q10 and q90 of sqrt(v), sorted ascending with zero-based floor(p(n-1)); also
report n and fallback counts. Retain both observations whatever their contrast.
This replaces the old unexecuted 1.5-fold stratification gate and cannot label
the pair as representative or certified high/low-noise regimes.

## 6. Verification, execution bounds and disposition

Approval would authorize an isolated validation harness, deterministic checks,
one T1+T2 campaign and report. It would not modify Citlali. Before any stochastic
or T2 signal run, hash source, settings and approval into a run-start record;
record Python/NumPy/SciPy/netCDF versions, platform and thread limits. Use the
local tolteca Python with one worker and one numerical-library thread. The whole
campaign is limited to 7,200 seconds, 4 GiB aggregate peak process-tree memory
and 2 GiB new products. Preserve earlier attempts and products. A failed bound
means incomplete, not permission to trim cases. Check budget usage throughout.

Required deterministic checks precede the campaign:

1. Equal coefficients produce equal maps; constant values and the signed pixel
   template have the prescribed response. Half-open edges and empty support
   follow exact integer rules. Verify threshold indices for N=0,1,2,3,4,8,81
   and the separate normalization/science predicates.
2. Check the fallback algebra on three groups: e=(2,3,5), valid raw w=(2,4)
   for the first two, and missing third. Then wbar_H=16/5 and
   gamma=(5/8,5/4,1), full occurrence mean one. Check invalid reasons separately
   for n=0,1, zero scatter and nonrepresentable reciprocal; H empty is
   unavailable. Check n=2 positive scatter stays admitted, and groups with
   e=0 do not enter normalization. A common raw-weight rescaling leaves gamma
   unchanged. Check accumulation of fallback signal/count/Q in its actual
   pixels; use b=(1,2,3) constant per group in this fixture, so its single-pixel
   N4U numerator is 23.75 and Q=10, map=2.375. The paired U map is 2.3.
3. Verify same occurrence exclusion in U/N4U, every retained occurrence maps
   once, all windows partition I0, no UID-based column merge, and the generation
   cannot consume evaluation values. Replacing finite evaluation values must
   leave gamma unchanged. Source/null training contamination in T1 must
   recompute each generation; T2 fixed-state injections must not.
4. Validate the metric formulas against hand-constructed amplitude, signed
   centroid/width, unavailable-moment, missing-pixel and fallback cases. Verify
   the 242-tail enumeration and interval direction, including pass, demonstrated
   failure and inconclusive examples. Arithmetic rtol=1e-12, atol=1e-14 in
   test units; discrete memberships/counts must match exactly.

T1's analytic U checks remain required. Expected scientific failures do not
stop later declared rows; unexpected arithmetic, source-identity or required
output failures stop interpretation. After approval, a routine implementation
bug may be fixed and the authorized campaign continued/rerun within the same
method, gates, inputs, population and scope, retaining every attempt. Any change
to those scientific choices returns for review.

Save all T1 maps and scalar trial outcomes, regeneration identities, estimated
coefficients, supports, analytic checks and interval calculations. Save all 72
T2 maps, fixed populations/generation, original/chunk coordinate adapter audit,
region masks, support causes, numerators/Q/counts/squared-coefficient sums,
influence/fallback maps, metrics and readable figures. Preserve logs, runtime,
peak memory and SHA-256 manifests. Process one case at a time if needed within
the fixed scope. No raw export is copied or overwritten. Rehash both discovery
inputs at closeout and verify saved products independently of the main metric
accumulator; visually inspect report figures. No build/config gate is required
for this isolated empirical harness; its required checks are those above.

At closeout, tabulate each T1 case's conditional pass/failure/inconclusive result
and every T2 alert/unavailable metric without averaging away a bad array. No
T2 row is labeled “uniform qualified” or “noise-aware qualified.” If synthetic
mechanisms and complete discovery evidence give a concrete reason to continue,
prepare a separate replication decision with the unchanged candidate and
limits and a justified evidence/uncertainty design. Do not open 129081 under
this permission. Independent-pointing replication and adequate uncertainty
must precede any pointing-policy recommendation. If this screen offers no
persuasive benefit or is unresolved, recommend closing this exploration and
returning to the simpler historical reference and available contract; another
estimator or uncertainty search is a new owner decision, not automatic work.

Q02-C and its bundled MAP handoff remain open in every outcome. No production
family, FRUIT feedback, convergence threshold, replay, qualification, Unity
operation or new scientific-author dispatch is authorized by this proposal.
