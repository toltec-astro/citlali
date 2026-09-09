# Exact proposal for the training-only weight diagnostic

Q02 r0.5, 2026-09-09. **Not executed.** The [review and recovery](README.md)
govern this protocol. All choices here are proposed together for the single
bounded decision `SCI-FRUIT-Q02-WEIGHT-PRECISION-SCREEN-R0.5`.

## 1. Question, target and limits

Compare three training windows using one centered inverse-scatter statistic.
Estimate the fractional standard deviation of the jointly normalized relative
coefficients under the explicit approximation in section 4. Compare that
estimate with the approved provisional 0.20 goal and with changing scatter.

The estimand is repeatability of the candidate coefficients on the fixed
diagnostic occurrence population, conditional on delivered PTC/calibration
state and retained geometry. This is an exploratory precision assessment.
It does not measure true detector noise, optimal map weights, upstream learning
uncertainty, or the variability of a whole reduction. Training and evaluation
share PTC learning; disjoint rows are not independent data realizations.

## 2. Exact empirical input permission

The proposed permission covers only the following immutable discovery exports.
The common root is `/Users/gwilson/work_toltec/local_data`.

| Observation | File relative to that root | Bytes | SHA-256 |
| --- | --- | ---: | --- |
| 123424 | `beammaps/pointings/reduced/redu01/123424/raw/toltec_commissioning_pointing_123424_ptc_timestream.nc` | 760674540 | `b53ad776173b1bd2b12569c87ae15ad78a3db1c76ef4515c855255b0e4f20ba9` |
| 152389 | `fruit-development/point-152389/refactor/reduced/redu00/152389/raw/toltec_commissioning_pointing_152389_ptc_timestream.nc` | 138244964 | `894fff53babfb0070e7d0a8b831aa88098d87f9772d4a8ecb253aef009d35643` |

Verify both identities before accessing signal values. A mismatch stops access;
do not search for substitutes. Input hashes above are recovered from the
existing preflight, not newly measured in this paper revision. Read the named
files only; do not discover adjacent observations or open 129081.

Admit the exact legacy export adapter evidenced in
[INPUT_PREFLIGHT.md](../r0.4/INPUT_PREFLIGHT.md) and its
[checker](../r0.4/check_discovery_inputs.py), for this diagnostic only:

- Preserve detector column identity as (observation, column), annotated by
  embedded UID, network and array. Do not merge inactive duplicate UIDs.
- Interpret `signal` in its declared mJy/beam units, retaining the float32
  rounding of 152389 and float64 values of 123424. Calculate statistics in
  float64 without recalibration. No physical flux or common-beam claim follows.
- Use per-occurrence `flags == 0`. NetCDF masked/fill values and nonfinite
  signal or coordinates are unavailable occurrences, in both training and the
  prospective evaluation census. Preserve their counts and reasons.
- Use embedded `apt_array` values 0, 1, 2 for a1100, a1400, a2000 separately.
  Required finite UID/network identities must be unique among participating
  columns as in the preflight. An unexpected identity/flag/schema discrepancy
  stops the diagnostic; do not create a new flag or detector exclusion rule.
- Use the evidenced altaz relation below for both files, checking 123424
  against its stored `det_lon`/`det_lat`. Use radians internally. Require
  identical finite patterns and maximum coordinate disagreement at most
  1e-15 rad, as in the preflight. This is an adapter check, not a science margin.

```text
E = TelElAct
x = az_phys  + (cos(E) apt_x_t - sin(E) apt_y_t + pointing_offset_az) pi/648000
y = alt_phys + (cos(E) apt_y_t + sin(E) apt_x_t + pointing_offset_alt) pi/648000
```

Admit these recovered half-open storage edges, tied to the two hashes:

```text
0, 289, 594, 899, 1204, 1509, 1814, 2119, 2424, 2729, 3034, 3339, 3628
```

Verify the original labels and source-supported recovery against
[DISCOVERY_PREFLIGHT.json](../r0.4/DISCOVERY_PREFLIGHT.json). Preserve the
defective original labels; do not alter an export. For chunk c of length l_c,
the first floor(l_c/2) stored rows are training candidates and the remaining
rows are evaluation candidates. Use `TelTime - TelTime[0]` as elapsed seconds
under its recovered source meaning; require finite strictly increasing values.
Retain actual gaps. Its `rad` label and the acquisition `SAMPRATE` do not
define elapsed time or exposure. Report observed spans rather than n/rate.

Fix the source-guard centers at (x,y)=(0,0), with radius 28.59742792952955 arcsec
for 123424 and 27.38359774550894 arcsec for 152389. These are the earlier
metadata-derived three-beam guards, explicitly proposed here as empirical
training masks. They do not prove absence of source signal or establish an
admitted beam convention. Training requires x^2+y^2 > R^2 after unit conversion.
No fitted centroid, map, injected truth, stored weight or noise product is read
to set the mask. Do not enlarge it based on the result.

Signal access is restricted to training candidate rows for statistics.
Evaluation signal values may be inspected only for masked/nonfinite status to
complete the fixed occurrence census; do not retain their values or compute
their moments. Evaluation geometry/flags may supply counts, including counts
inside the central guard. This is admission bookkeeping, not a holdout outcome.

Both legacy bundles lack complete frozen-parent conformity and some original
executable/raw-input provenance. The proposed owner permission admits only
this empirical diagnostic on their exact delivered quantities and geometry.
It does not adopt PTC/AST/MAP successors or bypass those later gates. The
diagnostic can proceed under this limited permission without another export.

## 3. Three candidate windows and joint normalization

For K=1, 2, 4, group consecutive original chunks without overlap, starting at
chunk zero: window j contains chunks jK through (j+1)K-1. There are respectively
12, 6 and 3 windows; all twelve original chunks enter each comparison. No
observation-wide window or per-detector choice of K is included.

A coefficient group g is a detector column and K-window. Its training set J_g
is the union of eligible training occurrences in its chunks. Let e_g be the
number of eligible evaluation occurrences in those chunks, with no source
guard applied to evaluation. Every group with e_g > 0 is required. No pixel
grid, MAP support threshold or evaluation noise value defines this population.
It is a precisely declared pre-grid diagnostic population; future mapping must
bind its actual population and reassess normalization if membership changes.

```text
n_g       = |J_g|
mu_g      = sum(b_i over J_g) / n_g
v_g       = sum((b_i - mu_g)^2 over J_g) / n_g
w_g       = 1 / v_g
E_a       = sum(e_g over all required groups in array a and observation)
wbar_a    = sum(e_g w_g over those groups) / E_a
gamma_g   = w_g / wbar_a
```

Normalize once over all evaluation occurrences of the observation and array,
separately for each K. Do not normalize separately by window or across arrays.
The diagnostic is causal for a later offline map action: all training must
exist before that action. It is not an online estimator available at the
acquisition time of each evaluation sample. Values are never applied here.

Use the n denominator exactly, without a variance debiasing correction.
At least two distinct eligible occurrences and finite strictly positive v_g
are necessary to calculate w_g; this is arithmetic availability, not a new
precision cutoff replacing 64. Near-zero positive scatter is retained with its
value, not capped. If any required w_g is unavailable, wbar_a and every gamma
in that observation/array/K are unavailable. Preserve raw-scatter diagnostics
where defined, but do not normalize only the survivors. Mark an empty array
population unavailable. Report n=2 and other sparse groups explicitly; zero
estimated fluctuation is not proof of zero uncertainty.

## 4. Proposed uncertainty approximation

Use a first-order, time-correlation estimate on the centered-scatter statistic.
This avoids treating raw sample count, or correlation of b alone, as the
information available for estimating its scatter. Propagate the complete
normalization jointly, including correlations between detector groups.

The working premises are locally stable first and second moments within each
candidate window, finite fourth moments, and sufficiently short dependence
of the centered squared signal. Treat geometry, masks, calibration and the
delivered PTC state as fixed. Noise-dependent flag selection, residual source
signal, long memory and shared learned-state uncertainty are not corrected by
these premises. Observed correlations can include effects of shared PTC state,
but this does not simulate relearning or establish its unconditional variance.

For each required group, define its influence on inverse scatter at original
training time t. It is zero outside that group's eligible training rows:

```text
q_g(t) = 1[t in J_g] ((b_td - mu_g)^2 - v_g) / n_g
c_g(t) = -q_g(t) / v_g^2
cbar_a(t) = sum(e_g c_g(t) over all required groups in array a) / E_a
h_g(t) = (c_g(t) - gamma_g cbar_a(t)) / wbar_a
```

The last expression is the derivative of the full occurrence-mean
normalization, not a normalization of marginal error bars. It is an analytic
first-order propagation of that joint operation; no independent error is
assigned to its denominator. In particular, a common multiplicative change
in all weights cancels. All detector groups share one original time axis.

For each fixed correlation span tau in **0.125, 0.25, 0.5, 1, 2 and 4 seconds**,
define the positive-semidefinite Bartlett kernel and compute:

```text
K_tau(t,u) = max(1 - abs(T_t - T_u)/tau, 0)
Vhat_gamma_g(tau) = sum_t sum_u h_g(t) K_tau(t,u) h_g(u)
p_g(tau) = sqrt(Vhat_gamma_g(tau)) / gamma_g
```

Use actual elapsed times, not compacted training-row positions. Keep all
cross-chunk pairs within the span; do not declare different chunks independent.
No data are moved, interpolated, detrended or resampled. No Monte Carlo trials,
random seed, covariance inverse, PCA fit or new NOI estimator is needed.
The kernel tapers empirical score cross-products; it is not a claim that
physical noise has exactly this correlation function or vanishes after tau.

These are six fixed uncertainty-sensitivity calculations on each of the three
K designs. Neither tau nor K is optimized from the answer. Report all six, even
if increasing tau lowers an estimate. At spans approaching available training
duration, centering and poor information can make the estimate misleadingly
small. Report, per group and tau, the number of occupied half-open tau-second
bins anchored at T=0, their training counts, and the total training span. These
are resolution diagnostics, not effective independent-sample counts.

Compute the analogous fractional inverse-scatter estimate
sqrt(c_g' K_tau c_g)/w_g wherever w_g exists and its score is nondegenerate,
labeling it unnormalized. It
cannot substitute for missing p_g. A zero score norm makes uncertainty
`unresolved_degenerate_score`, including the symmetric two-sample case; it
does not receive p=0 as evidence of precision. Classify n_g=2 this way
regardless of floating-point roundoff in the centered scores.
If any required c_g has this
degeneracy or an unavailable uncertainty contribution, normalized uncertainty
is unresolved for the whole observation/array/K, even if the numerical gamma
values exist: the denominator's uncertainty is not known. An exactly zero h_g
with all required c_g nondegenerate may instead express joint cancellation;
record that distinction and its first-order conditioning explicitly.
Nonfinite quantities remain unavailable. A negative quadratic form beyond the roundoff tolerance in
section 7 is an implementation failure, not a negative variance.

This is a finite-data exploratory approximation, not a calibrated confidence
procedure. Report min/max p over the six settings as **sensitivity**, never a
confidence interval. There is no Monte Carlo error because no draws occur.
The finite-data sampling uncertainty and coverage of Vhat_gamma itself are
**unavailable in this screen**; bandwidth sensitivity does not estimate them.
No 68% or 95% coverage, upper bound on true precision, or certified attainment
of 20% is asserted. Establishing such coverage would need a separately reviewed
calibration or repeatability design if the result merits proceeding.

The statistical motivation is the lag-kernel/multiplier variance connection in
[Shao (2010), sections 2–4](https://publish.illinois.edu/xshao/files/2012/11/JASA-DWB.pdf).
That paper treats dependent-series inference under explicit asymptotic premises.
This masked, short, PTC-conditioned application is a manager-proposed
adaptation; the citation supplies no finite-sample guarantee or validation for
these data. It is a manager planning reference, not a newly admitted independent
author reference or frozen scientific authority.

## 5. Check changing scatter and residual dependence

Perform these descriptive checks using the same eligible training only:

1. Keep per-original-chunk n, mean and scatter, including chunks with no
   evaluation occurrences for a detector that participates elsewhere. For each
   K=2,4 group, report the exact decomposition
   v_pool = sum(n_c v_c)/n_pool + sum(n_c (mu_c-mu_pool)^2)/n_pool.
   A chunk with n_c=1 contributes its defined centered scatter zero to this
   algebra, but has unavailable inverse scatter. A chunk with n_c=0 contributes
   nothing; retain its missing-support count. Report the mean-change term as a
   fraction of v_pool and the maximum/minimum positive child scatter ratio
   only when all K child scatters have n_c>=2 and are positive. Otherwise the
   ratio is unavailable. These changes are not separated into drift versus
   sampling noise by this screen.
2. Split each original training candidate interval into its earlier
   floor(floor(l_c/2)/2) rows and its remaining training rows, before flags or
   the source guard. Report the two eligible counts and scatters and
   log(v_late/v_early) when each has at least two rows and positive scatter.
   Do not estimate from the reserved evaluation half.
3. Check correlations of both standardized centered b and its square minus one.
   Standardize each detector's training rows with that original chunk's mu_c
   and sqrt(v_c), requiring n_c>=2 and positive v_c. For each detector and lag
   bin [0,0.125), [0.125,0.25), [0.25,0.5), [0.5,1), [1,2), [2,4), [4,8)
   seconds, use distinct ordered-in-time eligible pairs. Separately report
   within-chunk and cross-chunk pairs. For each series s, report
   sum(s_t s_u)/sqrt(sum(s_t^2) sum(s_u^2)) over exactly those pairs, with pair
   counts. A zero denominator or no pairs is unavailable. These empirical
   correlations have no white-noise confidence bands; pairs are not replicates.
4. Report training counts, mean and scatter separately in the fixed radial
   bands R<r<=2R and r>2R, per coefficient group. Report their log scatter ratio
   only when both have at least two rows and positive scatter. A radial contrast
   can reflect sky, detector paths or noise and cannot prove contamination or
   its absence. It never changes the guard, population or coefficient law.

Do not use a lack of visible drift or correlation as proof of stationarity or
independence. Poor support, long-lag dependence and unstable scatter are reasons
to report unresolved precision, not to select a favorable window or detector.

## 6. Products and interpretation

Produce a short owner report with one table for each observation/array and
three K rows. Retain machine-readable group results so unavailable values and
their denominators can be inspected. Include:

- required groups, distinct detector columns and evaluation occurrences;
  training counts, masked/nonfinite/flag/source exclusions and missing groups;
- raw scatter and gamma where available; unweighted group and evaluation-count
  weighted 10th, 50th and 90th percentiles of p at every tau, and the fraction
  with an estimated p<=0.20 at each tau, always showing unavailable counts;
- that fraction both among available groups and against all required groups,
  with unknowns explicitly separate from above-target estimates; no pass
  fraction or detector veto is defined;
- evaluation occurrence share supplied by unavailable groups and by groups with
  any finite p>0.20, plus central-guard evaluation shares; where the complete
  coefficient vector exists, also show their candidate coefficient mass
  sum(e_g gamma_g)/E_a. This is occurrence influence, not noise or sky power;
- the temporal, radial and lag diagnostics, dependence-span sensitivity,
  degenerate scores and unresolved coverage; no exposure inferred from counts;
- elapsed wall time, peak aggregate memory, output bytes, input/settings/source
  hashes, environment versions and every failure/incomplete row.

For reproducible descriptive quantiles use the smallest observed value whose
cumulative normalized weight reaches the requested probability; weights are
one per group or e_g as labeled. Never describe unavailable entries as zeros.
No subgroup is selected after looking at these diagnostics. An observation
with different detector scatter may motivate a later regime proposal, but
does not retrospectively define an approved low/high-noise population.

The owner disposition is about the next measurement design. A promising
window still needs its exact T2 coefficient/population/uncertainty bindings and
the original U/N source-recovery, morphology, leakage, native/common support,
noise and cost tests. Feedback convergence and terminal-product behavior stay
at T3. A stable coefficient can be scientifically unhelpful. Independent-pointing
replication remains necessary before a policy recommendation; 129081 stays
reserved until its separate access decision. No outcome here qualifies N or U.

## 7. Implementation and bounded execution after approval

Implement an isolated analysis tool and deterministic verification only after
approval of this protocol. Use `/Users/gwilson/tolteca/bin/python`, one worker
and one numerical-library thread, at most two wall-clock hours, 4 GiB peak
aggregate process-tree memory and 2 GiB new output for the complete diagnostic.
Stream or batch detector groups without changing equations. Bound the run to
the two observations, three K values, six tau values and checks above.

Write new products under a new, nonexisting directory rooted at
`validation/fruit_q02_weight_precision_2026-09-09` in this worktree. Use a
distinct attempt directory on a repaired rerun; never overwrite an attempt or
an external reduction. Plot headlessly with Agg and task-specific writable
Matplotlib/cache directories. Do not copy full input timestreams into results.

Before data execution, verify the interval recovery, disjoint splits and gap
preservation; whole-array normalization and common-scale cancellation;
first-order derivative against deterministic finite differences; symmetry and
nonnegative kernel quadratic forms on irregular times; direct versus batched
calculations; and unavailable behavior for a missing required group, empty
population, one/two samples, constant signal and nonfinite values. Include an
explicit cross-detector example where joint normalization changes the result
from treating detector errors independently. This is arithmetic verification,
not a stochastic T1 campaign or coverage study.

Integer membership and unavailable identities must agree exactly. For finite
dimensionless comparison outputs use rtol=1e-12 and atol=1e-14; finite-difference
derivative verification uses central steps 1e-4, 1e-5, 1e-6 in dimensionless
perturbation amplitude and requires agreement at the smallest step within
rtol=1e-6, atol=1e-8. Record all three errors. For the quadratic form, a negative
value may be rounded to zero only within 1e-12 times the sum of absolute
contributions, plus 1e-14 in gamma-squared units. Record every such adjustment;
it still cannot certify zero uncertainty. Apply the analogous relative rule
in the natural squared units for raw inverse-scatter forms, with no arbitrary
dimensioned absolute tolerance. Unexpected arithmetic or output failure blocks
interpretation. Do not silently weaken tolerances.

Record the harness and settings digests before reading training signal.
Missing-scatter scientific rows remain in a completed report as unavailable;
they do not justify dropping later rows or adding cases. Input identity,
schema, time or adapter failure stops access/interpretation. Resource exhaustion
leaves an explicit incomplete report and preserved partial products.

Routine implementation defects may be repaired and rerun within the authorized
method, inputs, population, bounds and scope, retaining attempts. A change to
those scientific bindings returns to the owner. No reduction, injection,
weighted map, FRUIT action, Unity access, qualification, production change or
push is included.
