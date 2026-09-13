# Bounded repair of the terminal POINT source readout

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md) and
[reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md).
The [owner directive](OWNER_DIRECTION.txt) authorizes this one saved-map repair
and comparison, identity `SCI-FRUIT-POINT-PROFILED-READOUT-2026-09-13`.
Adopt its correction: a free Gaussian peak and a coefficient against the known
injected profile are different measurements on a distorted map. Neither the
fixed-template 11/12 count nor the remaining crossed-noise errors establishes
whole-source fidelity or an isolated FRUIT/PTC defect. Cite the
[previous result](../fruit_point_fixed_template_readout_2026-09-13/SCIENTIFIC_REPORT.md)
for its six lower-cost feasible witnesses, preserving all old results.
This is an isolated empirical readout experiment, outside contract authorship.
Starlet remains parked for POINT; no production or feedback code changes.

## Frozen scientific problem and implementation boundary

The recovered source fitter is `experiment_r02.py:fit_source` in
`fruit_point_coherent_feedback_2026-09-11`. Its model is

    A exp[-0.5 (u²/sigma_x² + v²/sigma_y²)] + b0 + bx*x/90 + by*y/90

with u = cos(theta)*(x-cx) + sin(theta)*(y-cy),
v = -sin(theta)*(x-cx) + cos(theta)*(y-cy), and sigma = FWHM/sqrt(8 ln 2).
Parameters are A, cx, cy, log(FWHM_x), log(FWHM_y), theta, b0, bx, by.
A and all plane coefficients are unrestricted, including negative A. Bounds:
cx,cy in [-80,80] arcsec; both FWHM in [4,60] arcsec; theta in [-pi,pi].
The Gaussian peak is continuous, not the maximum sampled map pixel. Major/minor
width sorting and orientation modulo pi are reporting conventions only.

The exact masks are saved central supported pixels at radius <=60 arcsec and
that same mask intersected with radius <=52, intersected with finite map values.
At least 100 pixels are required. Every residual has equal weight. The old
solver divides model-minus-data by the population standard deviation of the
map after a full-domain plane fit (floor 1e-12). This constant normalization
is preserved; reported SSE is in original map units. No robust loss, shape
prior, penalty, background extension or source-association change is allowed.

The ordinary initializer first fits a plane, then uses the existing masked
Gaussian convolution score to locate a positive maximum independently for
circular FWHM 6, 18 and 42 arcsec. Each starts at angle 0.2 radians. The old
nine-parameter SciPy TRF fit uses its analytic Jacobian, x_scale='jac',
ftol=xtol=gtol=1e-8 and max_nfev=250 per start. Only successful finite starts
compete by cost; the old records retain only the selected result and number of
successful starts. They do not retain every attempted solution, first-order
information or isolated old fit timing. Its gradient implementation is checked
on an analytic fixture here, without assuming that identifies the saved defect.

Call-site audit: the operational evaluator calls this fitter on the total map;
`reevaluate.py` adds the inner-domain probe and separate judgments. The current
nominal utility evaluator imports that repaired evaluator. The same older
source module also calls the fitter through `describe` and coherent Gaussian
feedback inference; other historical diagnostics and alignment-prior code use
it too. No original function or call site is edited. The experimental fitter
has a separate map/coordinates/mask entry point and imports no Data, cleaner,
feedback inference or production code. Selected pure functions are recovered
by name from the frozen source. This cannot describe trajectories rerun with
changed internal fits.

## Registered work and numerical choices

Use saved terminal pass 6. Primary: H and D, both seeds 20260911/20260912,
P/C, three arrays, both domains: 48 problems. Safeguards are H-shift_20260911,
N_20260911 and E_20260911, chosen by role and first existing seed, P/C and
three arrays on both domains: 36 more. Do not add or replace cases.

For each primary problem first solve A,b0,bx,by at the saved free geometry,
using the same unrestricted SVD solve and exact objective. This diagnostic is
separate from the repaired initialization and is not an operational policy.
Its first-order diagnostic uses raw SSE units; the repaired solver uses the
preserved residual normalization. No injected geometry enters either solve.

The replacement profiles those four linear coefficients by a thin SVD at every
trial geometry. Use the default NumPy least-squares rank convention:
cutoff = machine epsilon * max(design shape) * largest singular value.
No normal-equation inverse is formed. The exact profiled-residual derivative
includes both the projection of A*dg and the derivative of the fitted linear
coefficients; central differences test it at nonzero residual on a deliberately
incorrect geometry. Rank-deficient or nonfinite output is recorded as infeasible.

Keep exactly the three ordinary starts above (no fourth). Optimize only the
five geometry parameters with SciPy TRF, linear loss, dense exact subsolver,
constant characteristic scales [10 arcsec,10 arcsec,1,1,1],
ftol=xtol=1e-10, gtol=1e-8 and max_nfev=250. The constant geometry scales affect
numerical steps only, not bounds or priors. These settings are frozen before
benchmark execution. No start expansion, evaluation-budget increase, solver
family change or numerical retry after seeing results.

Among finite, in-bounds rank-four results, select minimum raw SSE; exact cost
ties use the original start order. Include evaluation-limit results in this
comparison; if the minimum is incomplete, retain its parameters diagnostically
but withhold the operational fit. The ordinary evaluator otherwise receives
only the selected fit. All attempts, statuses and costs remain recorded.
A separate numerical-completion assessment is not a new astronomical gate:
project the scaled gradient at active bounds (within 1e-8 characteristic units),
and require infinity norm / max(normalized half-SSE,1) <=1e-6 plus normalized
linear normal residual <=1e-10. Record the unit-step box gradient mapping too.
Report disagreements with optimizer success and all finite/constraint checks.

For ambiguity reporting only, call starts similar in cost if their raw SSE
is within 1e-8 + 1e-6*max(best SSE,1). Retain full start spreads too. Compare
peak spread to 5% and centroid spread to 1 arcsec; angle spread alone does not
reject a circular source. No global-optimum certificate is sought.
The witness/objective comparison tolerance is 1e-8 + 1e-10*max(|SSE1|,|SSE2|,1),
registered before execution. Witnesses test saved outputs only; they never
initialize, choose or qualify a fit.

## Evaluation, timing and preservation

Freeze all repaired results before evaluating against the known-template
witnesses or injection truth. Independently recompute models, costs and
first-order information from saved parameters, with no optimization in
verification. Apply the existing data-only evaluator unchanged: fixed positive
wavelet evidence and calibration, association within the fitted core,
positive fit, original parameter/core/support/compact-domain checks,
centroid probe <=0.25 arcsec, peak probe <=1%, peak/outer MAD >5. Shape residual
>25% stays a warning; max width >24 is still a compact-domain limitation.
The separate operational accuracy budgets remain 5% peak and 1 arcsec.

Source evidence is reconstructed from the same saved outer plane, calibration
scales, support and map; no calibration or feedback fit is rerun. Before the
benchmark, the isolated evaluator must reproduce all 42 original terminal
judgments and their numerical diagnostics. New judgments are stored alongside
old ones. Fixed-original-geometry and fixed-truth coefficients are diagnostic,
not newly available operational measurements. Retain full denominators.

Report original -> fixed-original-geometry -> repaired peak, geometry,
background, SSE and ratios; retain the truth-template coefficient separately.
Give both domains and matched/crossed H/D ratios, numerical versus usable
counts, the four inner-domain rejection cases, and each safeguard. Two noise
realizations cannot identify the source of remaining noise dependence.

Measure the full normal fitter including plane/score initialization, all starts,
actual residual evaluations, analytic derivative work and linear solves.
Separate fixed-geometry diagnostics, verification, evidence and report costs.
There is no fresh old-fitter timing run; old terminal evaluation time includes
both domains and other evaluator work and is not a fit-only benchmark.
Prior starlet trajectory cost failures remain unchanged.

Preserve prior packets, all reduction products and the two opaque archive
statuses. Hash-bind sources, inputs and exact problem identities before the
run. The finite cap is 84 normal fits, 252 starts, 63,000 normal residual
calculations, and 48 original-geometry diagnostic solves. Analytic fixtures are
implementation tests only. No PTC, feedback, new maps/noise, reserved pointing,
threshold change, OOF, production or automatic next experiment.

Return separate fitter and POINT-interpretation verdicts, including failures.
The result must improve trust in the comparison, not favor either arm.

Numerical reference: [SciPy least-squares termination and scaling](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.least_squares.html)
and [NumPy least-squares rank convention](https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html).
The installed runtime version is recorded at execution; no library is updated.
