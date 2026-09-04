# SCI-FRUIT EL-F12 — Bounded intervention design r0.1

Date: `2026-09-04`

Decision candidate: `SCI-FRUIT-EL-F12-RESPONSE-AWARE-INTERVENTION-SCREEN-R0.1`

Status: **proposed development method; not implemented, approved, or run**

## Purpose and recovered evidence

This design follows the [program charter](../../../../README.md), the
[accepted input bindings](ACCEPTED_INPUT_BINDINGS_R0.1.md), and the
[separate gate sequence](EMPIRICAL_LANE_GATE_ARCHITECTURE_R0.1.md).
The [continuity audit](EL_F12_CONTINUITY_AUDIT_2026-09-04.md) records the source
recovery. This is implementation-informed empirical preparation, excluded from
the Stage B author channel. It neither starts a new contract nor edits frozen
scientific authority.

EL-F10 explained the direct JINC deletion response. EL-F11 established its
temporal persistence for one retrospectively chosen target. EL-F4 also showed
that wholesale suppression of hard penalties can repair that event while
damaging other science. Adopt those findings as bounded development evidence;
do not infer a selector, a detector judgment, or generality from them. No
additional explanatory UID 4460 study is needed for this design.

The hypothesis is that screening the map consequence of *new* exclusions can
avoid some harmful actions while retaining useful ones. High predicted response
alone cannot tell useful from harmful. That distinction is the experiment's
question, not an assumption used to classify candidates.

## Control, inputs, and fixed horizon

All arms use the historical complete-product recurrence with `alpha=1`:
reread the original observation, select the preceding complete model, subtract
it, process the residual, restore the accepted model, and carry the newly
completed product plus complete learned state. Do not reset masks, weights,
model history, D19 targets, or other learning to isolate a favorable effect.
Use the existing raw observation JINC route and complete-map penalty evidence.
Feedback-model bypass and other rejected recurrence changes stay disabled.

Use only observation 123424/sub-observation 0/scan 2 and arrays a1100, a1400,
a2000. Input bytes are those in the existing EL-F2
`INPUT_INVENTORY_R0.1.md`, `TEXT_FITREPORT_INPUT_INVENTORY_R0.3.md`, and
`FROZEN_INPUTS_R0.3.md`, bound by this proposal's manifest. Preserve the legacy
APT limitation. Reverify all external files before execution; no substitution
is allowed. The base/input/common/text-fit-report stack is the EL-F2 stack.
The only science overrides are those explicitly defined here: `alpha=1`, the
already implemented EL-F5 injection position, and the experimental action arm.
Path, output-retention, provenance, and diagnostic settings are registered too.

Start every primary trajectory fresh at absolute iteration 0. Injection is
off or 100 mJy/beam in each array from iteration 1 at FITS map-world
`(AZOFFSET, ELOFFSET)=(0,-60)` arcsec, using the existing post-RTC injection
path. Save every iteration through 6; no quality-dependent early stopping.
Iteration 6 is a hard experimental limit, not a declaration of convergence.
No alpha-1.25 checkpoint or outcome selects a start, action, or terminal map.

H below is a same-build historical-recurrence control, not the exact historical
executable or astronomical truth. The full historical-artifact and population
requirements in [Gate-D readiness](EL_G0_GATE_D_READINESS_R0.2.md) remain open.

## Causal candidate population and timing

For each branch independently, at the end of iterations `k=0,...,5`:

1. Complete the ordinary science map, diagnostics, and proposed learning
   records using only that branch's state through `k`.
2. Enumerate every newly effective, scan-local, factor-zero
   `map_pixel_outlier_detector_dominance` record from `mapdiag:` for that
   observation. Preserve the current minimum-pixel rule (four in this stack),
   source protection, and contribution identity.
   A key is `(observation, array, UID, zero-based scan)`; a duplicate proposal
   for one key is not a new opportunity.
3. Exclude from intervention any key already hard-excluded on entry to `k`,
   already assigned an experimental action, or independently excluded by
   another reason. Keep these records and explicit reasons in the census.
   Do not revive rejected/flagged data. This census concerns proposed actions;
   it does not predict the future runtime flags or assert that every proposal
   will pass its ordinary application cap in `k+1`.
4. Calculate the response below for **all** remaining keys and their disjoint
   union. Use no UID list, known bad scan, source-truth mask, control-branch
   outcome, later iteration, or final-quality target as an input.
5. Freeze the census, response summaries, selection and assigned actions in
   checkpoint state before starting `k+1`. Every decision records its source
   iteration and first application iteration. No action changes iteration `k`.

Maximum eligible population is 16 new keys per boundary; maximum newly
assigned keys is 64 over a trajectory. Exceeding either cap stops that
trajectory as unavailable. Never keep just the most favorable keys. An empty
census is a valid no-op. If no key is selected anywhere, the screen reports
`no_selected_opportunity`, not success, and the alternatives must reproduce H.

This is causal selection on an exposed development observation. It is not an
outcome-blind population study, even though future values are barred from the
decision rule. The rule must be tested for independence from UID renumbering,
input traversal order, injection metadata, and all later-result files.

## Response calculation and fixed selector

Use the signed JINC numerator `N`, denominator `C`, quadratic accumulator `Q`,
absolute sums, and occurrence/support identities from the completed iteration.
For a key `t`, form its contribution `N_t,C_t,Q_t`. On usable common support:

\[
M=N/C,\qquad M_{-t}=(N-N_t)/(C-C_t),\qquad D_t=M_{-t}-M.
\]

For a set of disjoint keys, subtract their summed accumulators once; do not
sum separately normalized deletion-response maps. Retain signed leverage,
contrast, absolute coefficient mass, quadratic support, occurrence and unique
detector counts separately. The formal map coefficient is nonlinear and must
not be substituted for additive `C` or a detector's leverage.

`D_t` is exact for deleting a contribution while holding the current processed
samples fixed. It is only a screening predictor for the next iteration's
historical pre-cleaning exclusion. It does not predict changes to shared
RTC/PTC cleaning, future weights, flags or feedback. Actual H/alternative
trajectories, not this formula, measure those consequences.

Use the existing EL-F10 finalization and forward-error construction, including
binary64 unit roundoff `2^-53`, finalization safety factor 16, identity safety
factor 64, and its denominator conditioning rule. Re-finalization must reproduce
the actual map; the signed identity must pass its forward-error bound.
Missing accumulators, mismatched identity, malformed records, or failed
accounting invalidate the screen. A mathematically ill-conditioned deletion is
a *support risk*, distinguished from a broken receipt.

Let `S` be the completed branch's finite scientific support under its unchanged
registered map policy. A target footprint consists of pixels in `S` with
positive target absolute coefficient mass. On that footprint retain both the
conditioned common deletion support and every lost/unconditioned pixel.
Deletion that loses any pixel of `S` or makes its calculation unconditioned
sets `support_risk=true`; do not discard those pixels to obtain a favorable
score. Gains outside `S` are reported separately.

For each array define a descriptive map scale, without a source mask:

\[
b=\max\{1.4826\,\operatorname{median}_{S}|M-\operatorname{median}_{S}M|,
          1\ {\rm mJy/beam}\}.
\]

For a nonempty conditioned footprint `V_t`, set
`r_t=RMS(D_t over V_t)/b`, with equal pixel-area weighting. The selector is:

- select an individual key if `support_risk` or `r_t >= 1`;
- if the combined deletion of all eligible keys in an array has support risk
  or `r_union >= 1`, select all eligible keys in that array;
- otherwise leave each unselected key's historical action unchanged.

A zero-contribution key with no support risk has score zero. A footprint with
no conditioned pixel has unavailable score and is selected only through its
explicit support-risk flag. An empty/nonfinite map support cannot supply `b`
and invalidates the screen. The scale is a robust map-scatter descriptor, not
a noise standard deviation, significance, or uncertainty estimate. The floor
and cutoff are new proposed heuristics, not derived from EL-F11. No trial or
threshold sweep is authorized, including a sweep on retained EL-F11 values.

## Three fixed action arms

| Arm | Treatment after selection at `k` | Duration |
| --- | --- | --- |
| H | Existing hard exclusion before cleaning | Historical persistence |
| Hold | Suppress only the selected new map-dominance exclusion; otherwise retain ordinary participation and map coefficients | `k+1` through 6 |
| Half | Suppress only that early exclusion, retain participation in cleaning, and multiply that key's final JINC map coefficient by 0.5 | `k+1` through 6 |

The two alternatives use the same selector, each on its own causal state.
H computes the census in audit mode and always applies the historical action.
An assignment is made once per key. Relearning that reason is logged but cannot
stack another factor or extend the horizon. Other exclusion reasons, APT flags,
sample flags, masks, weight estimators, and application caps are unchanged and
take precedence. At the ordinary application point, evaluate the unchanged cap
on the complete proposed set before experimental suppression, using the
then-available pre-action state. If that gate withholds the historical action,
withhold its experimental replacement as well and record it as not applied;
Half must not attenuate a proposal that the ordinary cap rejects. This runtime
eligibility gate cannot use a resulting map or revise the selector score.
An inability to preserve this precedence makes the trajectory unavailable,
not permission to change the cap.

Half is a multiplier on final map coefficients, not a change to the learned
scalar penalty factor or to shared cleaning weights. For each otherwise
admitted occurrence with coefficient `c` and processed value `x`, its numerator
and signed denominator contributions become `0.5*c*x` and `0.5*c`; its quadratic
contribution becomes `0.25` times the original. Propagate the same coefficient
through the kernel and any generated noise realizations. Recompute final
normalization and formal/empirical map weights by the existing finalizer;
never copy the H weight map or multiply its finalized weight by 0.5. Retain
ordinary occurrence counts for otherwise admitted samples with the positive
multiplier, including signed JINC contributions, and record that multiplier
separately. The factor does not establish physical inverse variance or a new
covariance claim. No pixel-local clipping or coefficient search is included.

Selected records, reason precedence, first application, horizon, selector
version, and coefficient factors are explicit restart state. Do not overwrite
historical records or treat these diagnostic/decision objects as sky products.
Continuation beyond 6 is unsupported for these experimental states and fails
closed; a later owner decision must define any longer-lived method.

## Implementation feasibility and compatibility gates

The current diagnostic names one target before accumulation. It cannot merely
be relabeled a non-oracle selector. A prototype must retain the ordinary final
PTC science occurrences needed to account for the full end-of-iteration
candidate census, then evaluate those occurrences against the completed map
without rerunning RTC/PTC. A bounded local spool is permitted. Exact
accumulation order and signed JINC placement must be retained. Its full I/O,
wall time, and memory cost counts against the candidate. If this cannot be
done within the stated resources, report infeasibility; do not replay once
per detector or substitute a proxy. Diagnostic noise passes cannot add
duplicate science occurrences.

After owner approval, freeze a registration before any real-data run containing
the exact source/build/dependency/binary identities; every raw, telescope, APT,
text fit-report and config hash; analyzer and metric definitions; source/array
coordinates and region masks; run order; comparison normalizations; caps; and
the complete scientific decisions in this document. New binary hashes are
implementation unassessed today, not missing historical input permission.

Require the local CLI build, all enabled CTests, baseline and FRUIT Python
tests, and full configuration preflight. Focused synthetic tests must cover
no opportunities, all keys, joint deletion, conditioning/support loss,
nonfinite/missing state, cap exhaustion, reason precedence, duplicate keys,
the half-factor quadratic/kernel/noise propagation, and exact restart. A
synthetic high-influence useful exclusion must demonstrate that influence
alone is not interpreted as a beneficial rescue.

The first two trajectories H0 have the new machinery disabled. H0's uninjected
maps must reproduce the retained EL-F2 alpha-one control at iterations 0–6
bitwise. The next two H trajectories have accounting/selection enabled but
still take historical actions. H must reproduce H0 bitwise in every ordinary
signal/kernel/weight and formal-coefficient plane. Compare scientific
checkpoints and ordered per-iteration learning records exactly after a
prospectively enumerated provenance-only normalization; compare D19 target
state as well. A whole-file cumulative CSV is not interchangeable with a
single-iteration CSV. Any new scientific difference fails the gate.

One binary serves every member of the matrix. If an authorized routine repair
changes that binary after a run, earlier runs cannot silently remain same-build
comparators. Retain them, count any replacements against the explicit ceiling,
and stop for a new owner decision if completing a consistent matrix would
exceed it. Analysis-only repairs need no Citlali replay when the registered
method and inputs are unchanged.

## Bounded execution matrix

One CPU thread, `--grppiex seq`, local execution only. Fixed primary order:

| Order | Trajectory | Iterations |
| ---: | --- | --- |
| 1–2 | H0 uninjected, H0 injected; selector disabled | 0–6 each |
| 3–4 | H uninjected, H injected; selector audited | 0–6 each |
| 5–6 | Half uninjected, Half injected | 0–6 each |
| 7–8 | Hold uninjected, Hold injected | 0–6 each |

Require both H compatibility gates before launching alternatives. Require
injection/control iteration-zero identity and pre-first-action identity for
each alternative. Every scientifically promising alternative additionally
restarts its own injected and uninjected completed iteration-3 checkpoints
and reproduces iterations 4–6 bitwise, including experimental decision state.
If no assignment exists before iteration 4, this checks later decision creation
but does not prove real-data restoration of an already active assignment;
disclose that limitation and require active-state coverage in replication.

Maximum: eight primary trajectories/56 passes, plus four conditional restart
trajectories/12 passes. One isolated diagnostic-defect replacement of at most
seven passes is allowed under the standing owner direction, with the original
attempt retained: absolute ceiling 13 trajectories/75 passes. No replacement
for unfavorable science or a failed scientific gate. No environmental retry
is implicitly added. Non-replay local parser/analysis repairs may be rerun
under the standing direction without changing this count.

Limits: 1 hour per trajectory, 12 hours aggregate including analysis and any
replacement, 16 GiB peak resident memory per process, 32 GiB temporary spool
within 64 GiB total newly retained output. These are ceilings, not expected
performance. Use a new isolated local root named
`fruit-el-f12-response-aware-intervention-r0.1` under the existing
`/Users/gwilson/work_toltec/local_data/fruit-development/` area after approval.
No directory is staged by this proposal. Preserve all old reductions and
unsuccessful new attempts; an identity, resource, nonfinite-product, required
output, or unexpected error/critical-log failure stops the affected work.

## Measurements and proposed scientific protections

At every saved iteration retain complete maps, kernels, weights, checkpoints,
candidate/decision/application ledgers, support masks, logs and resource
receipts. Form `T_k=M_injected,k-M_uninjected,k` independently for each arm.
This is **total adaptive response**, not matched-operator source transfer;
use the [response measurement frame](EL_RESPONSE_MEASUREMENT_FRAME_R0.1.md).
Kernel-normalized source metrics describe that paired response locally.

Use the existing fit definitions, 25-arcsec source search radius, 20-arcsec
aperture at `(0,-60)`, fixed 20-arcsec real-Neptune aperture at
`(12.53903,-5.334553)` arcsec, and injection-centered 40–120 arcsec annulus
excluding 25 arcsec around that fixed Neptune position. These are fixed
evaluation regions on all three arrays, not selector inputs. Freeze the
same-pixel WCS mapping before execution; retain signed maps and positive and
negative leakage separately so cancellation cannot conceal arcs.

The comparison domain per array/iteration is H's injected/uninjected common
scientific support in these predeclared geometric regions. Report metrics on
the four-map common subset, plus H/candidate support fractions and all gained,
lost, and unavailable pixels on the union. Any candidate loss of an H-domain
pixel fails protection, even if the common-subset metric improves. No
remapping, silent zeros, or post-outcome region adjustment is allowed.

The following are proposed development bounds, not qualification tolerances.
Apply non-inferiority at every iteration 1–6 and in **every array**, not just
where an intervention occurred. `H` in a bound means the same metric in H at
the same iteration. For RMS quantities use
`B(H)=max(1.10*H,H+0.1 mJy/beam)`; the fixed absolute floor prevents ratios of
nearly zero baselines from driving the screen.

| Dimension | Required record and protection |
| --- | --- |
| Source recovery | Central and whole-kernel recovered fractions versus 100 mJy/beam truth; each absolute error from unity at most H's error +0.01. At iteration 6 both fractions must lie in [0.95,1.05]. Report aperture flux too. |
| Morphology | Major/minor width divided by same-iteration processed-kernel width; absolute deviation from unity at most H's +0.01, and at most 0.03 at iteration 6. Centroid error at most H+0.02 arcsec throughout and at most 0.1 arcsec at iteration 6. |
| Leakage | Full-map and aperture residual RMS after the fitted kernel component; separately annular and real-Neptune RMS of T. Each at most B(H). Retain peaks, signed sums, residual images and radial summaries. Never interpret these as a source-only halo. |
| Useful-exclusion protection | In the uninjected branch, real-Neptune fitted amplitude and widths may change by at most 1% relative to H, centroid by at most 0.02 arcsec; annular and full-map post-source-fit residual RMS at most B(H). Undefined fits fail the screen. |
| Support | No lost H comparison-domain pixel, no missing array or required metric; report gained support, formal coefficient changes, occurrence counts and reason-specific losses separately. |
| Convergence | Full-map and annular RMS of T_k-T_(k-1), central/whole-kernel recovery changes, oscillation and drift. At 5→6, RMS changes at most B(H), and each recovery change at most max(1.10*abs(H change),0.01). Report hard-cap censoring and descriptive time-to-quality; no early-stop recommendation. |
| Runtime/memory | Wall/CPU time, peak RSS, spool and retained bytes, census sizes and accounting time for every trajectory. Report Half/Hold versus H0 (total overhead) and H (action effects with diagnostic overhead shared). No scientific loss can be exchanged for speed. |

Report the full census for selected, unselected, already excluded, and
overridden-by-other-reason records. Report no-injection regressions and any
loss of useful exclusions; do not label high response as a correct action or
low response as a benign detector. These data have no independent per-detector
truth labels. A population false-action rate is therefore unavailable. The
uninjected protections are a necessary regression check, not a substitute for
independently judged useful/benign-action cases in replication.

## Disposition, stopping, and replication

H compatibility/accounting failure makes the experiment invalid. A required
candidate metric, support, or resource failure is retained as a failed or
unavailable candidate, never dropped from the paired report. An adverse
scientific result is not rerun. Preserve all registered arms; one arm's ordinary
scientific failure does not permit replacing it with a new arm. A shared
implementation/gate defect stops dependent work until correctly dispositioned.

A candidate is **eligible for replication only** if every protection passes,
both its restart checks pass, and its a1400 annular and real-Neptune T RMS each
decrease from H by at least 20% **and** 0.1 mJy/beam at both iterations 5 and 6.
Those two leakage endpoints are prioritized because they motivated this
development branch; they do not narrow the all-array protection population.
No averaging of a failure against another improvement is allowed. Otherwise
report `not_promising`, `no_selected_opportunity`, or the typed invalid/
unavailable result. Report both candidates in full; do not choose a winner
by an unregistered composite score or pixel-level p-values.

Before a policy recommendation, a separate owner-approved replication must
bind an independent observation, exposure history, input/config hashes,
support, useful-exclusion controls, and resources. Prefer a pointing selected
by an outcome-free inventory before its candidate outcomes are inspected.
The exact observation is **unresolved**, and its runs are **not authorized**
by EL-F12. Reusing exposed 152389 could be disclosed development replication,
but never untouched qualification. Freeze any advancing method and these
protections before replication; a changed cutoff, horizon, factor, region,
profile or metric begins a new development generation and cannot inherit a
successful replication label.

Replication failure, harmful loss of useful exclusions, or inability to select
without later information ends this method candidate. Even replication success
allows only an owner review of the narrowly supported development policy;
full Gate D, broader angular-scale/flux-transfer work, held-out qualification,
Stage B and production require their existing separate decisions.
