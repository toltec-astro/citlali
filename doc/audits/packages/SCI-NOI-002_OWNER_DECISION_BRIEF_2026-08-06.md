# SCI-NOI-002 owner decision brief — 2026-08-06

Status: coordinator review record only. Exact-d501 audit
`4f1fec36f7802f3b5e8ac067377679946930983c` is `amend` with all eight P1
findings open, production `existing_use_only`, no count/default selection, and
no evidence authorization. Owner decisions D001/F001 and D002/F002 are
approved as recorded below. Owner direction D003/F003 is a qualified
package-level provenance contract, not acceptance of the audit's broad
per-product premise. The remaining questions do not authorize repair, evidence,
recipient dispatch, or production use.

## Owner-approved first decision: F001 / D001 — what does a realization stack estimate?

**Issue and current code.** Exact d501 calls centered pixel scatter with divisor
`R` “noise variance,” without naming the target random law, completed design,
support domain, or dependence correction; it degenerates at `R=1` and is low
under an iid common-law interpretation.

**Consequence.** The existing quantity can describe this finite,
source-imprinted stack, but it cannot by itself support physical-noise
covariance, inverse-variance, significance, or count-adequacy claims.

**Engineering versus science.** Engineering can record completed identities,
support, centering, and divisors. The scientific choice is the target law and
the conditions under which any calibrated uncertainty claim may be made.

**Approved policy — 2026-08-06.** Retain the existing `1/R` computation only
as empirically centered conditional finite-stack scatter for the completed
`source_imprinted_current` realization stack. Persist completed `R`, joint
assignment/design identity, source-imprint mode, common support/response/
validity, and exact normalization. `R=1` may have mathematically trivial zero
descriptive scatter, but is invalid for uncertainty or S/N use.

It is not repeated-physical-noise variance/covariance, inverse variance,
precision, significance, threshold authority, aperture uncertainty, or
production noise calibration. A future physical-noise estimator is separate:
prioritize balanced interleaved scan-split null maps and fixed-consumer
projections, especially aperture/background and source-amplitude operators.
Delete-group or residual block resampling is admissible only with preserved
detector/common-mode correlations and validation against genuine repeats or
blank apertures. Calibration, extinction, and persistent detector-calibration
terms remain separate systematics.

**Owner question.** Answered by this approval. No physical-noise estimator,
evidence request, default, count, Unity action, or repair is authorized.

**May defer.** A numerical formula, implementation SHA, count, and evidence
plan may wait until the target is chosen. This governs F002--F007 and all
consumer handoffs.

## Coupled scale and consumer decisions: F002, F004, F005

### F002 — global scale of the MAP pattern

**Issue and current code.** A reciprocal median of MAP nonprecision coefficient
times the same centered stack scatter globally rescales that coefficient and
silently uses scale one when the calibration region is empty.

**Consequence.** It is at most an in-sample engineering scale diagnostic, not
marginal precision or covariance calibration.

**Engineering versus science.** Engineering can make empty/invalid calibration
states explicit. The scientific choice is whether a bounded global-scale target
exists at all, and what data/validity region defines it.

**Approved policy — 2026-08-06.** Retain
`alpha = 1 / median_D(q_p V_p)` and `alpha q_p` only as a
`global_nonprecision_scale_diagnostic`. Never label or infer it as inverse
variance, precision, physical-noise weight, covariance correction,
significance calibration, or proof that `q_p` has the correct spatial form.
Persist original `q_p` identity, calibration-region definition and valid-pixel
count, `m`, `alpha`, realization/design/source-imprint identity, overlap/
same-stack fact, and explicit validity. An empty, nonfinite, or invalid region
is unavailable and must not silently substitute `alpha=1` as successful
calibration.

Do not expand this use or make it a new default. Preserve current live
application/overwrite behavior only as explicit `existing_use_only`
compatibility pending FLT and MODE consumer review; this approval does not
authorize removal or expansion. Long term, keep the scaled diagnostic separate
from the authoritative mapmaking coefficient unless independent validation
establishes proportionality to a named physical inverse-variance target.

**Owner question.** Answered by this approval. No implementation, evidence,
Unity action, default, count, or production change follows.

**May defer.** Any spatial covariance model, precision relabeling, and repair
are not selected here. Interacts with MAP’s nonprecision `weight_I` contract
and FLT-001/FLT-002.

### F004 — distinct S/N-like statistics and source claims

**Issue and current code.** Coefficient-standardized pixels, direct-stack
ratios, pooled values, source-finder values, and source-table amplitude/map-RMS
ratios are different statistics but share S/N-like interpretation.

**Consequence.** No null tail, threshold, search multiplicity, completeness,
purity, or catalog-inference claim is established.

**Engineering versus science.** Engineering can give each statistic a unique
name and truthful label. Scientific policy is needed only before a claimed
tail, detection threshold, or catalog property.

**Recommendation.** Require distinct descriptive labels now; keep significance
and threshold claims fail-closed pending a later named consumer contract.

**Owner question.** Which, if any, consumer needs a statistical significance or
search/catalog claim rather than a descriptive display?

**May defer.** Threshold values, source-finder repair, and any campaign. This
routes SCI-SRC-001 and conditions FLT-002.

### F005 — filtered/aperture uncertainty

**Issue and current code.** Filtered `point_source_uncertainty` is pixelwise
stack scatter, without filter-response-corrected aperture/template projection
or off-diagonal covariance.

**Consequence.** It cannot support point-source or aperture uncertainty.

**Engineering versus science.** Engineering can trace exact response/support.
The scientific choice is which approved compact method—if any—must support a
named aperture/template use.

**Recommendation.** Continue to prohibit aperture and point-source uncertainty
claims; use only a separately justified projection or blank-aperture method.

**Owner question.** Which concrete filtered aperture/template use, if any,
needs an uncertainty claim, and is a response-bound blank-aperture method
acceptable as its intended route?

**May defer.** Full covariance, flux products, and implementation. This is
coupled to FLT-001 D002/D003 and FLT-002 response work.

## Adaptive and count-policy decisions: F006 and F007

### F006 — FRUIT feedback

**Issue and current code.** FRUIT uses median realization RMS in an adaptive
gate while the standard science text enables that gate but disables its stated
empirical-product prerequisite.

**Consequence.** Fixed stack moments do not validate repeated selection,
addback, stopping, or subsequent noise estimation.

**Engineering versus science.** Engineering can expose a consistent
configuration prerequisite. The scientific policy is the complete adaptive or
independent split/cross-fit procedure and its failure law.

**Recommendation.** Keep this feedback authority unavailable; route the full
choice to SCI-FRUIT-001 with the held post-core handoff.

**Owner question.** Is statistical FRUIT feedback a required scientific use,
and if so should SCI-FRUIT-001 define a full adaptive procedure or an
independent/split procedure before any enabled claim?

**May defer.** A repair, tolerance, source-model decision, and evidence study.

### F007 — use-specific count adequacy

**Issue and current code.** Science and Pointing have active configured counts;
OOF and standard Beammap are disabled/effective zero. None establishes a
use-specific adequacy or default. `64` is optional validation capacity only.

**Consequence.** A configured count is not an uncertainty, resource, or
production guarantee.

**Engineering versus science.** Engineering can preserve disabled-zero/no-work
and positive-enabled admission. The scientific/operational choice is a named
enabled use, target estimator, error/resource/failure criteria, and acceptance
policy.

**Recommendation.** Preserve current defaults and disabled lanes; select no
count until one named consumer needs it.

**Owner question.** Which enabled consumer, if any, should first receive a
use-specific adequacy decision, with what error and resource/failure criteria?

**May defer.** Any numeric count, default, Beammap expansion, or study. Standard
Beammap remains optional explicit opt-in only, disabled/effective-zero/no-work;
its configured 10 is inert and not a requirement.

## Engineering follow-through, not new owner choices: F003 and F008

### F003 — provenance and labels

**Qualified owner direction — 2026-08-06.** The authoritative provenance unit
is the reduction package/bundle. Put complete semantic provenance once in an
authoritative compact package manifest/sidecar, rather than redundantly in
every FITS/HDU/file. Each individual product needs only a stable package/
provenance join, product identity/version and applicable digest/scope join,
plus the minimum validity/restriction label needed to prevent misuse if
detached. Audit and validation must verify package integrity and joins.

A detached product is unverified/out-of-contract; that does not prove that all
metadata must be duplicated. Record requested/effective/completed realization
count at package scope. Require per-pixel realization counts only when support
actually varies and a consumer needs them. Do not add dense covariance, sign
streams, per-sample IDs, or duplicated metadata.

This is an engineering/provenance policy, not acceptance of the audit’s initial
claim that every individual product is intrinsically under-specified. F003
remains open pending implementation, package-integrity/join validation, and
re-audit. It authorizes no implementation, evidence, defaults/count changes,
Unity, repair, or production action.

### F008 — proportional evidence

Exact-d501 structural tests do not establish estimator, calibration,
dependence, tail, feedback, or aperture validity. Recommendation: hold all
evidence; later admit only exact-successor fixtures or studies tied to a
concrete unresolved closure question with predeclared acceptance and
FRAMEWORK-NUM-001 approval when costly. No owner choice or request is needed
now.

## Held recipient interfaces

The six audit proposals are held only: FLT-001 and MODE-001 for future
amendment/re-audit; FLT-002, SRC-001, and FRUIT-001 quarantined until their
independent cores; and BEAM-001 inactive unless the optional capability is
explicitly enabled. They do not dispatch recipients or decide their policies.
