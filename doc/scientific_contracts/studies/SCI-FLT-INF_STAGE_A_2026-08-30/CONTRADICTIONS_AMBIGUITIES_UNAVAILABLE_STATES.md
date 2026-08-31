# SCI-FLT-INF contradictions, ambiguities, and unavailable states

Record identity: `SCI-FLT-INF-GAPS v0.1/r0.11`

Status: Stage A owner-review record; absence is preserved rather than repaired

## Contradictions

### Resolved: `wiener_filter` label versus scientific estimand

The active full path publishes a normalized numerator/denominator field built
from a template, parent weights, and a shaped inverse spectral model. That is
implementation-consistent with a local template-amplitude estimator. No exact
signal prior or posterior covariance was recovered. The historical/internal
mathematics also groups “matched or Wiener” approximations together.

Owner disposition under ODQ-001: the historical path is an **optimal matched-
template amplitude estimator**, not a posterior/Wiener reconstruction. The
historical implementation label may remain where compatibility requires it,
but it is not the scientific identity. A genuine posterior reconstruction is
a separate future method and contract.

### Requested full method versus realized lowpass substitution

The full path can substitute a constant spectral field when the required PSD
is missing or invalid while retaining the requested full-method name.

Disposition: current behavior is inventory evidence, not admissible science.
Future authority must fail closed or use an explicit selector whose output
retains the alternative method identity.

### Numerical zeros versus unavailable/null state

Small denominators are zeroed and output values at those pixels become zero.
No authority establishes that the estimand is scientifically zero there.

Owner disposition under ODQ-006 and ODQ-007: a nonfinite/nonpositive
normalization or any missing/nonfinite/invalid required support input makes
the affected location null, unavailable, or failed. It never establishes a
scientific amplitude of zero. Parent-shaped numerical storage does not change
that state.

### Historical edge-fill direction versus current fixed scope

Historical SCI-FLT-001 D001 approved median fill only as a numerical device
with a scientifically eroded region. Current SCI-FLT-FIXED Stage A selects
full-footprint-only and defers fill, taper, truncation, and support
renormalization.

ODQ-007 independently adopts complete-support-only base admission and permits
fill solely as a numerical device when conservative erosion proves that no
admitted output depends on it. It does not adopt historical median fill,
thresholds, taper, or erosion mechanics. The historical decision remains
precedent only; adaptive edge/background conditioning is deferred to a
separate future method and the protected fixed package is unchanged.

### Historical empirical calibration versus frozen NOI typing

Historical FLT D003 retained one robust global empirical calibration of the
formal spatial pattern, while frozen NOI now sharply separates conditional
second moment, reciprocal scale, covariance, precision, and consumer weight.

Disposition: preserve the old decision as historical scope intent, but require
a new exact FLT/NOI derived-product boundary. No current coefficient promotion
is authorized.

### Registry commit identity versus recoverable Convolve object

The registry names commit `800e8ae433f87d3fb7521fcb1a7fdf1d32532949`,
which is unavailable in the local object database, but states SHA-256
`8d336242...` for the document. The document recovered at
`1bf77eadd7be1d12f285c272bb5f91511a3259f0` has that exact SHA-256.

Disposition: record the original commit as unavailable and the later exact
object/digest as recovered. Do not assert the two commit objects are identical
without the missing object.

## Ambiguities

### Parent coefficient/covariance identity

The full path takes a square root of the published map coefficient field and
uses it inside an inverse-spectral estimator. Frozen MAP authority does not
make that field precision by name. The exact covariance model, if any, is not
identified.

ODQ-004 disposition: a future author must develop bounded coefficient-role
options in both contract views. No role is selected, and precision/covariance
meaning remains unavailable.

### Spectral state meaning

The spectral field is shaped, clamped, radially interpolated, and normalized.
It is unclear whether it is intended as noise covariance, a relative spectral
weight, an empirical preconditioner, a transfer mask, or a heuristic.

ODQ-004 disposition: historical use of a radially symmetrized average map noise
PSD is admitted as a candidate to examine, not as a default, exact covariance,
stationarity/isotropy proof, optimality proof, or implementation prescription.
The author must produce the scientific options.

### Resolved: template meaning

Available templates include a parent kernel, analytic Gaussian/Airy profiles,
and a high-pass delta. Their normalization, parent-beam relationship, source
model, unit-source convention, and use as filter kernel versus estimator
template are not uniformly specified.

Owner disposition under ODQ-005: base v0.1 uses one exact immutable declared
template-response product per application, representing parent-map response
per unit amplitude `A`. It binds its amplitude convention/units, compatible
parent, grid/WCS/phase, support/tails, array/beam/calibration relationship,
validity, and provenance. Parent-bound point-source and explicitly supplied
scientific-template sources are admitted; Gaussian/Airy is only complete-
product construction. Target/source/NOI-learned templates and high-pass/delta
are deferred.

### Resolved at normalization level: denominator meaning

Owner disposition under ODQ-006: the denominator is the exact estimator
normalization `D(x)=<t_x,Q t_x>`. Its reciprocal is a variance only when the
ODQ-004 selection makes `Q` the authorized inverse covariance and all required
model, support, and validity assumptions hold. Otherwise `D` remains a
normalization coefficient. Nonfinite, nonpositive, or unresolved `D` makes
the sample null/unavailable/failed, never an amplitude of zero.

### Approximation adequacy

Several stopping paths and an internal denominator floor exist. No scientific
tolerance ties realized tail/update summaries to response, amplitude, or
uncertainty error.

ODQ-006 disposition: the exact reference operator is authoritative. A future
implementation-blind author must develop quantitative conformance-envelope
alternatives in both contract views, with identical option identities and
bounds for normalization, template response, support/null behavior, and
uncertainty. The scientific owner must select or reject those alternatives
before an approximate route can freeze. An iteration or tail cap is not
success unless the selected bound is met; outside-envelope operator changes
are separately versioned methods or unavailable.

### Resolved at policy level: covariance and denominator precision

No universal parent covariance, output covariance representation, or proof
that the historical denominator is precision was recovered.

Owner disposition under ODQ-009: when a matching authoritative parent
covariance exists, exact fixed-state covariance is
`C_cond=L C_parent L^T`. Missing entries and cross blocks remain unavailable,
not zero or independent. `D(x)^-1` is only a marginal conditional variance
when ODQ-004 selects exact inverse covariance and every GLS premise holds;
otherwise `D` is normalization only. Frozen-NOI second moments, calibration
uncertainty, and full-procedure uncertainty remain separate. Exact persistence
and representation alternatives are delegated to both future contract views
for later owner disposition.

### Resolved for base; deferred for adaptive edge parity

The learned window is shared with members, but the real signal uses an affine
background transformation and members are zero-centered/windowed. It is
unclear what exact transformed-member population corresponds to the published
science product.

ODQ-007 disposition: none of that learned state is admitted by base v0.1. The
base complete-support and boundary identity must be consistent between science
and admitted members, while ODQ-010 retains the exact NOI-generation choice.
Any future adaptive method must separately resolve fixed-state versus
full-procedure parity and cannot borrow the base identity.

### Resolved: kernel-response product

The parent kernel is processed under a uniform-weight full path while signal
uses spatially varying weights. It is unclear which response is intended and
whether the resulting kernel is adequate for spatially varying source
response.

Owner disposition under ODQ-008: for fixed realized state the exact response
row is `L_x u=<t_x,Q_x u_x>/<t_x,Q_x t_x>` and the declared template response
is `R_t(x,y)=L_x t_y`, with unity matching response at admitted locations. The
off-diagonal response may be position dependent, asymmetric, anisotropic, or
nonstationary. A uniformly processed kernel is not a universal response unless
translation invariance and identical weighting, complete support/validity,
centering/phase, boundary, and normalization are proved on an exact domain.
The implementation-produced kernel establishes no such proof or conformity.

ODQ-008 also makes the signal unit the declared template-amplitude unit rather
than an automatically inherited parent unit. Parent nominal beam identity is
provenance, and any matched-filter beam/solid angle must be derived from the
exact response. Parent/template calibration dependence is joint; no
independence, cancellation, or missing covariance may be inferred.

### Resolved: observation/coadd parent selection

Runtime chooses observation filtering or coadd filtering from the coadd mode.
No science establishes equivalence, selection rationale, or a composition rule.

Owner disposition under ODQ-003: both exact ordinary-MAP observation bundles
and exact ordinary-MAP coadd bundles are admitted as distinct observation-local
and coadd-local parent/grouping identities. No equivalence, commutation,
filtered-result combination, or filter-owned cross-observation operation is
approved. JINC and derived-map parents are deferred.

### Resolved: map product versus source-analysis product

Point-source-named planes and downstream Gaussian fits exist, but the boundary
between a map-domain amplitude field, response correction, selected source
amplitude, and catalog inference is not authoritative.

Owner disposition under ODQ-002: the selected package publishes a matched-
filtered map and performs no source detection, selection, peak interpretation,
deblending, fitting, or catalog construction. No SRC ownership boundary is
introduced. Any future source analysis is an independent contract that may
consume the filtered map if later authorized.

## Unavailable scientific states

| State | Reason unavailable | Consequence |
| --- | --- | --- |
| exact matched-template scientific identity and package boundary | **available at identity level**: ODQ-001 through ODQ-013 and the package-identity approval select `SCI-FLT-MATCHED`; delegated author options remain unselected | numerical product remains unavailable pending package-local authorship and owner disposition of the authored options |
| genuine Wiener/posterior method | no complete prior/likelihood/operator/posterior specification recovered | no posterior reconstruction product |
| matched-template map-filter realization | owner-selected estimator, filtered-map product role, parent roles, template-response identity, exact reference operator, complete-support rule, and exact fixed-state response/unit identity exist, but weighting/covariance, exact realized influence extent, numerical response representation, and any approximate conformance envelope remain unresolved | no authorized numerical matched-filtered map |
| parent covariance/inverse-noise | ODQ-004 delegates option development; no option or parent coefficient meaning is selected | denominator cannot yet be called Fisher information or inverse variance; numerical `C_cond` is unavailable |
| covariance representation | ODQ-009 fixes the conditional identity but delegates exact/structured/projected/lineage-resolvable/unavailable options to both future contract views | no numerical covariance product or independent-pixel consumer claim is authorized |
| exact realized template-response product | product identity/source classes, fixed state, and complete-support consequence are approved, but no numerical instance, selected approximation envelope, or realized influence extent is authorized | numerical application remains unavailable |
| approximation-qualified operator | ODQ-006 approves the exact reference and bounded-approximation policy, but no quantitative envelope option has been authored and owner-selected | approximate route unavailable; exact evaluation remains conformant in principle but blocked by the other unresolved gates |
| adaptive edge method | ODQ-007 expressly defers learned support/background/fill/taper to a separate future contract; current behavior and old policy do not define it | edge-conditioned scientific support unavailable |
| data-thresholded mode selection | inactive implementation fragment and no method contract | no route or product |
| automatic fallback | no selector authority or realized-method product identity | requested-primary output fails closed |
| source detection, selection, fitting, catalog, or source-learned filter | excluded from selected package; no independent current contract or active route recovered | deferred without present ownership assignment |
| observation/coadd equivalence | no commutation or population result | separate methods/parents only |
| JINC or derived-map parent route | excluded from v0.1 by ODQ-003 | deferred and unavailable |
| fixed-state transformed NOI | exact INF owner authority and parity absent | transformed UNC unavailable |
| per-member-relearned NOI | complete learning graph and member method absent | separate method unavailable |
| NOI-informed successor | owner learning/update rule absent | no successor generation route |
| empirical coefficient promotion | frozen NOI boundary not satisfied | no precision/inverse-variance/consumer-weight claim |
| standardized significance | frozen NOI permits only exact conditional-scale standardization; significance/detection is outside the selected package | no significance/detection claim |
| full-procedure response | ODQ-008 selects fixed-state response only; exact relearning/re-estimation graph and perturbation family remain for ODQ-010 | no full-procedure response claim |
| effective matched-filter beam/solid angle | ODQ-008 requires derivation from the exact response under an explicit measure/domain/convention; no numerical response representation or derivation is selected | parent nominal beam remains provenance only |
| calibration covariance for template amplitude | parent/template dependence must be joint, but numerical dependence/covariance facts are not supplied | missing contribution unavailable, not zero or estimator normalization |
| detailed public product bundle/VAL profiles | ODQ-013 fixes tiered atomic roles/lifecycle and FLT policy ownership, while exact persistence and profile-granularity options remain unauthored/unselected | publication unavailable pending package-local contract authorship and owner disposition |
| package-local Stage A packet and author manifest | holding-study owner decisions and `SCI-FLT-MATCHED` identity are complete, but the package-local recovery/scope/firewall packet has not yet received exact-byte owner approval | Stage B dispatch unavailable |
| FLT→FRUIT interface | minimum scientific identity is approved by ODQ-012; exact persistence/reconstruction options and any future FRUIT-required additions remain unauthored/unselected | interface must be preserved in FLT; FRUIT science remains unavailable pending its own tranche |

## Confidence assessment

| Recovery conclusion | Confidence | Basis |
| --- | --- | --- |
| one combined `SCI-FLT-INF` contract is scientifically incoherent | high | roadmap split rule plus distinct estimand/state/response/lifecycle families |
| active full path is structurally template-amplitude-like | medium-high as implementation observation | direct algebra inspection; implementation conformity remains unassessed |
| owner-selected scientific identity of the historical full path is matched-template amplitude | authoritative | exact ODQ-001 owner approval |
| ordinary-MAP observation and coadd parents are both admitted but non-equivalent | authoritative | exact ODQ-003 owner approval |
| historical radially symmetrized average map noise PSD is the selected model | false/unselected | ODQ-004 admits it only as an author-evaluated candidate |
| base-v0.1 template is one immutable declared response-per-unit-amplitude product | authoritative | exact ODQ-005 owner approval |
| exact reference estimator is `A_hat=<t,Qm>/<t,Qt>` and approximations require a selected quantitative envelope | authoritative | exact ODQ-006 owner approval |
| base-v0.1 output admission is complete-support-only and adaptive edge conditioning is deferred | authoritative | exact ODQ-007 owner approval |
| template-amplitude units and exact location-indexed fixed-state response apply; no universal kernel or inherited parent beam is presumed | authoritative | exact ODQ-008 owner approval |
| active full path is a complete posterior/Wiener reconstruction | excluded as scientific identity | exact ODQ-001 owner approval; no explicit signal prior or posterior covariance recovered |
| current NOI-member application is learned-once/fixed-state | high as implementation observation | state is resolved from real parent and reused for members |
| per-member relearning is currently active | low/negative recovery result | no active route found; absence limited to inspected base |
| adaptive edge operation is scientifically consequential | high | mask/background/window are parent-derived and alter signal/weight/kernel |
| `normalize_errors` is part of the base estimator | low | it occurs downstream through empirical NOI products/coefficient scaling |
| registry's missing commit equals the recovered later commit | unavailable | matching file digest is insufficient to identify missing commit contents |

## Stop conditions

Stage B must not be commissioned while:

- the package-local Scope Brief and exact author manifest lack owner approval;
- multiple selected estimands remain in one proposed package;
- method substitution can occur without explicit realized identity;
- fixed-state and relearned NOI graphs are not separated; or
- the proposed author inputs contain implementation-derived conclusions.

Scientific freeze and every numerical route remain blocked until the owner
selects or otherwise disposes of the authored ODQ-004 option set and the
ODQ-006 quantitative conformance-envelope option set. Missing noise/covariance
authority and missing approximate-route bounds remain typed unavailable until
then.
