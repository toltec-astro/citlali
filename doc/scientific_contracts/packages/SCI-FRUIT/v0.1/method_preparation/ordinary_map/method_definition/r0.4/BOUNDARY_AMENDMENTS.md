# Proposed bounded PTC and MAP amendments

Revision r0.3, 2026-09-09. These are proposed scientific clauses for Q02,
not adopted upstream authority. Frozen PTC v0.1/r0.5, CAL v0.1/r0.5 and MAP
v0.1/r0.7.1 remain controlling. “Shall” below specifies the proposed successor.
No frozen source, VAL registry or source-binding register is edited here.

## B01. Exact residual-pass permission and ownership

Proposed amendment identity: `SCI-FRUIT-OM-BC-IN@proposal-r0.3`.
An explicitly requested PTC residual-parent pass shall admit only the exact
immutable CAL-derived residual constructed by M01–M04, under an approved
instance of this specialization. This is an additional scoped route alongside
the unchanged ordinary CAL-parent route. An external-residual identifier,
compatible array shape or FRUIT request alone shall not activate it.

FRUIT shall own model inference, subtraction/rejoin, iteration placement,
continuation and terminal selection. PTC shall own one configured-rank
Learn–Consider/resolve–Apply pass on the actual residual. PTC shall acquire no
internal FRUIT recurrence or astronomical source selector. Bootstrap uses the
existing ordinary CAL route; feedback passes require this separate permission.

## B02. Actual parent and typed compatibility

Each residual pass shall bind: this amendment and method identity; observation,
array/group/segment; exact original measured ancestry and CAL realization;
iteration; candidate/accepted/applied model identities; inference evidence and
decision; projection/embedding; subtraction; residual-generation identity;
units, beam/template/spectral/calibration reference; occurrence and coordinate
associations; all support roles; origin and transitive influence; and required
permissions and lifecycle states. Rows are samples, columns are exact detector
occurrences. A shape, timestamp or position match shall not establish identity.

Its numerical parent shall be rho_k, not its CAL ancestor or prior cleaned
residual. Quantity compatibility shall satisfy B08. Missing or conflicting
required facts shall block the pass at their declared scope before its output
is published; no CAL fallback, implicit no-op, zero fill or borrowed parent.

## B03. One fit on this immutable residual

Within each admitted pass/group/segment, PTC shall compute the ordinary
detector-wise, binary-influence arithmetic mean over its finite
basis-fit-admitted residual population and learn the requested positive-rank
PCA/SVD subspace once. Internal scaling remains identity. It shall use one
declared grouping level, with no cross-array fit; array-wide grouping remains
the first-method proposal requiring Q01/Q03 applicability. Exact use supports,
fit/metric and rank/degeneracy guards come from the bound effective PTC plan.

Resolve Theta_k before Apply. Its location and subspace shall remain fixed
within that application and conditional queries. The next pass learns on its
new immutable residual; identical resulting subspaces are allowed. There shall
be zero support-changing refinement, automatic rank search, zero-rank rescue,
lower-rank fallback, forced state change or previous-Theta substitution.
Learned centering/eigenvectors are results, not preselected owner settings.

## B04. Application, reference loss and removed signal

Apply shall use the complete ordinary resolved PTC map on rho_k, including its
discarded centering location lambda_k and exact group-local masked coordinate
solve, support, metric, normalization, matrix-rank/tolerance and failure rules.
Reevaluating modal coordinates under this map shall not refit the subspace.
The complete map may be affine; its linear part is a distinct response object.

Publish the fitted correlated component and total removed signal separately:

    U_total,k = rho_k - PTC.Apply(rho_k; Theta_k).

Neither the CAL ancestor nor only the fitted correlated component may replace
that definition. Preserve lambda_k, realized removed subspace, fixed-state null
space and known full-procedure invariant/unidentifiable-mode states. The model
rejoin shall not restore lambda_k or claim measured optical DC. Failure of the
required solve invalidates its required output, without changing rank/support.

## B05. Conditional response and retained output

A fixed-state kernel/perturbation shall receive the exact linear part of the
current resolved Apply map on its declared domain. It shall not have lambda_k
subtracted from the perturbation, refit PTC, bypass the data's required support
and solve guards, or borrow another pass's operator. An input response already
realized on the PTC grid shall not receive upstream processing a second time.

For this explicit residual route, the existing PTC output-retention predicates
and four-axis decisions shall apply to the residual-derived PTC product with
its actual parents and causes. Adoption must bind that route's applicability
to the exact `SCI-PTC:output_retention@1` semantics and controlled source
identity; this proposal does not silently expand a registered evaluation.
CAL classification, direct origin, model influence and every required failure
cause shall persist. Retention shall establish neither MAP membership nor
independence, sky truth, model fitness or stronger response/uncertainty.

## B06. PTC clause disposition

All identifiers below refer to copied final PTC r0.5 sources, not an inferred
latest candidate. This is the exact scope of the proposed residual exception.

| Frozen clause | Proposed residual-pass disposition |
| --- | --- |
| REQ-001; CAL-parent definition and ordinary input requirements | Add B01–B02 as an explicitly requested alternative parent for this pass only. Preserve ordinary CAL input semantics. |
| REQ-081 / external model-residual identity; detailed PTC-OD-011 | Supply B02 identity and B08 compatibility; identification is necessary but not permission. B01 is the separate proposed permission. |
| REQ-023, REQ-093 | Preserve mean/fit-support/binary-influence rule and nonrestoration; evaluate on the actual residual. |
| REQ-024 | Preserve identity internal scaling. |
| REQ-090–092, REQ-094–095 | Preserve ordinary configured positive rank, one grouping level and no adaptive/zero-rank/fallback route; bind exact effective request. Only parent permission changes. |
| REQ-096 | One fit per immutable residual pass/group, zero support-changing refinements. Repetition is FRUIT's outer orchestration over new parents, not a PTC internal refinement. |
| REQ-083 | Preserve complete fixed-state application and solve guards; total removed signal is relative to rho_k as B04 states. |
| REQ-062, REQ-097 | Preserve fixed-state linear-part kernel semantics and data-equivalent guards; no centering subtraction from perturbations. |
| REQ-063–065 | Retain full-procedure response definitions and required disclosures. Numerical availability is not granted by this amendment. |
| REQ-052–055; PTC-OD-010 | B09 supplies an explicit proposed analysis/gridding family and QC/retained-use records; PCA fit coefficients remain separate. |
| REQ-078–079, REQ-069 | Preserve forbidden input/quantity routes and producer ownership; PTC output is a transformed intermediate, not an independent sky estimate. |
| REQ-099 | Apply only B07's bounded disposition. No blanket removal or reinterpretation of its exclusions. |

## B07. Explicit REQ-099 boundary

The ordinary CAL route shall retain all REQ-099 exclusions. For the separately
requested residual pass, permit only the externally constructed sky-subtracted
parent and repeated invocation by the separately authorized FRUIT method.
Source protection and recurrence remain outside PTC; this permission does not
make them options of ordinary PTC or permit PTC to select sources internally.

Automatic rank, support-changing refit, conditioned-r influence, simulation
or empirical input populations, numerical full-procedure response, and stronger
covariance remain excluded from this bounded route. Numerical full-procedure
PTC/FRUIT response or an injection population needs its own exact authority,
domain and state-change protocol before execution. RF-03/04/05 definitions and
prospective experiments in this packet are requirements/proposals, not proof
that those tiers are available. A local conditional kernel supplies none of
these permissions. No raw-frequency BEAM, joint x/r, cross-array, polarimetric
or global mapmaking route is added.

## B08. Correction-domain and additive-reference proposal

Proposed embedding identity: `SCI-FRUIT-OM-CAL-X-CORRECTION@proposal-r0.3`.
Use only the inherited nonpolarimetric, top-of-atmosphere,
point-source-equivalent calibrated-x quantity in mJy per fixed nominal beam,
with exact beam/template, spectral/calibration and coordinate lineage. Unit
spelling alone, including a STOKES token, establishes no compatibility.

The selected MAP representative shall be interpreted as a declared correction
in that quantity's difference space. Project by the same AST/WCS half-open
containing pixel as M04, with scale one and no added offset. Removal and rejoin
use this declared correction representative on matching applicable supports;
outside D_a the explicit residual-only branch applies. In D_a a resolved
inference rejection is intentional zero correction, not missing data or a
measured zero. An unknown model value shall not be embedded as zero.

The embedding binds the measured/reference origin and its losses separately
from model/prior-supported content. The complete affine PTC output retains its
current lambda_k and unconstrained-mode disclosure; adding the chosen correction
does not invert centering, restore absolute sky or establish unit sky response.
No additional rebase, beam conversion, response inversion, normalization,
extrapolation or unit conversion is implicit. If those exact conventions do
not furnish compatible additive operands, the route stays unavailable and Q02
must resolve the specific mismatch. The conditional H B m = m identity is
restricted to finite supported complete restoration of the applied model.

## B09. Analysis/gridding family, current QC and retained use

Retain proposed PTC-owned family
`PTC-ANALYSIS/UNIFORM-OCCURRENCE@proposal-r0.1`. It assigns dimensionless
gamma_i=1 on exact bootstrap PTC output-retained occurrences satisfying the
declared family, finite transformed-signal, valid occurrence/support and
compatible product/unit/group requirements. This population I_Gamma,0,a is
fixed before first MAP formation and immutable thereafter. MAP admission and pixel
placement remain separate. There is no fit statistic, noise law or empirical
normalization in this constant family. MAP alone supplies its ordinary N/Q
normalization. This explicit proposal is never an inferred unity fallback.

The family record shall bind owner/version, exact index (detector-time
occurrence here; no implicit broadcast), value-generation identity, population
and support, constant definition and factors, unit, normalization, provenance,
payload readability/typed availability, lifecycle, uncertainty and forbidden
meanings. Values retain their bootstrap generation. Theta_k, modal coordinates
and lambda_k are different coefficients/state and receive current generations.

Proposed PTC-owned QC identity:
`SCI-PTC:uniform_occurrence_gridding_qc@proposal-r0.1`. Each current evaluation
shall bind the family/value generation and current PTC application, exact
occurrence mapping and immutable parents. At bootstrap, residual/model/rejoin
roles are not_applicable with the bootstrap reason; later evaluations bind
the actual residual and rejoined generations. Each evaluation shall test
family identity/permission, index/population compatibility, readable typed
coefficient availability, current PTC retention, required product/support and
finite transformed-signal facts, and absence of an unauthorized value change.
Its four axes are those in B11. Missing/conflicting required facts yield unknown
applicability and unavailable decision; decisive false predicates yield
ineligibility; all true yield eligibility, with independent realization state.
Only requested/applicable/eligible/realized passes producer QC.

The retained-use record shall explicitly bind old coefficient generation and
new product/QC generation, compatible quantity/reference/occurrences, unchanged
values and population, actual learning/model influence and all causes. A new
Theta does not itself invalidate a constant analysis family, nor does constant
value excuse fresh compatibility/QC. Required mismatch/failure blocks this
method; it does not trigger reweighting, membership shrinkage or renormalization.
Only MAP classifies coefficient values for finite-positive contribution; typed
availability is not finiteness, and producer QC cannot rescue MAP nonmembership.
These values establish no precision, covariance, sensitivity, S/N or NOI law.

| Record | Distinct current meaning |
| --- | --- |
| gamma_0 and QC_0 | Bootstrap family/value generation and its actual PTC/CAL compatibility decision, before MAP; no inherited pass result. |
| gamma_(0->k), QC_k and MAP admission_k | Same coefficient values with explicit retained-use compatibility and new current evaluations; learned Theta_k and model/rejoin parents remain distinct. |
| Residual- or post-rejoin-estimated gamma_k | Not requested. Either would be a separate estimator/population and method choice, never an alias of retained gamma_0. |

## B10. Proposed MAP input and boundary successor

Proposed successor identity: `SCI-PTC_TO_SCI-MAP v0.1/r0.2`, status
`proposed_not_adopted`. It preserves r0.1's logical handoff and ordinary PTC
branch, adding an explicitly tagged FRUIT-rejoined descendant branch. For the
new branch, z_i shall be the exact Zjoin_k occurrence, never relabeled as
unchanged PTC output. All B01–B09 permissions, records and current retention/QC
must be supplied; bootstrap remains the ordinary PTC branch.

The handoff shall carry the actual CAL, applied model, residual, learned and
resolved Theta_k, PTC result, projection and rejoin parents; request/effective
plan/realization/publication states; quantity/reference/support; origin and
transitive influence; coefficient value and current compatibility/QC;
response/uncertainty class, conditioning, unavailable terms and causes; and
the exact admission evaluation. Preserve observation, detector occurrence/UID,
stable RTC output n, segment/group, PTC generation and distinct FRUIT generation.
AST signal coordinates shall join on the same n and exact ancestry, not time,
shape, numeric coordinate equality or container position.

The original exposure branch remains byte-semantically that of the frozen
boundary: ALIGN acquired/valid-original seconds, original occurrence/donor
lineage, deduplication and the AST original ALIGN-grid footprint in the target
WCS. Rejoin, repeated iterations, model support and descendants create or
relocate no exposure. Signal placement and original-footprint placement remain
different coordinate roles. No new response/noise product is invented.

## B11. Proposed MAP admission-profile successor and clause crosswalk

Proposed profile: `SCI-MAP:map_upstream_admission@3`, status
`proposed_not_registered`. MAP owns the scientific predicate; VAL registers
and evaluates its exact approved version. It shall have two explicitly
identified routes, never selected by shape similarity:

1. Ordinary PTC: preserve every @2 applicability, retention, coefficient/QC,
   coordinate, cause and failure predicate without weakening it.
2. FRUIT-rejoined: require the exact requested method and adopted B01–B10
   permissions, actual current PTC result eligible for retention, complete
   rejoined product and authorized correction, exact identity/quantity/reference/
   support/coordinate compatibility, and the exact B09 coefficient family and
   requested/applicable/eligible/realized current QC. Required direct synthesized
   or replaced measured-representative origins remain vetoes. A declared model
   correction is a distinct operation with recorded influence, not permission
   to erase an upstream origin veto. Preserve CAL engineering-only classification
   where PTC retains it; this creates no science qualification.

Each occurrence-level result shall retain independent request
{requested, not_requested}, applicability
{applicable, inapplicable, applicability_unknown}, eligibility
{eligible, ineligible, decision_unavailable} and realization
{realized, incomplete, failed, not_produced} axes. Only
requested/applicable/eligible/realized shall pass. Missing/conflicting required
permission, identity, generation, parent, coefficient or coordinate facts
produce applicability_unknown and decision_unavailable; a decisive false
predicate produces ineligible; all required predicates true produce eligible.
No-product, disabled PTC, retention/QC nonpass and unsupported direct CAL route
remain decisive. Preserve all causes and per-predicate evidence; no universal
veto follows merely from a transitive influence label.

Response and uncertainty remain advisory to base signal admission, with exact
state/domain/limitations carried; a requested claim may require them separately.
A profile pass establishes only a route candidate. MAP shall still perform all
structural/payload/finite-positive/coordinate/contribution/companion gates before
accumulation and apply its exact frozen arithmetic and failure scopes.

| Frozen MAP requirement / record | Proposed change or preserved duty |
| --- | --- |
| REQ-002; PTC-to-MAP boundary r0.1 signal | Add only B10's explicitly tagged descendant input; preserve ordinary input. |
| REQ-003, REQ-043, REQ-046 | Add FRUIT generation and all parents while preserving occurrence and PTC identity. |
| REQ-004; profile @2 | Add this scoped @3 branch, retaining separate producer QC and MAP four-axis decisions. No @2 alias or retroactive change. |
| REQ-005 and original-footprint boundary | Same-n AST signal join and original ALIGN-grid exposure join unchanged. |
| REQ-011/015/016/026/029–032 | One-hot half-open placement, finite-positive coefficient, normalization and exact support-policy formulas unchanged; numeric support policy still Q04. |
| REQ-047, REQ-049–050 | All gates before aggregate mutation; retain complete unfiltered base bundle, immutable generations and required publication/failure semantics. |
| REQ-051 | Same-operator noise propagation only with its actual state/support/coefficient conditioning; not full iterative NOI. |
| REQ-052 | Narrowly permit the B10 descendant input and its recorded FRUIT dependence; keep its other excluded routes and downstream claims unavailable. Mode registration remains B12. |

Adoption shall publish the exact owner-approved successor sources, matching
MAP/VAL profile copies, revised VAL registry and source-binding rows, and their
manifest hashes together. It must name which clauses of frozen MAP are amended;
the remaining arithmetic binds exactly v0.1/r0.7.1. No hypothetical future hash
or registry version is asserted here. Prior @2/boundary r0.1 products and
evaluations remain immutable and are not compatibility aliases of the proposal.

## B12. Mode capability and remaining boundary limits

M05's observing-mode policy interface is required now. It does not supply a
numerical input or establish downstream POINT, OOF, BEAM or catalogue fitness.
SCI-MAP-OD-006 leaves POINT/OOF ordinary-MAP mode/WCS registration open; Q02
shall bind the exact input/output and mode registration before those runs.
PTC REQ-078 and MAP REQ-052 do not admit raw frequency-shift BEAM or an OOF
transfer/residual route by analogy. A CAL-bound BEAM case must demonstrate the
actual calibration-relevant quantity compatibility; otherwise it needs a
separate input/quantity authority, not this correction embedding.

No JINC, optimal/maximum-likelihood mapmaking, reprojection, empirical-noise
method, coadd population, global solver, downstream fitting or alternative
PTC estimator is admitted by these clauses. Response and NOI methods, simulation
inputs, numerical policy settings and execution retain their separate gates.
