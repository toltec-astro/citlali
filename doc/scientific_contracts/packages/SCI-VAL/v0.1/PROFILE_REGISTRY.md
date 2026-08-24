# SCI-VAL v0.1 — Owner-Bound Profile Registry

Status: r0.3 continuing contract registry; one mandatory canonical atomic
profile registered and source-current; package-qualified policy records
otherwise unbound unless stated

Date: `2026-08-24`

## Registry Rule

The registry is a binding and replay mechanism. It does not write, approve,
or inherit scientific-use policy. A usable profile record must bind:

1. immutable registry key and profile version;
2. exact named scientific use;
3. actual scientific owner;
4. authoritative source and exact version or digest;
5. applicability domain and object type;
6. required restrictions and missing-fact behavior;
7. permitted exceptions, including any non-exceptionable invariants;
8. response and uncertainty roles, each selected from `structural_gate`,
   `required_permission`, `decisive_exclusion`, or `advisory`;
9. aggregation and propagation compatibility; and
10. supersession and incompatibility behavior.

Missing or conflicting binding information makes the profile unavailable for
evaluation. A reserved name is not a usable policy.

## Mandatory Canonical Registered Profile

| Field | Binding |
| --- | --- |
| Registry key | `SCI-VAL:independent_exposure@1` |
| Named use | `independent_exposure` |
| Scientific owner | Grant Wilson, as scientific owner of the owner-approved SCI-VAL v0.1 contract-level invariant |
| Authoritative source | `REVISION_DIRECTIVE_R0.3.md`, `VAL-R03-D001`, preserving the invariant approved in `REVISION_DIRECTIVE_R0.2.md`, `VAL-R02-D003`; formal clauses `SCI-VAL-REQ-019`, `SCI-VAL-REQ-043`, and `SCI-VAL-REQ-045`; current adjacent-source compatibility is bound by `SOURCE_BINDING_REGISTER.md` SHA-256 `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430`, including frozen SCI-ALIGN r0.3, frozen SCI-RTC r0.12, and the approved WP-2 exposure-lineage boundary |
| Domain | Exact sample-detector occurrence with authoritative representative-source identity and origin state |
| Decisive invariant | An exact representative source that is synthesized or replaced is not an original independent astronomical exposure |
| Exception permission | None for the decisive invariant; an attempted exception makes the supplied policy invalid |
| Other restrictions | Must be supplied by the actual scientific owner in an immutable compatible successor record; VAL supplies none |
| Compatibility | Compatible only when the direct-origin invariant, domain identity, and no-exception rule are preserved exactly; a weakening requires a different named use and is not a compatible successor |
| Response/uncertainty roles | No role is imposed by this direct-origin invariant; any added compatible restriction must select only from the four closed roles and retain the same owner/source binding discipline |
| Aggregation and propagation compatibility | `atomic_only`. Aggregation and reverse propagation are `not_applicable` under this profile identity. Any future aggregate is a separately registered, owner-bound proposition under the Aggregate Profile Rule. Any propagation it authorizes creates a new derived proposition and lifecycle generation; it preserves the exact atomic source references and cannot rewrite the atomic decision or the original acquisition and valid-original facts |
| Missing/conflicting behavior | `applicability_unknown` plus `decision_unavailable` when structural/profile binding or required origin authority is unresolved; an authoritative synthesized/replaced origin is `ineligible` |

This registry entry names its scientific owner explicitly so that the
contract-level invariant is not misread as a VAL-authored downstream policy.
The `SCI-VAL` namespace identifies the registering package; it does not make
SCI-VAL Core the policy owner.

The former draft key `VAL.core.independent_exposure@1` is not registered and
has no compatibility alias. An input carrying that key is an unbound profile,
not a migration accepted by inference.

The atomic-only disposition does not define a generic usable exposure,
detector fraction, retained exposure, projected exposure, coadd exposure, or
threshold. Physical-acquisition and valid-original facts remain owned by
SCI-ALIGN and preserved through the approved timestream exposure-lineage
boundary. A future scientific use owns any additional use-qualified exposure
quantity it defines.

## Aggregate Profile Rule

Every aggregate is a distinct scientific proposition and therefore requires
its own complete registry record. In addition to the general fields above, an
aggregate profile binds:

1. its own immutable registry identity and version;
2. the exact compatible atomic source-profile identity and version;
3. its scientific owner and authoritative source;
4. aggregate object type and applicability domain;
5. population and time support;
6. numerical operator, denominator and missing-decision treatment;
7. threshold and boundary polarity, if any;
8. advisory or binding role, uncertainty treatment, and failure scope; and
9. propagation authority and lifecycle-generation rule.

An aggregate cannot reuse the atomic profile identity or appear as another
atomic instance. For illustration only,
`SCI-PTC:detector_independent_exposure_fraction@1` could identify a future
PTC-owned detector aggregate whose atomic source is exactly
`SCI-VAL:independent_exposure@1`; this example is hypothetical and is not a
registered profile in v0.1. Base atomic compatibility remains homogeneous.
No heterogeneous transformation profile is registered here.

## Package-Qualified Names Reserved But Not Bound Here

The following names prevent broad labels such as `estimator_fit` from hiding
scientifically different questions. They identify expected owners, not usable
policies. Each remains unbound until its owner supplies a complete immutable
record satisfying the registry rule.

| Reserved profile name | Expected scientific owner | Status in this registry |
| --- | --- | --- |
| `SCI-PTC:basis_fit_admission` | SCI-PTC | Unbound; no predicates or thresholds supplied by VAL |
| `SCI-PTC:loading_fit_admission` | SCI-PTC | Unbound; distinct from basis fitting |
| `SCI-PTC:operator_application` | SCI-PTC | Unbound; fit exclusion cannot define it by inference |
| `SCI-PTC:output_retention` | SCI-PTC | Unbound; post-fit output action remains stage-specific |
| `SCI-PTC:coefficient_qc_population` | SCI-PTC | Unbound; coefficient/QC estimation is not an alias for estimator fitting |
| `SCI-PTC:response_companion` | SCI-PTC | Unbound; response availability and admission remain owner-supplied |
| `SCI-PTC:empirical_or_simulation_population` | SCI-PTC | Unbound; no realization-population policy is inferred |
| `SCI-MAP:map_upstream_admission` | SCI-MAP | Unbound; accepted MAP boundary exists, but no exact owner-approved profile is registered here |
| `<PACKAGE>:diagnostic_display` | Owning package or diagnostic consumer | Namespace template only; display permission conveys no stronger scientific use |

The earlier broad label `analysis_or_gridding_contribution` is not a v0.1
registry key. It is replaced by the narrower `map_upstream_admission`; no
automatic alias is permitted because the old wording could be mistaken for
actual numerical contribution.

## Registry Change Rule

A new source digest, profile version, domain, restriction, exception rule, or
compatibility declaration creates a new immutable registry record. It cannot
rewrite an earlier evaluated decision. A renamed profile is a new identity
unless its scientific owner publishes an explicit semantics-preserving alias;
VAL does not infer aliasing.
