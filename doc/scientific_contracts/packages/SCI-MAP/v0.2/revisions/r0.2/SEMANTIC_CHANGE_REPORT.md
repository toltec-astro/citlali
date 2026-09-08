# SCI-MAP v0.2 document r0.2 semantic-change report

Status: **CANDIDATE for scientific-owner disposition**. Baseline is the exact
SCI-MAP v0.2/r0.1 author basis. The owner directed a bounded revision, not a
new MAP derivation. All 52 requirement and 25 prediction IDs are retained.

## Changed semantics

| Directive area | r0.1 issue | r0.2 candidate behavior | Stable subjects |
| --- | --- | --- | --- |
| Ordered lifecycle | Four- and five-stage wording conflicted. | One order: `requested/effective/observation_resolved/applied/realized`. Only reached stages exist; a pre-application stop has no applied/realized record, and an applied terminal failure has no fabricated realized product. | REQ-009, 031, 046--048; PRED-012 and provenance/profile prose. |
| Product roles | Base and stronger response/covariance products lacked six durable independently requestable role tokens. | Six exact observation/coadd roles, each with request, parent, ordered membership, lifecycle, dependencies, and failure scope. Stronger roles do not rewrite base parents. | REQ-036--041, 043, 046--050; PRED-013--019. |
| Aggregate companion admission | PRED-014 could reject every response-incompatible base-coadd member. | Under `SCI-MAP:base_coadd@1`, unavailable/basis-incompatible response preserves signal membership and produces unavailable coadd response with exact cause. A companion can reject atomically only when the selected role/profile requires it. Existing stronger-role requests never subset a realized base parent. | PRED-014; profile `SCI-MAP:observation_coadd_admission@2`; REQ-037, 040--041, 047, 049. |
| Full-procedure response | Two arrays or a state delta could be read as enough to subtract. | Every numerical PTC full-procedure, MAP-of-PTC, and PTC+MAP re-resolved difference requires exact product/quantity/unit/rows/frame/WCS/reference/gauge/null/support/basis/perturbation/common-domain/both-map/unmatched-row/convention/state/availability fields. | Equations 13--16; REQ-008, 016--017; PRED-005--006, 013. |
| NOI terminology | Generic empirical-noise/significance language exceeded the current NOI claim. | NOI owns its exact realization-generation, empirical-uncertainty, conditional-scale, and standardization methods/products. Q, formal precision, complete covariance, exact conditional marginal moment, standardized signal, and future significance remain distinct. Stronger interpretations are unavailable by default. | REQ-024, 050; PRED-022; all views. |
| ECS vocabulary | Result and artifact states were not fully normalized. | Requirement result is exactly `pass/fail/blocked/not_applicable/not_assessed`; evidence-artifact realization is separately `complete/incomplete/failed/not_produced`. All 52 placeholders initialize to `not_assessed`/`not_produced`. | ECS evidence table/verdict; templates. |
| Product/notation identities | Undefined MAP-local publication `q` and a `K_p` response/covariance collision remained. | PTC publication is bound by its exact product/application/role identity; local undefined `q` is removed. `K_p` remains response accumulator; stacked observation covariance is `bold Sigma^obs`. `Pi` remains the MAP plan and `J_out` the row selector. | REQ-006, 042; equations and prose. |
| Coverage-cut override | An unbound function-like expert request and inconsistent stage count could self-authorize `c>1`. | Exact typed `coverage_cut_above_one_override_requested` binds request/authority/scope/reached stages/value/cause/provenance. Pre-application checks `c_observation_resolved`; `c_applied` binds the event; `c_realized` is later only. Domain, inclusivity, exact order statistic, no default, and small-N transitions are explicit. | REQ-009, 031--032, 046--047; PRED-012, 025. |
| WCS maximum | An unspecified representative subset could be mistaken for a global maximum. | The exact finite population includes every scientifically addressable center and every outer-footprint boundary vertex/corner. Same generation/frame/orientation, exact index relation, binary64 great-circle metric, fail-on-invalid policy, and 0.1 arcsec maximum over only that finite population are bound. | REQ-038, 045--046; PRED-018. |
| Source closure | r0.1 lacked exact successor role/profile/VAL identities. | Exact candidate scientific source, role, occurrence/aggregate profile, and paired candidate SCI-VAL Registry/register bytes and hashes are supplied. New route remains candidate and not source-closed/frozen pending acceptance/activation/freeze. | Source/profile/binding records and all covers. |

## Preserved without scientific change

- ordinary positive-coefficient observation mapping and equal-observation
  centered-integer coaddition;
- exact realized positive-rank PTC transformed calibrated-`x` input, with no
  direct CAL, PTC-disabled, or inferred no-op route;
- stable occurrence identity, same-`n` AST signal coordinate, and
  `SCI-MAP:one_hot_containing_pixel@1` half-open placement;
- owner-approved uniform Registry/family/handoff and all separate
  selection/publication/association/handoff/admission gates, with no unity,
  alternate-family, or default fallback;
- exact `A_MAP,Pi` over support-authorized scientific rows and separation of
  MAP-local, fixed-state, full-procedure, re-resolved, and whole-chain response;
- distinct geometric, route, contribution, count, exposure, support, and
  MAP-validity facts; original occurrence exposure uses its own AST coordinate;
- exact order statistic, two inclusive thresholds, no universal cut default;
- complete immutable bundles, covariance honesty, no independence inference,
  no correlated-GLS promotion, immutable base/unfiltered parentage; and
- every existing claim ceiling and nonclaim.

## Identity dispositions

The literal admitted uniform `@draft-0.1` IDs and boundary
`v0.2-draft.1/r0.4` are retained, with no revocation or relabeling. Occurrence
policy `SCI-MAP:map_upstream_admission@2` retains its predicates under a new
candidate source binding. Changed aggregate semantics receive
`SCI-MAP:observation_coadd_admission@2`; historical `@1` is immutable and no
alias. The six product-role IDs and two candidate SCI-VAL generation IDs are
new durable records explicitly requested by the owner directive; no new REQ
or PRED obligation was needed.

## Claim disposition

This report describes intended contract semantics only. It reports no
scientific acceptance, implementation availability/conformity, representation
or response fidelity, covariance fidelity, validation, observational
performance, achieved significance, readiness, production suitability or
authorization, or Unity activity.

