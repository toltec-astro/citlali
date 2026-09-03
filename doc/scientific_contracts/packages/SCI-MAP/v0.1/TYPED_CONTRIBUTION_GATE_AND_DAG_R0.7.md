# SCI-MAP v0.1 r0.7 Typed Contribution-Gate Table And Evaluation DAG

The normative ten-row legend is rendered in the formal contract. For every
gate `g`, MAP retains typed state `sigma_g(i,p)`, immutable reason, causes, and
failure scope, then derives Boolean estimator-membership projection
`b_g(i,p)`. Contribution membership is `prod_g b_g=1`. Exclusion is never
represented by multiplying a payload by zero.

Every VAL-evaluated gate has the complete axes: request is `requested` or
`not_requested`; applicability is `applicable`, `inapplicable`, or
`applicability_unknown`; eligibility is `eligible`, `ineligible`, or
`decision_unavailable`; realization is `realized`, `incomplete`, `failed`, or
`not_produced`. The sole pass is `(requested, applicable, eligible, realized)`.
Every other tuple is nonpassing. Missing or incompatible source/profile
binding is a structural failure, with exact cause and immutable reason; it is
never repaired from similarly named state.

| Gate | Owner and fact | Passing projection | Nonpassing state and retained provenance |
| --- | --- | --- | --- |
| signal availability | PTC transformed signal | exact signal available | unavailable/conflict; PTC cause and generation retained |
| output retention | PTC/VAL `output_retention@1` | requested, applicable, eligible, realized | every other four-axis tuple; complete decision retained |
| coefficient availability | PTC coefficient declaration | exact family/generation, index/broadcast, transformed-product compatibility, readable payload presence, typed availability and cause | missing/unreadable/unavailable/conflict/generation mismatch; no finiteness claim |
| coefficient QC | PTC-owned coefficient/QC profile | `pass_gamma_qc=1` only for requested/applicable/eligible/realized | every other tuple or structural failure; exact PTC decision retained; never MAP admission or rescue |
| MAP upstream admission | MAP policy, VAL evaluation | `@2` requested, applicable, eligible, realized | every other tuple; full evaluation retained |
| AST validity | AST same-`n` coordinate | exact bound coordinate valid | invalid/unavailable/conflict; AST parent and causes retained |
| signal finiteness | MAP on PTC value | finite `z_i` | nonfinite/unavailable; value class and provenance retained |
| coefficient numerical class | MAP on authorized PTC value | finite and strictly positive | exact zero noncontributing; negative finite, non-finite, or unrepresentable invalid; exact class retained |
| projection/boundary | MAP one-hot plan | unique in-grid half-open pixel | outer boundary/out of grid/invalid; coordinate and decision retained |
| MAP-local companion/contribution | MAP effective plan | all role-required companions and local permissions pass | excluded/unavailable; exact failed predicate, scope, and causes retained |

## Ordered evaluation DAG

1. **A — structural:** identity/parent, generation, source binding, profile
   binding, producer availability, and required decision realization.
2. **B — safe numerical classification:** retrieve only structurally admitted
   payloads; classify `z_i` finite/non-finite/unrepresentable and `gamma_i`
   positive/zero/negative/non-finite/unrepresentable.
3. **C — coordinate and product:** AST coordinate validity, one-hot
   placement/boundary, and MAP-local required companions.
4. **D — membership:** form the conjunction of the ten Boolean projections.
5. **E — accumulation:** perform numerical payload arithmetic only for exact
   members.

Reading an authorized payload for Stage B classification is not accumulation.
All required gates pass before payload arithmetic, so an excluded non-finite
value cannot contaminate an accumulator.
