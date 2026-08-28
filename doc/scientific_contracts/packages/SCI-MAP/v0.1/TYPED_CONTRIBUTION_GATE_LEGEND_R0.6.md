# SCI-MAP v0.1 r0.6 Typed Contribution-Gate Legend

The normative ten-row legend is rendered in the formal contract. For every
gate `g`, MAP retains typed state `sigma_g(i,p)` and causes, then derives the
Boolean estimator-membership projection `b_g(i,p)`. Contribution membership
is `prod_g b_g=1`; payloads are evaluated only after membership is known.
Exclusion is never represented by multiplying a payload by zero.

| Gate | Owner and fact | Passing projection | Nonpassing state and retained provenance |
| --- | --- | --- | --- |
| signal availability | PTC transformed signal | exact signal available | unavailable/conflict; PTC cause and generation retained |
| output retention | PTC/VAL `output_retention@1` | requested, applicable, eligible, realized | every other four-axis tuple; complete decision retained |
| coefficient availability | PTC coefficient declaration | exact value/family available for occurrence | unavailable/conflict; declaration and causes retained |
| coefficient QC | PTC coefficient/QC decision | exact named permission passes | excluded/unavailable; QC decision and causes retained |
| MAP upstream admission | MAP policy, VAL evaluation | `@2` requested, applicable, eligible, realized | every other tuple; full evaluation retained |
| AST validity | AST same-`n` coordinate | exact bound coordinate valid | invalid/unavailable/conflict; AST parent and causes retained |
| signal finiteness | MAP on PTC value | finite `z_i` | nonfinite/unavailable; value class and provenance retained |
| coefficient positivity/finiteness | MAP on PTC value | finite and strictly positive | zero/negative/nonfinite/unavailable; value class retained |
| projection/boundary | MAP one-hot plan | unique in-grid half-open pixel | outer boundary/out of grid/invalid; coordinate and decision retained |
| MAP-local companion/contribution | MAP effective plan | all role-required companions and local permissions pass | excluded/unavailable; exact failed predicate, scope, and causes retained |
