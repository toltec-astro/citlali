# SCI-MAP v0.2/r0.4 requirement and prediction parity report

Status: **author-side candidate traceability report**. It records source-level
inventory and amendment mapping, not application evidence or conformance.

## Stable inventories

| Inventory | Required set | Candidate initialization |
| --- | --- | --- |
| Requirements | Exactly one each of `SCI-MAP-REQ-001` through `SCI-MAP-REQ-052`; no other REQ ID | ECS and `templates/REQUIREMENT_RESULTS_INITIAL.json` initialize every result to `not_assessed` and artifact realization to `not_produced`. |
| Predictions | Exactly one each of `SCI-MAP-PRED-001` through `SCI-MAP-PRED-025`; no other PRED ID | ECS prospective fixture/bound/result remains `not_assessed`. |

The complete per-ID mappings are `../CROSSWALK.md` and
`../PREDICTION_CROSSWALK.md`. No new independent obligation was found, so no
new stable ID was added.

## Author-source snapshot result

On the stable author-source bytes with core aggregate
`3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31`,
an exact canonical-macro parse found 52 unique contiguous requirement IDs and
25 unique contiguous prediction IDs in numerical order. The corresponding
Markdown tables contain the same ordered sets. The initial JSON contains the
same 52 requirement identities, all and only `result = not_assessed` and
`artifact_realization = not_produced`, with no MAP attempt, product, or
failure-evidence identity asserted. All seven JSON template files parse as
JSON. This is a source-inventory result, not an ECS requirement pass or an
implementation-conformance result; the manager's final imported-source
verifier must reproduce it.

## Directive amendment parity

| Amendment | Requirement coverage | Prediction coverage |
| --- | --- | --- |
| Quantity-specific zero and fixtures A--D | REQ-009, 012, 031--035, 038--039, 043, 045--048 | PRED-001, 003, 012, 015, 017--018, 025 |
| Empty-support short references and exact cause | REQ-009, 012, 031--033, 039, 048 | PRED-012, 015, 017, 025 |
| ECS evaluation/verification/consumption order | REQ-004, 006, 037, 039, 046 | PRED-010, 013--017 |
| Lifecycle/status deduplication and view parity | REQ-001, 009, 043, 046--048 | All prediction renderings inherit one shared presentation. |
| Retained profile and role bindings | REQ-004, 006, 036--041, 043, 046--050 | PRED-010, 013--019 |
| Retained response, comparison, NOI, WCS, and cut rules | REQ-008, 016--024, 031--032, 038, 040--046, 050 | PRED-005--007, 012--013, 016, 018--022 |

## Mechanical final-packet expectation

The manager's final verifier shall parse canonical macro invocations in
`src/common/requirements.tex` and `src/common/edge_cases.tex`, compare the
exact contiguous sets and every crosswalk row, and check the 52 JSON
placeholders by `requirement_id`, `result`, and `artifact_realization`. It
shall also verify that PRED-025 cannot be implemented as a generic
“zero means invalid” rule. This author report does not promote that mechanical
result into scientific acceptance or conformance.
