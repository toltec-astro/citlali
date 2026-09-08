# SCI-MAP v0.2/r0.2 requirement and prediction parity report

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
`6a7e64377e31b8b0e4dac90a27a7662ac825bd59c5a87d8237eb533d7b8d37c0`,
an exact canonical-macro parse found 52 unique contiguous requirement IDs and
25 unique contiguous prediction IDs in numerical order. The corresponding
Markdown tables contain the same ordered sets. The initial JSON contains the
same 52 requirement identities, all and only `result = not_assessed` and
`artifact_realization = not_produced`. All seven JSON template files parse as
JSON. This is a source-inventory result, not an ECS requirement pass or an
implementation-conformance result; the manager's final imported-source verifier
must reproduce it.

## Directive amendment parity

| Amendment | Requirement coverage | Prediction coverage |
| --- | --- | --- |
| Five-stage lifecycle | REQ-009, 031, 046--048 | PRED-012, 025 and role predictions where lifecycle matters |
| Six product roles and aggregate response rule | REQ-036--041, 043, 046--050 | PRED-013--019 |
| Full-procedure comparison gate | REQ-008, 016--017, 046, 050 | PRED-005--006, 013 |
| NOI terminology | REQ-019--024, 050 | PRED-019, 022 |
| ECS result/artifact vocabulary | All 52 evidence records and final verdict | All prospective prediction results remain `not_assessed` |
| Product/notation cleanup | REQ-006, 016, 042--043, 046 | PRED-010, 019--020 |
| Typed coverage override/small N | REQ-009, 031--032, 046--047 | PRED-012, 025 |
| WCS finite population | REQ-038, 043, 045--047 | PRED-018, 025 |

## Mechanical final-packet expectation

The manager's final verifier shall parse canonical macro invocations in
`src/common/requirements.tex` and `src/common/edge_cases.tex`, not arbitrary
mentions in prose; compare exact contiguous sets; verify every crosswalk row;
and check the 52 JSON placeholders by the fields `requirement_id`, `result`,
and `artifact_realization`. This author report does not promote that final
mechanical result into scientific acceptance or conformance.
