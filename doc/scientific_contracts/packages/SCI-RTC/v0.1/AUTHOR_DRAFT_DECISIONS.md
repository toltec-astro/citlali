# SCI-RTC v0.1/r0.1 author-draft decisions

Status: implementation-blind author choices for scientific-owner review; none
is scientific-owner approval.

These decisions consolidate the four approved author inputs into one shared
normative core. They do not select the open numeric and policy choices in
`SCIENTIFIC_OWNER_DECISION_LEDGER.md`.

| ID | Author-draft decision | Packet basis and consequence |
| --- | --- | --- |
| `SCI-RTC-AUTHOR-D001` | The six `src/common/*.tex` files are one normative core and are imported exactly once by each view. View-specific prose is explanatory or informative conformance guidance only. | Enforces one scientific authority without duplicating equations or requirements between audiences. |
| `SCI-RTC-AUTHOR-D002` | The revision identity is `v0.1/r0.1`; requirement IDs are sequential `SCI-RTC-REQ-001` through `SCI-RTC-REQ-054`, and prediction IDs are sequential `SCI-RTC-PRED-001` through `SCI-RTC-PRED-026`. | Provides stable, exact review and crosswalk targets. |
| `SCI-RTC-AUTHOR-D003` | Product-role meaning is a tuple of role, quantity, unit, sign, reference/baseline, and valid domain. The role-domain operator is identity for raw roles and an imported CAL operator only for separately authorized calibrated roles. | Reconciles raw Beammap `Delta f/f` with optional calibrated `mJy/beam` without cross-role inheritance. |
| `SCI-RTC-AUTHOR-D004` | The owner-modified raw donor rule is represented literally as `flxscale_q/flxscale_d` under `z_i = flxscale_i x_i`, with exact occurrence/domain compatibility and a nonzero target factor. | Replaces the retained core's obsolete responsivity derivation; no legacy responsivity role is retained or required. |
| `SCI-RTC-AUTHOR-D005` | Calibration-before-replacement is the default for a CAL-authorized role. An alternative must prove equality of the complete affine operators, including offsets, response, uncertainty, identity, validity, and domain. | Preserves the binding order decision while making the allowed equivalence falsifiable. |
| `SCI-RTC-AUTHOR-D006` | The central role-specific RTC operator is expressed once in the shared equations as ALIGN, optional role CAL, replacement, ordered filters, and phase-zero selection plus declared affine state terms. | Reuses and specializes the retained derivation instead of independently repeating it in either view. |
| `SCI-RTC-AUTHOR-D007` | Phase zero means point selection at input index `Mn`. For a coherent segment, the representative assigned-grid time is the selected point time and cardinality is zero for empty input or `1 + floor((N-1)/M)` otherwise. | Resolves identity/cardinality directly from the binding phase-zero operator; full scientific support still expands through every prior stage. |
| `SCI-RTC-AUTHOR-D008` | Transitive influence is the cause-preserving closure through alignment, donor, filter, state, edge, and sampling dependencies. Any synthesized or replaced source in that closure makes the output scientifically ineligible. | Implements the strengthened owner rule beyond center-only or direct-weight semantics. |
| `SCI-RTC-AUTHOR-D009` | Complete response is a local/factorized detector-time response containing every enabled response-changing stage, or it is typed unavailable. Scalar LTI response is restricted to a proved fixed, detector-separable, translation-invariant interior. | Prevents partial-kernel promotion and preserves limiting LTI reasoning. |
| `SCI-RTC-AUTHOR-D010` | The state lifecycle is requested, effective, observation-resolved, learned/resolved when applicable, and realized; learned narrative additionally names bootstrap and applied records without backflow. | Reconciles the general one-way state rule with the learned-mode lifecycle. |
| `SCI-RTC-AUTHOR-D011` | Learned resolution is represented as the maximum factor in a finite safe set whose predicates are conjunctive. Missing numeric owner policy leaves the learned plan unavailable rather than treating the predicate as passed. | Preserves maximum-safe reduction and exposes all owner decisions needed before application. |
| `SCI-RTC-AUTHOR-D012` | Conditional covariance, selection covariance, nuisance/systematic covariance, model covariance, and cross terms remain separate. The corrected retained total-covariance term is `Sigma_y^stat`. | Applies the supersession cover's transcription correction and prevents unknown terms from becoming zero. |
| `SCI-RTC-AUTHOR-D013` | Output eligibility is not a universal downstream bit. The shared core fixes only the RTC synthesis/replacement exclusions and separated validity inputs; named consumers may further restrict but not weaken them. | Preserves VAL ownership while making the owner-approved transitive exclusions binding. |
| `SCI-RTC-AUTHOR-D014` | The atomic bundle includes TOD, ordered identity, complete response status, support, influence, typed causes, separated validity, uncertainty availability, one-way provenance, diagnostics, and completion/failure. | Makes a finite TOD explicitly insufficient without prescribing a storage schema. |
| `SCI-RTC-AUTHOR-D015` | Diagnostics are classified as inert, advisory, or selected-policy inputs; optional inert detail cannot alter any scientific/numerical output or failure. | Reconciles observational diagnostics with learned or other policy effects. |
| `SCI-RTC-AUTHOR-D016` | The scientist-facing view uses ten substantive explanatory narrative pages before the shared-core appendices; the engineering view contains the same shared authority plus nonnormative evidence guidance and no independently restated displayed equations. | Implements the requested audience genres while preserving one authority. |
| `SCI-RTC-AUTHOR-D017` | Deterministic predictions cover role separation, donor direction/availability, operator order, response, masks, edges, non-finite state, phase zero, aliasing, covariance, learned-plan selection/restart, reset, and inert detail. | Converts the retained limiting cases into stable package-level falsification IDs without asserting any test result. |
| `SCI-RTC-AUTHOR-D018` | Open selected-policy and numeric choices are retained in a separate owner ledger with an exact unavailable consequence for each; successor-only modes are marked deferred. | Prevents author inference from current behavior and keeps unaffected RTC products independently available. |

## Review disposition

At scientific-owner review, each decision should be accepted, modified, or
rejected explicitly. A modification that changes normative meaning requires a
new revision and synchronized regeneration of both PDFs and the exact
crosswalk. Compilation, mechanical coverage, or visual QA cannot approve any
decision in this file.
