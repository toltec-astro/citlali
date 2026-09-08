# SCI-MAP v0.2 document r0.4 semantic-change report

Status: **CANDIDATE for explicit scientific-owner disposition**. The baseline
is the exact SCI-MAP v0.2/r0.3 accepted architecture. All 52 requirement and
25 prediction identities remain unchanged. The owner authorized only the four
bounded categories below.

## Four bounded change categories

| Category | r0.3 defect or need | r0.4 candidate correction | Stable subjects |
| --- | --- | --- | --- |
| 1. PRED-025 semantic clarification | The prior wording treated zero too broadly and conflicted with the admitted estimator and zero-based indexing. | Zero is classified by quantity. A finite zero signal numerator with finite-positive normalization and valid support, its normalized zero, an in-bounds zero index, and a valid zero centered-integer offset remain admissible. Zero normalization forbids division and row support; a zero coefficient remains noncontributing; `coverage_cut = 0` and `N = 0, Q_star = 0` retain their meanings; non-finite, overflowed, out-of-domain, and unrepresentable quantities retain their separate failure scopes. Empty applied output support retains outcome `not_produced` and cause `no_support_authorized_output_rows`. Fixtures A--D record the discriminators. | PRED-025; REQ-009, 012, 031--035, 038--039, 043, 045--048; `records/PRED025_QUANTITY_SPECIFIC_ZERO_FIXTURES.md`. |
| 2. ECS sequencing clarification | Execution step 3 could require a decision artifact before producing that artifact. | For a new decision, the ECS verifies profile semantics, approval, Registry/source binding, activation/evaluability, and required object facts; evaluates; verifies the resulting artifact identity, four-axis state, causes, and realization; then permits MAP consumption. An existing decision is consumed only after exact binding and realization verification. No new VAL disposition logic is introduced. | ECS execution step 3 and existing VAL handoff record model. |
| 3. Cross-reference and prose repair | Full empty-support prose was inserted into sentences intended to carry a short reference; punctuation and a vague cause name were defective. | The complete empty-support rule appears once under label `rule:empty-support-disposition`; short grammatical references preserve each REQ obligation. The exact token `no_support_authorized_output_rows` replaces vague cause wording. | REQ-009, 012, 031--033, 039, 048 and their ECS renderings. |
| 4. Deduplication and pagination | The complete lifecycle appeared twice per view, candidate status was over-repeated, and avoidable page breaks created a one-entry contents spill and a mostly empty vocabulary continuation. | Each view presents one complete five-stage lifecycle table and uses precise references elsewhere. Candidate status remains visible once per rendered view. Normal flow removes the avoidable ECS spills, and the rationale authority paragraph is merged into its existing authority/status section. Typography controls are unchanged. | Shared lifecycle/status macros and the three view entries. |

## Preserved authority and science

The estimator, projection, coefficient family, exposure model,
`coverage_cut` policy, six product roles, response taxonomy, coadd arithmetic,
NOI claim ceilings, downstream ownership, all separate failure scopes, and
the accepted r0.3 source architecture remain unchanged. The occurrence
profile keeps identity `SCI-MAP:map_upstream_admission@2` and its exact
predicates. The aggregate profile remains
`SCI-MAP:observation_coadd_admission@2`. The retained SCI-VAL Registry identity
prefix is `v0.1/r0.3`; only its MAP generation suffix advances to
`map-v0.2-r0.4-candidate-2026-09-08`.

The literal admitted uniform `@draft-0.1` identities and boundary
`v0.2-draft.1/r0.4` remain unchanged, as do their historical approvals and
evaluations. No new selected family, parameter, estimator, requirement,
prediction, decision function, or route is introduced.

## Claim disposition

These exact r0.4 bytes await explicit conditional scientific-owner acceptance
and a later post-owner implementation-blind consistency and exact-SHA review.
They do not activate candidate profiles, realize a PTC handoff, supply an
unavailable parent, authorize a numerical route, establish implementation or
response/covariance fidelity, validate observations, establish significance,
or authorize production.
