# SCI-FRUIT — Stage A semantic-change report r0.4

Manager/owner review only; not an author input.
Scientific owner: Grant Wilson. Date: 2026-09-07.
Status: exact r0.4 candidate for owner review; no Stage B author launched.

This is the requested bounded micro-repair of the accepted r0.3 architecture.
It corrects inconsistent model operands, linearity premises, method/state
identity, parent quantity wording, response enumeration and completion gates.
It does not reopen the conditional-core architecture or perform a broad new
Stage A derivation. Program-order authority A and conditional-composition
authority B retain their stable September 6 identities; neither supplies
numerical-method authority C.

## Changes traced to the owner feedback

| Owner section | Concrete correction | Author-facing location |
| --- | --- | --- |
| 1. Model roles | Separate m_k^candidate, m_k^accepted and m_k^applied, including their domains. Pi_k and B_k consume the applied operand. Accepted/applied equality needs an exact method/state equality; all intervening transformations are explicit. Transition, bundles, response queries, covariance axes and predictions follow that distinction. | Scope 2–4/6; notation; lifecycle; method template; support; response/uncertainty; predictions 001–006, 009, 011 |
| 2. Fixed linear expansion | Fixed linear F_k=A_k alone gives the explicit three-term map expression. D_k is defined and used as a linear model-path operator only when A_k, Pi_k and B_k are all exact fixed linear on declared domains, units, support, state and conditioning. | Scope 2; notation; response/uncertainty; ledger; predictions 003/005/011 |
| 3. Method and realized state | Changes of rule or operator family change method identity. New coefficient values, learned states and model operands under unchanged rules change state/iteration/application generations. Scheduled relearning can occur during exact continuation without a method change. | Method template; lifecycle; scope; boundaries; response/NOI cover; predictions 006/014/018 |
| 4. Parent route | Every selected route supplies its own quantity, unit, frame, response, beam/calibration, support, validity, lifecycle and provenance. The ordinary PTC-to-MAP mJy/beam convention is explicitly conditional, with no generic input unit, Stokes I or true-sky inference. | Scope 3; notation; method template; boundaries B3 |
| 5. Response enumeration | Exactly seven named roles: the six required distinct roles plus the retained FRUIT-full-procedure role with upstream parent held fixed. Each binds perturbed/fixed/rerun facts, baseline/perturbed domains, comparison scope, codomain and unavailable/discontinuity behavior. No online adaptive estimator is introduced. | Identical names in Scope 7 and response table RF-01 through RF-07; ledger and method template |
| 6. Universal completion | The core's minimum bundle and hard claim dependencies cannot be waived. Methods may add requirements and named-use optional companions. Typed unavailable values preserve required roles and permit only a narrower claim admitted by both core and method. | Scope 4; lifecycle completion gates; method template; support; boundaries B5; prediction 020 |
| 7. Authority family | FRUIT-FEEDBACK-METHOD v0.1 names the required-record family. Every numerical method requires its own exact `FRUIT-FEEDBACK-METHOD/<method-name>@<revision>` identity and approval. Creating the family approves no instance. | Method template; scope; ledger; program; manifest; prediction 019 |
| 8. Packet closure | Exact payload identities, permissions, author links, frozen excerpt, required counts, manifest/sidecars and archive membership are checked. | AUTHOR_INPUT_MANIFEST.md and both digest sidecars |
| 9. Stage B disposition | After exact r0.4 packet approval, launch in a fresh implementation-blind thread. Acceptance of r0.3 as a repair basis does not satisfy that condition. Every numerical method and downstream availability/nonclaim boundary remains. | Scope 8–11; program launch direction; ledger; manifest |

## Response identity crosswalk

The previous catalog combined two fixed-input responses in RF-01. The r0.4
catalog separates them and keeps the extra parent-held-fixed role explicit.
These catalog revisions are not numerical-method approvals.

| r0.3 role | r0.4 role |
| --- | --- |
| RF-01 combined fixed one-step | RF-01 fixed one-step measured-input response and RF-02 fixed one-step model-input response |
| RF-02 one-step full procedure | RF-03 one-step FRUIT full-procedure response |
| RF-03 recursive multi-iteration | RF-04 recursive multi-iteration response |
| RF-04 selected terminal | RF-05 selected-terminal response |
| RF-06 whole chain | RF-06 whole-chain response |
| RF-05 upstream parent held fixed | RF-07 FRUIT full-procedure response with upstream parent held fixed |

All twenty SCI-FRUIT-PRED-001 through SCI-FRUIT-PRED-020 identities are retained.
Their requested scientific cases remain: 001–005 limiting/model-error cases;
006 PTC application distinction; 007–008 support and origin; 009–010 response
dependence/discontinuities; 011 joint covariance; 012 shared-parent dependence;
013–014 continuation/restart/relearning; 015–016 stopping/terminal identity;
017 grouping; 018 NOI procedure separation; 019 numerical-method admission;
020 core completeness versus downstream fitness. Corrections refine their
premises and roles, without introducing a numerical experiment.

## Preserved boundaries and exact inputs

The frozen PTC excerpt is byte-identical to r0.3, including its source-local
notation and full-rank/support restrictions. Its cover remains in the
boundaries and notation files. All seven operation-state categories, eight
support/influence roles and four information-origin classes are retained.
Candidate/accepted/applied support subdivisions do not create extra top-level
support roles. The full covariance still contains measured-input, applied-model
and both cross terms; candidate/accepted uncertainty is not substituted for
applied uncertainty by implication.

Immutable original measured-parent ancestry, no default input rule, separate
continuation/restart/branch/retry/completion/terminal identity, grouping and
shared dependence remain. Model-only content creates no acquired exposure.
NOI fixed-state and per-realization-relearning rules remain separate methods/
ensembles, while members under an unchanged rule have distinct realized states.
SCI-VAL evaluates named-use policy and does not own or execute FRUIT recurrence.
Terminal selection references an immutable completed iteration. All previous
nonclaims remain, including unavailable numerical science, conformity,
qualification, fitness, performance and production authority.

Only the eleven r0.3 author payloads, their manifest and manifest sidecar,
the current supplied owner direction, and newly authored packet files were
read for this micro-repair, apart from Git metadata. The prior payloads were
verified against their exact manifest before editing. Original upstream
sources, earlier internal recovery, implementation, configuration, schemas,
tests, audits/repairs, reductions, generated numerical products, validation,
production, performance, Unity and web materials were not inspected.
Inherited original-source identities are carried as provenance, not claimed
as newly verified contents or additional author inputs.

The pre-repair branch is codex/sci-fruit-v0.1-empirical-lane at
83d3d01a3a792d887706407513906d2ed7774c35. Its tracked tree was clean; the two
known untracked review archives remained present. Writes are limited to the
new scientific_core/r0.4 directory. Earlier packets, archives and reduction
products are untouched. Unlisted status/navigation documents are not edited
under the retained read boundary; this report records the new disposition.

## Closure and owner decision

Closure verifies each manifested payload's byte count and SHA-256, the manifest
sidecar, archive sidecar, exact archive member bytes and absence of extra paths.
It checks all model roles, complete linear premises, method/state separation,
the matching seven-role response catalog, seven state categories, eight support
roles, covariance cross terms, twenty predictions, A/B/C, method-specific
identity obligations, eleven Scope Brief sections and all bundle-local links.
These are document integrity checks, not numerical tests or replay.

The document closure checks passed for the delivered packet. Fixed-response
wording also preserves coefficient recomputation required by the unchanged
map; it does not silently substitute frozen-component subtraction. A query
that changes a method-defining rule requires an explicitly identified method
comparison rather than a state alias.

The archive contains eleven author-permitted payloads, this manager-only report,
the verbatim current micro-repair request, the exact manifest and its sidecar.
The archive's digest sidecar is external to avoid a hash cycle. Author
permissions remain explicit; manager provenance is not an author input.

The next owner action is approval of the exact r0.4 packet bytes. That approval
triggers the requested fresh implementation-blind Stage B launch; it does not
approve a numerical method or the later authored scientific substance. Current
numerical-method availability remains unavailable_pending_separate_owner_approval.
