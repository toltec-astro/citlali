# SCI-FRUIT — final Stage A amendment and packet-parity report

Document: v0.1/r0.4 (amended). Date: 2026-09-07.
Manager/owner review only; not an author input.
Scientific owner: Grant Wilson. Exact amended-packet approval pending.

The r0.4 architecture is accepted. This final bounded amendment applies the
owner's additive-space, response-catalog and packet-parity directions. It
does not reopen the architecture or perform another broad Stage A derivation.
The unamended r0.4 packet is preserved; the exact successor has the distinct
identity SCI-FRUIT-SCIENTIFIC-CORE-STAGE-A-R0.4-AMENDED.

## Final-feedback crosswalk

| Owner section | Exact amendment | Location |
| --- | --- | --- |
| 1. Additive spaces | Removal and rejoin require compatible scientific quantity, unit/numerical scale, frame/domain, grid/sampling, support, additive origin/reference or gauge, null-space state, calibration and response conventions. Y_k/Z_k are additive spaces or explicitly referenced affine/quotient spaces with well-defined arithmetic. Missing compatibility is unavailable, and a rejoined upstream-unavailable mode remains model_or_prior_supported. | Scope 2/4; notation type declarations; method additive-compatibility row; lifecycle bundle; support; response/uncertainty; predictions |
| 2. Response catalog | Retains the preferred seven-role interpretation. RF-03 is one iteration with prior S_k fixed. RF-04 fixes the sequence/index. RF-05 includes convergence/stop/resource/selection. RF-07 perturbs realized parent-domain payload with upstream producer operation, learned state and generation fixed, and references the RF-03/RF-04/RF-05 components of the complete FRUIT-only composition. RF-06 reruns included upstream producers. Old meanings have an exact supersession crosswalk; components are not independent aliases of the composition. | Scope 7; response catalog; ledger supersession table; method record; program instructions |
| 2. Response states | Requires all response-role identities and typed statuses required by the applicable method and claim. Each binds perturbation source/domain, fixed/rerun state, recurrence/terminal scope, baseline/perturbed comparability, branch/support/state changes, codomain, derivative or finite-difference convention, and unavailable/discontinuity behavior. Required statuses may be unavailable; no universal numerical response availability follows. | Scope 7; response catalog; method record; manifest |
| 3. Model spaces | Separately binds M_k^candidate, M_k^accepted, M_k^applied and every transformation. No equality of domain, support, unit, normalization, sign, representation, response, uncertainty or parentage is assumed. Equality is an exact method/state fact. Applied model remains the removal/rejoin operand. | Scope; notation; method; lifecycle; support; response; predictions |
| 4. Index wording | After completion of the iteration with absolute zero-based index N, continuation begins at absolute index N+1 from S_(N+1). N is an index, not a count. | Scope 6; lifecycle; boundaries; ledger; program; prediction 013 |
| 5. Terminal record | Adds all nine requested fields or typed unavailable statuses: selected iteration identity, complete considered population, rule/generation, convergence/stop/resource facts, causes, terminal response, terminal uncertainty, continuation/terminality, and downstream-use status references. No mutation, relabeling or replacement of selected bundle. Hard selection dependencies remain required. | Lifecycle terminal table; scope 4; method; boundaries; prediction 016 |
| 6. Packet parity | Exact source/payload identities, frozen PTC parity, current role counts, prediction IDs, additive premises, completion/method gates, links, manifest and archive sidecars checked. | Exact manifest and digest sidecars |
| 7. Stage B | Launch a fresh implementation-blind Stage B thread only after Grant approves the exact amended packet. Architecture acceptance does not satisfy that separate condition. Numerical methods and every existing nonclaim remain. | Scope 8–11; program; ledger; manifest |

## Response identity and conditioning checks

The current catalog names agree exactly between Scope Brief and response table:

1. RF-01 fixed one-step measured-input response.
2. RF-02 fixed one-step applied-model response.
3. RF-03 one-iteration FRUIT full-procedure response.
4. RF-04 fixed-sequence recursive response.
5. RF-05 selected-terminal response.
6. RF-06 whole-chain response.
7. RF-07 complete FRUIT-only parent-domain response.

The ledger maps every prior r0.4 response identity to its amended disposition.
RF-07's old fixed-numerical-parent meaning is retired. Its numerical query
payload now varies while the upstream producer operation/state/generation stay
fixed. Query payloads have separate identities and immutable baseline-parent
ancestry; they do not mutate the original measurement or claim a new producer
realization. RF-03 remains local, RF-04 has fixed extent, RF-05 includes terminal
decisions, and RF-07 is the outer composition. Exact degenerate specializations
reference an existing component response/status rather than duplicating it.

RF-01/RF-02 retain the unchanged conditioned map and all its defining state.
Coefficient recomputation required to evaluate that fixed map is still carried
out; it is not silently replaced by frozen-component subtraction. A change to
the recomputation rule changes method identity, while new evaluated coefficients
under an unchanged rule have distinct realized state/application generations.

## Preserved science and prediction coverage

The corrected three-term fixed-F_k expansion remains. D_k, its linear model
response and the complete joint covariance require exact fixed linearity of
A_k, Pi_k and B_k in the declared compatible spaces and conditioning. Both
parent/applied-model cross terms remain. Reference/gauge/null-space conditioning
does not turn a missing mode into a zero-uncertainty observation. Model-supported
content in such a mode is not observationally_measured, recovered_from_data,
acquired_exposure or unit_response_truth. This is a mode-origin annotation,
not an added fifth occurrence-support class.

All twenty SCI-FRUIT-PRED-001 through SCI-FRUIT-PRED-020 IDs remain. Their
scientific cases are retained: 001–005 limiting/model-error cases; 006 PTC
application distinction; 007–008 support/origin; 009–010 response and state
changes; 011 covariance; 012 shared-parent dependence; 013–014 continuation/
restart/relearning; 015–016 stopping/terminal identity; 017 grouping; 018 NOI
method/ensemble separation; 019 method-instance admission; 020 core completeness
versus downstream fitness. The final amendment adds reference/gauge/null-space
premises and consequences to the relevant cases and makes index/terminal state
explicit without adding experiments or numerical defaults.

All seven operation-state categories, eight support/influence roles and four
occurrence-origin classes remain. Universal core completion gates cannot be
waived. A/B/C identities are unchanged; FRUIT-FEEDBACK-METHOD v0.1 is still a
family requiring exact separately approved method-specific identities. Parent
quantity remains route-conditional. Immutable ancestry, no default recurrence,
separate lifecycle roles, downstream ownership and all nonclaims are preserved.
The frozen PTC fragment remains byte-identical, including its source-local
notation and restrictions, with its cover in the boundary and notation files.

## Read/write boundary and exact closure

The pre-amendment branch is codex/sci-fruit-v0.1-empirical-lane at
f9dfa82b35203b56c48a2aecf9b03053ed7dfdc5. Its tracked tree was clean; the two
known untracked review archives remained present. Writes target only the new
scientific_core/r0.4-amended directory. Previous packets, archives and existing
reduction products remain outside the write scope. No push is performed.

Only the eleven r0.4 author payloads, their manifest and digest sidecar, the
current owner amendment and newly authored packet files were read, apart from
Git metadata and document-packaging work. The thirteen prior input identities
are verified. Original upstream sources, internal recovery, implementation,
configuration, schemas, tests, audits/repairs, reductions, generated numerical
products, validation, performance, production, Unity and web material are not
inspected. Upstream source hashes are carried from the exact r0.4 manifest as
provenance, not as newly verified source contents. Unlisted status/navigation
files are not edited under this retained read boundary.

The packet-parity check verifies byte counts and SHA-256 values for every
payload, the manifest sidecar, deterministic compressed archive membership and
member bytes, the archive sidecar, required scientific table/ID coverage and
zero unresolved bundle-local links. These are document checks, not numerical
tests, replay or qualification. The archive contains eleven author-permitted
payloads, this manager-only report, the verbatim final owner amendment, the
manifest and its sidecar. The archive digest remains external to avoid a hash
cycle. Manager-only provenance is excluded from fresh author input.

Exact amended-packet approval is the remaining Stage A owner decision. After
that approval, the already directed fresh implementation-blind Stage B launch
may proceed. No author is launched by this amendment, and numerical-method
availability remains unavailable_pending_separate_owner_approval.
