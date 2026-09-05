# Independent exact-SHA review

Date: 2026-09-05. Reviewer: independent fresh-context read-only agent, distinct from the manager and scientific author.

**Verdict: REPAIR REQUIRED for readiness for owner scientific review.** There is one major scientific/behavioral consistency finding in the proposed UQ-05 failure classification. Architecture/ownership and repository/evidence checks pass with no findings. The numerical uniform-family proposal is otherwise coherent within the released scope. This verdict does not adopt science, activate a Registry record, freeze a contract, approve integration, or constitute the later post-owner implementation-blind consistency review.

## Candidate SHA and tree

| Identity | Exact value |
| --- | --- |
| Candidate | `f605be6f23560ef81aa05598091964081b8fe749` |
| Candidate tree | `5f222389e5b504640fde13b198db97e6222ccc53` |
| Sole parent / released Stage A subject | `959bb90947d4b9651fbba9f16a0eb73dde893b84` |
| Parent tree | `1e067d1d661d1ebd0bd8e05ce6cb4c9d652b8437` |
| Canonical base and unchanged current canonical tip | `00b974c9039d4c3025dcce18f26bca69d36af9c3` |
| Canonical tree | `6ca6ebe6561e1a3ba5c7fbeb5f555022f739c19b` |
| Owned branch | `codex/cti-od-002-coefficient-authority-2026-09-05` |
| Owned worktree | `/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05` |
| Canonical worktree, inspected read-only | `/private/tmp/citlali-refactor-mainline-after-map-space-repair-2026-09-04` |

Verified the sole parent, literal candidate/tree, branch identity, canonical ancestry of the parent, and parent ancestry of the candidate. Candidate and canonical worktrees were clean initially and at final checking, including fully enumerated untracked state. No repository, index, ref, branch, source artifact, or canonical worktree was mutated by this review.

## Work-order scope and authority sources read

Purpose: Tier 2 independent review of the first proposed `SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1` author return after the exact Stage A owner release. Owner: Grant Wilson. This is contract-library drafting/evidence, with no application implementation slot.

Governance preflight read repository `AGENTS.md`, `doc/governance/ENGINEERING_GOVERNANCE.md`, and `doc/governance/REVIEW_AND_CONFORMANCE.md`, including its independent-review template. Verified accepted governance `06a3ade51c1b3f38887295433d913811bf25cd14` is on canonical ancestry and its canonical incorporation record names the exact accepted document digests. Current candidate, accepted-object, and canonical bytes agree:

| Governance document | SHA-256 |
| --- | --- |
| Engineering Governance | `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76` |
| Review And Conformance | `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1` |

The documents' retained candidate headers do not override their accepted canonical incorporation. Timestream Successor governance was not invoked because this bounded study does not perform Timestream application work.

Process/sequencing sources: contract-library README, `PILOT_PROCESS_REVIEW_2026-08-16.md`, applicable dated status change, study README, exact released Scope Brief/recovery synthesis/process excerpts, Stage B release, prior exact-input review receipt, manager QA/completion, and mechanical-check record. The TolTEC context skill and routing maps were read for routing; the PDF skill was used for read-only artifact inspection. The roadmap was taken only through the released process excerpt's approved-roadmap binding.

Scientific inputs were limited to the exact released 11-file packet and its byte-identical copies, then the proposed draft. All seven admitted references were read: PROCESS_AND_SCOPE, PTC_COEFFICIENTS, REGISTRY_FRAMEWORK, MAP_BOUNDARY, JINC_BOUNDARY, NOI_BINDING, and VAL_RULES. No implementation, historical audit/repair evidence, other derivation, surrounding scientific source, or external scientific reference was opened. Manager records supplied provenance/review context only.

Subject-specific authority remains separated: the released Scope Brief controls included derivation; admitted PTC and registry excerpts control coefficient identity/ownership and lifecycle; MAP and JINC retain their separate admissions and numerical rules; NOI retains the MAP-derived design and rational-parent obligations; VAL supplies evaluation mechanics. Every new numerical, support, profile, permission, and boundary choice remains an open author proposal. Actual-owner confirmation is still explicitly open and blocks eventual Registry binding, without independently blocking draft review.

## Scientific/behavioral findings: major

### S1 — UQ-05 assigns incompatible outcomes to overlapping payload failures

**Evidence.** The claimed exact UQ-05 truth-domain rule in [definitions.tex](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/definitions.tex:192), especially lines 198–203, assigns **F** to a completed authoritative check reporting corruption or incomplete/extra/duplicate broadcast membership. The same restriction's table at [definitions.tex](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/definitions.tex:233) says missing or unreadable payload is unresolved/unavailable. The failure table at [assumptions.tex](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/assumptions.tex:69) assigns unreadable scalar/broadcast to decision unavailable or failed realization. Under the draft's own composition rule, a nonstructural F instead produces ineligible.

Concrete overlapping case: assume otherwise complete, consistent structural identities and a completed authoritative producer check of supplied scalar bytes using the declared decoder. If that check establishes corruption that makes the scalar unreadable, the exact UQ-05 paragraph supplies F/ineligible, while the UQ-05 row and failure table supply unresolved/decision-unavailable or failed realization. The draft does not distinguish the state in which unreadability is an established negative payload-fidelity fact from one in which a check or required fact is unavailable. It also does not identify which realization axis would fail in the failure-table alternative; the profile evaluation can itself be successfully realized while reporting a payload failure.

A related overlap occurs within this same classification issue. The structural gate at [definitions.tex](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/definitions.tex:160) explicitly treats duplicate/ambiguous occurrence identity as structural unavailability; [assumptions.tex](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/assumptions.tex:60) does the same for missing/conflicting broadcast identity or binding. UQ-05 nevertheless lists incomplete/extra/duplicate broadcast membership as F. A distinction between an authoritative, valid member domain and a stored payload representation that fails to reproduce that domain could make these cases coherent, but the draft does not state that distinction or consistently limit the F case to it. The profile inputs use the same member/broadcast binding vocabulary for both stages.

These clauses are present in the rendered artifacts, not merely stale source: scientific PDF page 4 contains both the truth-domain paragraph and the UQ-05 row, and page 8 contains the unreadable-payload failure row; engineering PDF pages 3 and 7 contain the corresponding truth-domain and failure rules.

**Affected authority.** Released [Scope Brief requirement 3](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_a/author_packet/SCOPE_BRIEF.md:64) requires complete T/F/U/C interpretation, missing/conflict behavior, and failure scopes. Admitted [VAL rules](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_a/author_packet/references/VAL_RULES.md:68) distinguish authoritative false from unavailable information; its [general structural gate](/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05/doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_a/author_packet/references/VAL_RULES.md:170) makes missing/conflicting structural material unavailable. Draft UFC-REQ-012, 014, 015, and 017 claim these exact distinctions.

**Consequence.** The owner is currently being offered more than one scientific decision result for the same failure class under a purportedly complete profile. Later conformance evidence could not select a unique expected eligibility/realization result. No route is currently active, so this finding establishes no application regression or operational failure; it blocks the claimed internal completeness needed for this candidate's review-readiness verdict.

**Smallest bounded repair.** Return this one issue to the isolated scientific author within the already released inputs. State disjoint authoritative structural-binding failures, completed negative payload-fidelity checks, absent/unevaluated/unreadable inputs, and profile-artifact realization failures. Explicitly distinguish the authoritative occurrence domain/broadcast identity from any malformed payload representation, if that is the intended distinction. Make UQ-05's truth rules, its table row, and the failure table agree for each case, while retaining the admitted VAL structural precedence and independent axes. Add only the discriminating cases needed to demonstrate those distinctions to the existing evidence/prediction traceability. The reviewer does not select the scientific classification or a new policy.

Regenerate both views from the same repaired modules, advance the document revision as the charter requires, and refresh only affected companion checks/manifests and manager evidence. Preserve this exact candidate and the sealed first-return bytes as predecessor evidence. A new candidate SHA requires a new exact-SHA review. No extra family, consumer rule, threshold, default, Registry activation, or stylistic round is requested.

**Other scientific assessment.** No additional actionable finding. Exact dimensionless unity, the finite nonempty single-observation support, scalar-to-occurrence relation, arithmetic-mean-to-one construction, no-parameter rule, and singleton/empty distinction form a coherent proposed numerical definition. No statistical weighting, precision, independence, or optimality claim is introduced. Crosswalk and ledger coverage are complete; their mechanical completeness does not resolve S1.

## Architecture/ownership findings: none

The proposed producer is SCI-PTC, owning coefficient meaning, immutable generations, and coefficient-use QC. MAP and JINC have independently open consumer permissions and retain their own admission and numerical classification. JINC retains its signed spatial coefficient and bundle rules. NOI remains a MAP-hosted consumer of exact parent contributions and lossless rational binding; neither its design law nor its four pending profile successors is adopted or changed. VAL evaluates a complete supplied proposition and gains no policy authorship.

The profile is occurrence-level with one named consumer and a generation-bound scalar/broadcast product; failure propagation is not presented as a new aggregate eligibility profile. Requested, effective, observation-resolved, realized, publication, and evaluation identities remain one-way linked. No application architecture, Engine state, runtime interface, persistence schema, generic framework, or sibling repository changes occur. Finding S1 concerns the proposed scientific outcome definitions, not a transfer of ownership.

## Repository/evidence findings: none

The 51-path candidate comprises one dated status addition plus 50 new Stage B paths: five manager records, 43 preserved draft files, and two final renderer logs. The original 20 study files, including the complete approved Stage A packet, are byte-identical to the sole parent. The earlier pending-release wording is explicitly superseded by the later owner-release record and does not require editing approved input bytes.

The release record binds the already-given exact owner approval, records `fork_turns="none"`, `gpt-5.6-sol`, `ultra`, and the isolated 11-file non-Git input directory. The author integrity/read record agrees with that firewall. Technical layout/render instructions and the two recorded completeness checks introduce no new scientific reference. I found no evidence of an input leak. This assessment uses durable dispatch/author attestations and exact artifact equality; it is not a reconstruction of the author's full execution transcript.

## Gate reproduction and environment

Checks ran with `/Users/gwilson/tolteca/bin/python -B`, Python 3.13.2, macOS 26.6.2 arm64, read-only Git commands, local pypdf/Pillow, and bundled Poppler. Independent Python assertions were executed directly in shell heredocs and wrote no tracked JSON. The manager scratch checker was read, not executed.

| Focused check | Independently reproduced result |
| --- | --- |
| Candidate, parent, canonical identity and clean state | PASS; exact SHAs/trees/branch, sole parent, canonical ancestry, staged/unstaged/all-untracked cleanliness |
| Governance effectiveness and bytes | PASS; accepted-object/current/canonical digests and canonical ledger/ancestry agree |
| Scope and predecessor preservation | PASS; 51 changed paths; only status addition plus new Stage B; all 20 predecessor files preserved |
| Released input integrity | PASS; exact manifest digest, ten declared content hashes/byte counts, exactly 11 files |
| Export archive | PASS; SHA-256 `7d586017d28f124de1323e9abcea856ddc83b29d53f29c339e4b34d1aee3f6b5`; all 11 files equal committed packet |
| Draft artifact manifest and source copies | PASS; 42 unique manifested artifacts plus manifest; all 43 match the original sealed author output; all 11 input copies and nine local reference/scope/recovery copies are exact |
| Required study structure | PASS; all six common modules included once through the thin wrapper; both entrypoints use that wrapper; required openings, two PDFs, crosswalks, ledger, and blocked-question record exist |
| Requirement/prediction traceability | PASS; exactly 25 unique sequential requirements and 25 three-column crosswalk rows; 16 predictions and 16 rows; all canonical sources/interpretations inspected |
| Decision register | PASS; seven recovered decided items, fourteen open consequential proposals, one open question, five deferred items; shared register and full ledger agree |
| Local Markdown navigation | PASS; 64 local links resolve, excluding fenced immutable excerpt provenance locators |
| Focused whitespace | PASS; exactly 12 enumerated Markdown hard breaks and one final blank line in each of the ten enumerated files; no other trailing source whitespace; default Git diff check passes for manager sources and status |
| Contract-library layout | PASS; `/Users/gwilson/tolteca/bin/python -B doc/scientific_contracts/verify_layout.py`; this library checker does not inspect the study draft, which was checked separately above |
| PDF integrity and metadata | PASS; exact sealed hashes, 12/11 pages, full version/revision title metadata, version/revision headers on all 23 pages, every requirement/prediction/open-decision ID present, no unresolved-reference placeholder |
| Final logs | PASS; byte-identical author logs and recorded hashes; no final TeX error, undefined reference/control sequence, overfull box, emergency, or fatal entry; 23/16 underfull spacing notices remain |

Artifact identities independently verified:

- Released input manifest: `3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00`.
- Preserved input-review receipt: `c151c8d67fe5115572f818adab6f9928df4b89eae2201b65216aaed4754358d0`.
- Draft manifest: `c15bf5e238dd3776a4cf83a506c6d1c7e3b8172761b1f621ca28cf42e9574124`.
- Scientific PDF, 12 pages: `e46cca90b1bb10c84bb1987bff3582b0fe825f3f8bc58650c7befc7b91f80fcb`.
- Engineering PDF, 11 pages: `a50d4012816adf107ac9ec6978f85102474b27ecf16b4089a6645e2cfcc49b92`.
- Scientific final log: `47273fcee0e06cb24b064a0e0e4c147205bed5b95eb2ab5b7855ccd90405a97e`.
- Engineering final log: `c3af062333f8a2c36ea4f8eaeb34c942af358dd2df93f2cee9a2a8780e0c7d2b`.

## Scope-exclusion verification and limitations

No application build, CTest/config preflight, baseline campaign, observation, Spack/Unity, performance, memory, or application determinism validation was run or required for this documentation-only review. No dependency was installed. No network/Unity access, push, commit, branch operation, integration, activation, or cleanup occurred. The original Stage A verifier was deliberately not applied to this later candidate because its original-parent/17-path gate identifies a different exact subject.

Both complete LaTeX views and all shared source modules were reviewed, with all-page PDF text/metadata/header checks. Independently rendered scientific pages 4 and 8 and engineering pages 3 and 7 at 110 dpi using bundled Poppler in isolated external scratch; inspected these complete relevant pages for the QC evidence, readable tables/equations, and clipping/overlap. Pixel checks confirmed the scientific page-8 header is present, matching its PDF text and page-4 header pixel count. The manager's all-page sealed visual QA is retained evidence; I did not repeat all 23 pages of visual inspection or rebuild the LaTeX. Final cached Tectonic 0.16.9 logs were inspected and hash-verified instead. Nothing here claims byte-reproducible PDF recompilation.

The numerical equations were assessed analytically against the admitted packet; no implementation conformance or observational inference was performed. Open proposals and the actual-owner question remain owner decisions. This first exact candidate and its preserved artifacts should remain intact while the author makes the single bounded consistency repair. Owner scientific review, later independent rendered-view consistency review, freeze, Registry admission, canonical integration, push, and application work remain separate gates.

Only this report and isolated PDF-review rasters were written outside the repositories. The final owned and canonical worktrees remain clean at the exact SHAs above.

## Changed-path inventory

Paths below are relative to the owned repository; `M` is the one status modification and `A` denotes a newly added path. Exact content is bound by the candidate tree and the artifact digests above.

```text
M	doc/REFACTOR_STATUS.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/AUTHOR_INPUT_REVIEW_959bb909.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/MANAGER_CHECK_RESULTS.json
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/MANAGER_QA_AND_COMPLETION.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/OWNER_INPUT_RELEASE_2026-09-05.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/README.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/BLOCKED_DEFINITIONS_AND_QUESTIONS.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/CROSSWALK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/DECISION_REGISTER_CHECK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/DRAFT_ARTIFACT_MANIFEST.sha256
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/INPUT_INTEGRITY_AND_READ_RECORD.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/LINK_AND_COPY_CHECK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/OWNER_DECISION_LEDGER.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PACKAGE_INVENTORY.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PREDICTION_CROSSWALK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PRIOR_WORK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/README.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/RENDER_RECEIPT.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/SCOPE_BRIEF.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/engineering_conformance.pdf
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/MANIFEST.json
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/PRIOR_WORK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/README.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/SCOPE_BRIEF.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/JINC_BOUNDARY.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/MAP_BOUNDARY.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/NOI_BINDING.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/PROCESS_AND_SCOPE.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/PTC_COEFFICIENTS.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/REGISTRY_FRAMEWORK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/inputs/references/VAL_RULES.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/JINC_BOUNDARY.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/MAP_BOUNDARY.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/NOI_BINDING.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/PROCESS_AND_SCOPE.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/PTC_COEFFICIENTS.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/REGISTRY_FRAMEWORK.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/references/VAL_RULES.md
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/scientific_rationale.pdf
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/assumptions.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/definitions.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/edge_cases.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/equations.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/notation.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/requirements.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common_core.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/engineering_conformance.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/scientific_rationale.tex
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/render_evidence/engineering_conformance.log
A	doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/render_evidence/scientific_rationale.log
```
