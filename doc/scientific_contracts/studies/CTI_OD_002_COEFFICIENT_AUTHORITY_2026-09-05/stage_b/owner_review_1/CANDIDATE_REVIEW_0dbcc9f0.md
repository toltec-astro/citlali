# Independent exact-SHA Tier 2 review — uniform coefficient r0.3

Review date: 2026-09-06. Reviewer: independent fresh-context read-only reviewer, not the author or manager. **Verdict: repair required.** One bounded scientific/behavioral wording finding remains. This verdict concerns readiness to return this exact draft for renewed scientific-owner review, not adoption of the scientific proposal.

## Exact subject and preflight

| Identity | Reviewed value |
| --- | --- |
| Candidate | `0dbcc9f0527d73fb873e29edaefa35095636ddc7` |
| Tree | `9025475cb0a01a5285cd52018a9aa784d055382c` |
| Sole parent | `448194ac3e650a12e5a3862d86a4a05f867a90c5` |
| Parent tree | `791dcef1fcf0ca148f4d3cdfb9899fec1fe31ab9` |
| Branch | `codex/cti-od-002-coefficient-authority-2026-09-05` |
| Owned worktree | `/private/tmp/citlali-cti-od-002-coefficient-authority-2026-09-05` |
| Initial and final state | Clean; no staged, unstaged, or untracked paths |
| Original canonical | `00b974c9039d4c3025dcce18f26bca69d36af9c3` |
| Current canonical checkpoint | `5244e04638db5aa92180feab52acd782229cfa32` |
| Original released input | `959bb90947d4b9651fbba9f16a0eb73dde893b84`; eleven-file manifest SHA-256 `3140f1769ce1d4ed970c21f8cb56ff50e697880d5676983314d7b0584afd8f00` |
| Returned draft manifest | SHA-256 `87eafdbaed29dc8d810a5bba9d4ba818d57a837c04337a4c71c31b06215f0ee9`; 47 entries plus manifest = 48 files |
| Owner supplement manifest | SHA-256 `3732fe857290645e52b6b1a37ec9e161eb46de5c89094380425e0983c29400ea`; three content files plus manifest |

The bounded work is the owner's first substantive scientific review return of the uniform coefficient proposal, revised from r0.2 to proposed **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1/r0.3**. Scope includes the five main corrections, three smaller corrections, distinct document audiences, companion evidence records, and one dated status entry. It consumes no application implementation slot. I used the explicitly owned temporary worktree, never the default desktop checkout, and wrote only this external report and task-specific review scratch.

Governance preflight was completed before scientific assessment: `AGENTS.md`, `doc/governance/ENGINEERING_GOVERNANCE.md`, `doc/governance/REVIEW_AND_CONFORMANCE.md`, the scientific-contract charter, the exact owner-round preflight and verbatim assessment, and the original admitted Scope Brief were read. Effective governance is the canonically incorporated owner-accepted commit `06a3ade51c1b3f38887295433d913811bf25cd14`, not the candidate-looking document headers. I independently verified its ancestry and the canonical incorporation record. The exact Engineering Governance digest is `70769787ce2ef4b7323cd2a38e221ade4af3310e0ad6b7b682e08cb4e4d61e76`; Review and Conformance is `691e6d6250102ef2f4a504397581ee67c5707d898ab20fb8dd9e874c47f99bb1`. Those bytes match the accepted governance identity, original/current canonical checkpoints, and candidate. This is scientific-library work, not a WP-7 application unit.

## Three separate dispositions

| Review category | Disposition |
| --- | --- |
| Scientific and behavioral conformance | **Repair required — one major, narrowly bounded wording finding (F01).** The exact-unity mathematics and other detailed owner-response rules are coherent. The failure-scope summary still permits two different outcomes for a bound corrupt payload. |
| Architectural conformity and ownership | **Pass for document-only draft scope.** Exact compact references and shared facts are permitted without mandatory repeated storage; required publication completion remains atomic at the proposed bounded unit; PTC, MAP, JINC, NOI and VAL retain their distinct responsibilities. No runtime implementation conformity is asserted. |
| Repository, branch and evidence hygiene | **Pass with the recorded independent-rebuild limitation.** Exact subject, clean state, 41 changed paths, preservation, manifests, source/excerpt identities, traceability inventories, committed PDF identities and independent page rendering pass. An independent cached TeX compile was attempted but stopped before TeX because of a local macOS library panic. |

These categories do not substitute for one another. Passing provenance does not repair F01 or adopt the proposed science.

## F01 — distinguish publication-binding failure from bound-payload corruption in the failure-scope table

**Severity/category:** major scientific/behavioral wording ambiguity, bounded repair; not an objection to the coefficient mathematics or permission policy.

**Evidence:** `stage_b/draft/src/common/assumptions.tex:118–123`, rendered engineering §8.2, page 9, groups “payload publication” with shared objects that are “missing, conflicting, corrupt, or misbound” and prescribes “Applicability unknown or decision unavailable; no UQ-05 evaluation or fallback.” In contrast, `src/common/definitions.tex:221–228`, rendered engineering §3.3, page 4, explicitly assigns authoritative payload digest/integrity failure, corruption, truncation and decoder-invalid bytes to `Q_integrity = F`, including corruption making the scalar unreadable. The immediately following failure-table fidelity row (`assumptions.tex:130–136`) and UQ-05 require producer-invalid/ineligible for that independent false. Science §6, page 5, also states authoritative payload corruption makes the payload invalid.

**Discriminating case:** an exact request and all authoritative identity/parent/support/publication bindings are valid; an authoritative check establishes that this bound payload's bytes are corrupt. There is no identity ambiguity. The detailed fidelity rule yields `Q_integrity=F`, UQ-05 F, producer-invalid payload, and a completed realized/ineligible handoff. The literal “corrupt ... payload publication” summary row instead permits structural failure and explicitly suppresses UQ-05. Missing payload bytes with otherwise valid bindings has the analogous risk of being moved from an independent U to structural failure, thereby hiding a separate decisive false.

The §3.2 gate prose is more precise: it calls for payload-publication **identities**. That supports the likely intended interpretation, but §8.2 does not restrict “payload publication” to its identity/binding record or exclude its bytes. Readers should not have to choose which of these explicit outcome rows governs. This is consequential decision semantics, not style preference.

**Affected authority:** owner review round 1 points 1–3 require consistent payload/permission states, mixed-evidence composition and separation of question formation from predicate failure. The admitted VAL General Structural Gate and Restrictions/Disposition clauses distinguish failure to establish identity from a false or unresolved predicate after establishment. The draft's UFC-REQ-013, 014 and 017 require those distinctions and cause preservation.

**Consequence:** a conforming implementer could suppress a required fidelity assessment and record structural decision-unavailable for the same established corruption that another implementer records as producer-invalid/ineligible. Both readings remain fail-closed for numerical use, but they differ scientifically in cause, producer validity, applicability and the ability of an independent false to dominate unknown evidence.

**Smallest bounded repair:** narrow the third failure-scope row to the authoritative identity/binding records needed to establish the question, including the **payload-publication identity/binding record**. Explicitly exclude bound-payload bytes, decoder/integrity results and stored membership/reference evidence from that structural row; those remain independently T/F/U/C under UQ-05 after binding. Preserve the existing shared scope for an actual corrupt or ambiguous authoritative binding record. No new scientific policy, source, coefficient family, algorithm or publication rule is needed. Check the corrected row against an authoritative binding-record defect, a bound corrupt payload, and bound unavailable bytes plus an independent false. Re-render the affected engineering view, refresh its exact manifest/receipts and any changed companion claim, commit coherently, and obtain a fresh review bound to the new exact candidate. This verdict does not transfer.

## Owner-response and scientific assessment

| Owner area | Independent assessment |
| --- | --- |
| OR1-01: payload versus permission | Detailed UQ-05/UQ-06 and permission rows correctly distinguish producer invalidity, authoritative named denial/nonauthorization, unknown evidence and same-fact conflict. MAP and JINC remain independent. F01 prevents unconditional completion of the summary-consistency check. |
| OR1-02: mixed evidence | Four separately assessed scalar, unit, integrity and membership facts have a disjoint aggregate rule: independent F, otherwise C, otherwise U, otherwise all T. Same-fact contradictory claims are one C, not a selectable negative branch. MIX-001–004 cover the requested combinations. The finite four-fact domain partitions into 175 F, 65 C, 15 U and one T combination; normalized eligibility agrees with admitted VAL false-dominates-unresolved conjunction after structural establishment. F01 concerns routing into this otherwise coherent rule. |
| OR1-03: request and scope | Explicit named-profile request is separate from selected-family value. Missing selection-record identity is structural; a bound wrong/unknown/conflicting value reaches UQ-01. Wrong q,g pairing in one attempted request stays local, while an authoritative misbound shared q-to-g object affects all references. No unrequested empty-pass profile is created. F01 is the remaining structural/predicate boundary ambiguity. |
| OR1-04: compact and bounded publication | Target `R_q` and stored `R-hat_g` are distinct. Exact independent identity/source/generation/scope/digest comparison can prove compact association without duplicate member lists. Enumerated storage requires every target exactly once. Shared immutable evaluations retain logical occurrence membership/request association; no per-sample coefficient or decision array is mandated. A q is explicitly a complete segment, scan or bounded within-segment/scan chunk; all its required components/links must be durable before handoff, with no barrier from unrelated observation units. Analytic unity does not require summing an array of ones. These remain owner-review proposals. |
| OR1-05: rank provenance | Exact frozen PTC requirements 090/094/095 establish the first ordinary configured-rank group-local route and positive integer rank/realizability obligations; 072 establishes required-output completion. Requirements 076/077 already preserve PTC-disabled and absence of a direct CAL map fallback. The revised prediction limits its rank statement to the admitted ordinary route and neither defines nor rejects an independently authorized future identity/zero-rank family. |
| OR1-S1: version versus realization | Changes to scientific family/profile/permission/boundary rules require the corresponding successor; different actual supports/products/requests/evaluations under unchanged rules produce immutable realizations without changing the family rule. Earlier parents and records are preserved. |
| OR1-S2: finalization | Construction records pre-existing inputs and a write-once target; UQ-07 examines those facts before finalization; Z records the final artifact outcome. A durable completed negative/unavailable evaluation can be realized. Failure to finalize invents no completed UQ-07 value or eligibility assertion. This removes the self-evaluation dependency. |
| OR1-S3: notation | a is application generation, q PTC product/publication, g coefficient realization, i detector-time occurrence and p consumer spatial index. JINC and NOI equations define the latter role explicitly. The reported product/spatial-index collision is removed. |
| OR1-A1: audiences | The five-page science rendering explains equal PTC coefficients, exact support, analytic normalization, validity/uncertainty limits, consumer consequences and owner choices. The fourteen-page engineering rendering carries complete formal tables, identities, mixed cases, scopes, lifecycle, requirements and evidence mapping. Both use the same six common modules. The shorter narrative is appropriate to the owner's direction; no stylistic expansion is requested. |

The exact one remains dimensionless, finite and strictly positive, with no numerical parameter or default. Empty support creates no instance, a singleton is valid, outside-support absence is not zero, and support/identity is not inferred from order, cardinality or numerical equality. A faithful coefficient neither restores signal validity nor supplies response, covariance, uncertainty, exposure or optimality. Repeated realizations retain exact identity; numerical equality supplies no borrowing permission.

MAP's independent admission and finite-positive numerical classification remain intact. JINC retains its signed spatial factor and independent permission/classification; equal PTC factors do not imply equal or positive final JINC weights. NOI still requires its frozen lossless rational identity for the complete exact MAP parent, including exact MAP/coefficient generations and representation source. No JINC-hosted NOI route, new PTC NOI-balance family or approval of NOI's four pending profile successors is introduced.

## Original complete-profile and crosswalk obligations

I compared the revised formal profile and companion `PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md` with the admitted PTC/VAL sources, not an earlier candidate's verdict. They preserve named use, actual-owner question, immutable identity/version/source, applicability/object, structural gate, restrictions, exceptions, response/uncertainty roles, truth/missing/conflict behavior, compatibility/supersession, scope and lifecycle. The exact approved PTC common named-use digest `c1fc8370007b65307769fb966c8523251695924aaff84f3e5b4c89b6d3380b8c` is present in formal source, companion record and engineering page 3. Source promotion requires a new exact immutable source binding before activation. `Gamma_rho=none` explicitly prevents shared fact reuse or failure propagation from silently becoming an aggregate eligibility profile. The one-way requested/effective-policy/observation-resolved/selected/realized-published/evaluation lineage and admitted VAL structural scope/influence bindings are explicit. The generic unsupported coefficient-QC placeholder is not activated.

All 25 requirement IDs, 16 prediction IDs and 10 worked-case IDs match their respective companion crosswalks without omissions or duplicate definitions. I checked the **rendered** science narrative, not merely the presence of shared source behind `ifengineeringview`. Science §§2–3.4 explain family meaning, exact support/binding, request/permission/fidelity separation and economical representation; §§4–4.2 explain uncertainty, inherited rank and immutable change; §§5–6 explain owner status and discriminating consequences. In particular the four-axis/finalization/nonexceptionability explanation is actually printed on science page 2 and supports requirements 015–016. The engineering-only full requirement and decision tables are not being represented as pages printed in the science PDF. The detailed cases add implementation-facing discrimination to the stated scientific meaning; they do not introduce another coefficient design. F01 is the identified content mismatch requiring repair.

The 14 UFC-OD proposals and actual-owner question UFC-Q-001 remain open in the common register and ledger. Support in principle is separated from adoption. Recovered constraints are decided, deferred topics stay deferred, and the actual-owner question blocks activation rather than mathematical review. No acceptance is manufactured from the author's completeness claims or from manager/mechanical passes.

## Authority and material read

Scientific reasoning was confined to the original approved eleven-file author input (README, Scope Brief, prior-work synthesis and seven reference covers/excerpts: PROCESS_AND_SCOPE, PTC_COEFFICIENTS, REGISTRY_FRAMEWORK, MAP_BOUNDARY, JINC_BOUNDARY, NOI_BINDING and VAL_RULES), the separate owner supplement, and this candidate's document artifacts. The verbatim owner assessment digest is `1e9074bfa86566a790f7d1a0284817aced321c66222c6899ef0e13a229a223e9`. The supplemental PTC clauses are whole lines 99, 124, 132 and 134 of the exact frozen requirements source, digest `f2047600cc06c234a78aa3ddf6a575abf2f9592b3e3da810491f6db0150fe21c`; the bound freeze record is `8357961a49272adc40e27a8aa9e760e0d01ff2419ae2c88a62c0f93c9f959e66`.

Read for repository/process evidence: the charter and governance documents listed above; the current dated status delta; `stage_b/owner_review_1/README.md`, source/preflight and prior-source recheck records; `stage_b/README.md`; manager QA/completion and check results; the owner response; all six common modules and both entrypoints; both PDFs; all crosswalk, ledger, proposed profile/boundary, blocked-question, package inventory, read/integrity, copy/link, decision-register and render records; original source/extraction manifests. Historical candidate-review reports were not used for scientific reasoning or to transfer a prior verdict. Hash-only retrieval of recovered source objects/excerpts established provenance without supplying excluded implementation/audit content as scientific authority. The TolTEC routing and PDF inspection skills supplied process/tool guidance only.

## Independent focused gates and environment

The independent checker is preserved only in review scratch, at `/private/tmp/citlali-ptc-uniform-review-0dbcc9f0-2026-09-06/check.py`; its receipt is `check-results.json` in the same directory. It does not invoke the manager checker that writes tracked results or the historical Stage A checker with obsolete scope.

| Gate | Independently reproduced result |
| --- | --- |
| Candidate/tree/sole parent/branch/clean status | Pass |
| Changed-path ownership | Pass: 41 paths, only study stage_b and one additive dated status entry |
| Effective governance/canonical advance | Pass: original canonical and accepted governance are ancestors of current checkpoint; relevant governance/frozen PTC bytes agree at all checked identities |
| Preserved original study | Pass: all 20 files at exact released input commit are byte-identical |
| Original packet and copies | Pass: 11 exact files, sizes/digests/inventory and copies; root convenience copies retain admitted bytes |
| Owner supplement and copies | Pass: 4 exact files, sizes/digests/inventory and copies |
| Sealed prior drafts | Pass: r0.1/r0.2 manifest digests and every historical manifested artifact verified directly from their retained exact Git commits |
| Recovery identity/extraction | Pass: 33 recovered source objects, 22 admitted source objects, 61 exact byte excerpts, and four supplemental whole-line clauses |
| Current returned draft | Pass: all 47 manifest entries, exact recursive 48-file inventory, no omitted or extra returned artifact |
| Shared core/IDs/ledger/response | Pass: six modules, 25/16/10 ID sets and crosswalks, 14 open proposals, nine owner-response IDs |
| Local navigation | Pass: 81 relative links in independently selected Stage B documents; historical review reports and verbatim excerpt/fenced source locators excluded. This is a different bounded count from the manager's 97-link selection. |
| PDF identity/pages/metadata/text | Pass: exact digests below, 5 + 14 pages, v/r in title metadata and all page text |
| Final committed TeX logs | No overfull box, TeX error, undefined control/reference, emergency or fatal entry found; underfull spacing notices are not scientific failures |
| Independent visual inspection | Pass: rendered every final page with bundled Poppler at 110 dpi and inspected all 19 images; no clipping, overlap, broken equation or unreadable table found |
| Independent cached source rebuild | **Not reproduced.** `/opt/homebrew/bin/tectonic --only-cached --keep-logs` on an exact scratch source copy stopped before TeX with macOS `system-configuration` NULL-object / `reqwest-internal-sync-runtime` panic, exit 101. No network was requested, no install/escalation was performed and the repository was unchanged. Existing final logs and exact PDFs were inspected; PDF page rendering was independently successful. |

Environment: local macOS 26.6.2 arm64; `/Users/gwilson/tolteca/bin/python -B`, Python 3.13.2, pypdf; cached local Tectonic; bundled `pdftoppm`. All scratch/rendering remained under the task-specific `/private/tmp` review directory; no GUI backend was launched. The failed independent compile is an environmental limitation, not evidence of a TeX defect. An initial scratch manifest-parser assumption about header comments was corrected and the complete final integrity check then passed; no candidate file was changed.

| Reviewed final PDF | Pages | SHA-256 |
| --- | ---: | --- |
| `stage_b/draft/scientific_rationale.pdf` | 5 | `efd265d423929dad071a499d3b81a0354d0a3bda2e21ebf7cc2c8ca5b192c266` |
| `stage_b/draft/engineering_conformance.pdf` | 14 | `f476d16ae997770a7fb9b3c2ebd24937da037fc70354de574b2c4d820e16013c` |

No C++ build, CTest, configuration preflight, application validation, representative Spack/Unity run, observational campaign, performance/memory/determinism benchmark or implementation conformance test was triggered or performed. They cannot validate this draft's owner decisions and no application behavior was touched. The ten cases and sixteen predictions are proposed future evidence obligations, not executed implementation tests. Firewall history is supported by the exact allowed packets and recorded read provenance; this review does not independently reconstruct every author process access.

## Exclusions, retained authority and next gate

Application source/config/tests, frozen package sources and PDFs, the active Registry, canonical/other worktrees and refs, WP-7 implementation, build-environment changes, Unity, network, installs, pushes, integration, activation and cleanup were outside this task. Changed-path and clean-state checks confirm no repository mutation by the reviewer. Exact historical source objects remain retained; nothing was reset, rebased, renamed, deleted or pushed.

F01 can be repaired within the already authorized owner-round revision. Preserve the current exact candidate and receipts, then review the new exact candidate. A future readiness pass permits renewed owner presentation only. All fourteen proposals, the actual-owner binding and CTI-OD-002 remain open. Complete scientific adoption, the charter's later fresh implementation-blind post-owner consistency review, freeze, Registry/profile activation, canonical integration, owner-held push and application implementation require their own governing decisions.

## Exact changed-path inventory

The following inventory is from `git diff --name-status 448194ac3e650a12e5a3862d86a4a05f867a90c5 0dbcc9f0527d73fb873e29edaefa35095636ddc7`. SHA-256 values bind the candidate contents. No deleted or renamed path occurs.

| Change | Repository-relative path | Candidate SHA-256 |
| --- | --- | --- |
| M | doc/REFACTOR_STATUS.md | 022ab92120654b5e8f8de05ec59c193cb739b428e8ee91f643f4f72e95e3a18b |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/MANAGER_CHECK_RESULTS.json | a9fbae9b9be492dd74ede3110a19539963539ae129f96127c9bb5a7ed6a5332e |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/MANAGER_QA_AND_COMPLETION.md | 4e1683922f5c72427ac0bc86956dd3266a56f2b39c9ee2ee25665e18e32283d9 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/README.md | 235c0da62b563eccaf244b942660e9b4beca22c31931fb08e88ea17ec7767ffd |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/BLOCKED_DEFINITIONS_AND_QUESTIONS.md | 1259ed5f0823309693e3c1e8f3ea15320bee2b52157031bec2b7ab9123251af6 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/CROSSWALK.md | a94ed180e5636c14a104e7891344e756b1eea4c693d979c67ed7a2f9937d582c |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/DECISION_REGISTER_CHECK.md | 4f649028a269b4590a853d6509d6e038bda901375d2d59ce9b8ec0dba1aba1ac |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/DRAFT_ARTIFACT_MANIFEST.sha256 | 87eafdbaed29dc8d810a5bba9d4ba818d57a837c04337a4c71c31b06215f0ee9 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/INPUT_INTEGRITY_AND_READ_RECORD.md | 9150a81a688263d17992dac3b4906837cb356de4675f4ea27f4037f5fd5691b4 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/LINK_AND_COPY_CHECK.md | c021f16260dc14f482ebe2e29281c41fa51767c473e760c8a7ca4e8c2b7242a7 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/OWNER_DECISION_LEDGER.md | c6707de6913860668ae853f3ca2fb6e3ed376ead9602679bc2e8cf3757740a93 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/OWNER_REVIEW_1_RESPONSE.md | 7042d940a7ad9876c0a7b7b0818496605a8e89dfd9e88446a976e09e01e33c04 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PACKAGE_INVENTORY.md | 3b81e8d85e0b2375acfd3b2a5df3ec36ac323024f39bdad755acdb0bc674229b |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PREDICTION_CROSSWALK.md | 376c393a75ae1aec52fa85ca3b9ad88903fdfe6d39223479b29b9a47fba92af5 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/PROPOSED_REGISTRY_AND_BOUNDARY_RECORDS.md | 137fd627d104817aae719e4cd7d8b492e9c13ca03f1c15bcb78a4c2cf6205e8f |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/README.md | 528e175c414ef9260cb78d941746df624cdb9f45b433ef3c87f58a8a63d4ae29 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/RENDER_RECEIPT.md | 8d38067bb164757a93e4a8bf0685f991a3fc69d9d0dc402e494f8c850a2d7a68 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/engineering_conformance.pdf | f476d16ae997770a7fb9b3c2ebd24937da037fc70354de574b2c4d820e16013c |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/owner-review-1-input/MANIFEST.json | 3732fe857290645e52b6b1a37ec9e161eb46de5c89094380425e0983c29400ea |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/owner-review-1-input/OWNER_REVIEW_1_VERBATIM.md | 1e9074bfa86566a790f7d1a0284817aced321c66222c6899ef0e13a229a223e9 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/owner-review-1-input/PTC_RANK_AND_PUBLICATION_CLAUSES.md | 9e16b036203e0fbb6d4899220864476be4a6cc3f412df6e2157f693fc066abea |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/owner-review-1-input/README.md | b54e80d86b270ae4c558e80a01bf68679e08b595091f44972f39cd0cb62e62a9 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/scientific_rationale.pdf | efd265d423929dad071a499d3b81a0354d0a3bda2e21ebf7cc2c8ca5b192c266 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/assumptions.tex | 8ec6e02297b3bb9f1bc1dd7ea0cad0c9131d418ad4553120e3dbaa157eab521d |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/definitions.tex | 87f18a4087a3af2344048edd1b8b7a7c1aa9643a02b0c77c56e81f071ab2e8a3 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/edge_cases.tex | 29fadddecd7c99223a1491b2599494744bd35f7c504f08852b775bd6538c86e8 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/equations.tex | 8e84d10a9febf5eb3805604bfce5d41ee15bd2e40110ca2f5294ef56a9c9e73e |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/notation.tex | 479c3dc1a45da1580abe5b9ec278cd4eead23c0465683bc0081d21b8b50681cf |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/common/requirements.tex | dd3b069d8a410b59509c09b6a7c1c47d0109d5dd438a32f2974434ad34b210ce |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/engineering_conformance.tex | 8afb20e0a9044804ae647f036a1f0d6813eecab37f80b2136ddeb9be20834ff8 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/draft/src/scientific_rationale.tex | c174a0e54444246677dccbb0c825b52ff71e252cf3d4008926a0a052b34a5500 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/CANDIDATE_REVIEW_448194ac.md | 402f9569c06b8d6197f51da14533d35efcff6995c42392d6cdc744ddd5522715 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/PRIOR_SOURCE_RECHECK.json | 5c9bea12a1ecfc0546eedcaae1dc62fc2e75bd2231c03e03b34404ee5554477a |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/README.md | 1981859068db663f7da8b613f98a9cd84b35dc7f1c1ea8bbb222135a4d084658 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/SOURCE_AND_PREFLIGHT_CHECK.json | 1b7ffd7e502a3a545e05f9b59cee30148aa14bebfa66e1fc830a946ac782d63b |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/author_supplement/MANIFEST.json | 3732fe857290645e52b6b1a37ec9e161eb46de5c89094380425e0983c29400ea |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/author_supplement/OWNER_REVIEW_1_VERBATIM.md | 1e9074bfa86566a790f7d1a0284817aced321c66222c6899ef0e13a229a223e9 |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/author_supplement/PTC_RANK_AND_PUBLICATION_CLAUSES.md | 9e16b036203e0fbb6d4899220864476be4a6cc3f412df6e2157f693fc066abea |
| A | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/owner_review_1/author_supplement/README.md | b54e80d86b270ae4c558e80a01bf68679e08b595091f44972f39cd0cb62e62a9 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/render_evidence/engineering_conformance.log | 67d2b644cec3382d6da37090425502078d18a5c0d31f649fe3a22c0d1dce3674 |
| M | doc/scientific_contracts/studies/CTI_OD_002_COEFFICIENT_AUTHORITY_2026-09-05/stage_b/render_evidence/scientific_rationale.log | 8d27030161177989b4b3b8796a9f190fc81e78123c5881bb2a14b06623f014a0 |

Review scratch receipt SHA-256: `8a33cd7bde263d306afffcf9b5fd61ec78201fc8ff03bf10bb3cb1c880d56f75`.
Independent checker SHA-256: `926ac127f2724e9bb7f6544bcc4e03e025bc6bed833e7e92c58bf39d50c533d7`.
