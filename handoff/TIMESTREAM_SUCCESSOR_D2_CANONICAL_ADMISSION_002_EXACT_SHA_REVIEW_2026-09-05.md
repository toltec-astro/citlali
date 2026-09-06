# Independent exact-SHA review

The following preserves the final report returned by the fresh-context,
read-only reviewer `/root/d2_admission_002_exact_review` on 2026-09-05.
The writer finalized the task-local evidence manifest after this report;
its manifest-pending statement describes review-time state.

**Independent exact-SHA verdict: PASS WITH RECORDED LIMITATIONS.** No blocker, major, or minor findings in this admission candidate. It is suitable for the owner's canonical-admission disposition; this review does not authorize canonical movement or push.

Reviewer: fresh-context read-only agent `/root/d2_admission_002_exact_review`.

| Identity | Independently verified value |
| --- | --- |
| Candidate | `aeea0eef04ec70d8142c8a20fd7b09dfb64725ed` |
| Tree | `751f274f57b13883eadb02ab546105601e1f66dc` |
| First parent: canonical | `00b974c9039d4c3025dcce18f26bca69d36af9c3` |
| Second parent: validated repair | `7d57a5acf893ae0f34c3639499484b3f5976768a` |
| Parents' common ancestor | `ae953ed4d87d1f693d2bbf42aebbc25ef730c771` |
| Validated repair tree | `c3c545c826c8e8c948c1d53fbf8301b52d68febe` |

The reviewed worktree is [citlali-timestream-successor-d2-canonical-admission-002](/private/tmp/citlali-timestream-successor-d2-canonical-admission-002), on `codex/timestream-successor-d2-canonical-admission-002`. It was clean at the beginning and end of review.

I read the applicable `AGENTS.md`, TolTEC context skill and relevant routing references, all three engineering/governance/review documents, current status and integration-ledger entries, the new admission handoff, relevant earlier D2 records, architecture and scientific-convention boundaries, ADR 0017 and the scientific-authority router, prior repair review/gate evidence, the v2 runner and its independent review, and the complete owner-supplied Unity receipt.

The normative governance bytes and digests match accepted `06a3ade51c1b3f38887295433d913811bf25cd14`; both it and effectiveness record `77507836325eff9f469062d5884481ea37599594` are on candidate ancestry. Historical candidate banners do not negate that recorded effectiveness.

| Review axis | Disposition |
| --- | --- |
| Scientific and behavioral conformance | **PASS WITH RECORDED LIMITATIONS; no findings** |
| Architectural conformity and ownership | **PASS; no findings** |
| Repository, branch, and evidence hygiene | **PASS WITH RECORDED LIMITATIONS; no findings** |

The bounded work is documentation reconciliation around the previously reviewed repair. Relative to canonical, the exact ten changed paths comprise the status and ledger, four D2 handoffs, the D2 public header, behavioral test, isolated-header test, and CMake registration. The complete inventory and all ten content digests in [CANDIDATE_PRESERVATION.json](/private/tmp/citlali-d2-canonical-admission-002-preflight-2026-09-05/CANDIDATE_PRESERVATION.json) were independently checked against Git objects and the clean working files. Newly authored content is limited to the [admission 002 handoff](/private/tmp/citlali-timestream-successor-d2-canonical-admission-002/handoff/TIMESTREAM_SUCCESSOR_D2_CANONICAL_ADMISSION_002_2026-09-05.md), status, and ledger.

Scientific and behavioral preservation is established by complete tracked-input equivalence: **all 1,600 paths outside `doc/` and `handoff/` have identical Git modes and objects to `7d57a5acf...`**. All seven prior non-status D2 paths—including the three earlier handoffs and all four source/build/test paths—are preserved. The input-manifest SHA-256 is:

`815524e7375a5cb0e9afc9a2ea852d0f81131792c967b78c09ea6dca75fd09cc`

Inspection confirms that the inherited repair rejects nonfinite low/high line-frequency metadata before sorting, retaining the 193 malformed-input cases. The comparator and subsequent finite metadata checks remain unchanged. The prior exact-source review independently reproduced failure against the inherited header and success against the repaired header; this admission changes neither subject's source.

D2 continues to own mechanical residual storage and completeness; VAL retains semantic sample/coordinate validation. Nonfinite residual `x/r` storage, explicit immutable snapshot binding, native axes, and separately retained processing evidence remain intact. No interface, lifecycle owner, scientific policy, `Engine` state, numerical operation, allocation strategy, or application wiring changes here.

Repository reconciliation is correct. Canonical's intervening **21 paths are documentation-only**; its **19 non-overlap documentation paths** are byte-identical in the candidate. The complete earlier D2 status/ledger bodies and canonical MAP acceptance additions survive. The candidate differs by 10 paths from canonical and 22 from the repair, with exactly the expected union. The original implementation, both closure commits, literal implementation base, prior admission candidate, and repaired source remain reachable; their retained refs are unchanged.

I reproduced these checks:

- Reviewed and ran `verify_admission_tree.py` in committed, read-only mode against the full candidate SHA: **PASS**.
- Independently checked ordered parents, common ancestor, complete input equality, preservation, ancestry, governance digests, all ten changed-content digests, status histories, and new handoff links: **PASS**.
- Ran the unchanged MAP acceptance verifier only in its clean original `00b974c9...` worktree: **PASS**, including 13 accepted packet files, 48 sources, 17 products, 32 edges, 16 traces, and preserved owner-decision dispositions.
- Ran scientific-contract layout verification at the admission candidate: **PASS**.
- Checked whitespace against both parents and final clean state: **PASS**.
- Checked v2 runner shell syntax and its SHA-256 against the receipt: **PASS**.
- Checked receipt source/tree, pinned dependency revisions, archive inventory count, and recorded runner/final-status/manifest identities for consistency: **PASS**.

The complete owner receipt records Unity job **64018093 PASS** at `7d57a5acf...`: **892/892 runnable tests**, 893 registered, **8/8 focused D2**, the established disabled `MapFitterLifecycle.ExactProductSequence`, matching clean CLI/source identity, clean pinned dependencies, 21 archive checksum successes, empty Slurm stderr, and Slurm `COMPLETED 0:0`. Its runner digest matches the independently reviewed local v2 artifact:

`0b65dff43288c6490a945251c14089409f27d8c4d8baf8341c501f4ee37478d0`

The recorded final-status and archive-manifest digests are respectively:

`c73eaf529296dfecdfea0e5013753b415d3453a78334a99596f939e915f72139`

`d7bcf27bc7134dbac254be707c958d9d17634950e7c6bf50076f702d39b12b8b`

Failed job 64008057 remains correctly classified as a runner-path failure before tests. The v2 correction changes the external executable locator and adds an executable guard; it does not modify application source or acceptance thresholds.

Evidence limitations are explicit:

- Unity outcomes and remote archive hashes are owner-supplied observations. I did not access Unity, download its archive, recompute remote checksums, or inspect its complete logs.
- Compiled Spack DAG/GNU 13.3.0 provenance establishes the source-build/test environment. Runtime deployment labels remain `unmanaged`/`unavailable`; managed-deployment or installed-artifact acceptance is not established.
- Local AppleClang/Homebrew/cached-dependency gates remain supplemental, including the disclosed dirty cached-Kids limitation. I inspected their existing review and gate records rather than rerunning broad builds/tests.
- Unity evidence remains attached to `7d57a5acf...`. Its reuse is supported by complete tracked-input equivalence; this does not claim execution of the merge SHA or identical generated provenance/executable bytes.
- No representative-science reduction is triggered by this documentation-only, unactivated admission.
- The existing task-local `SHA256SUMS` predates the complete receipt and final candidate; it is not a finalized closure manifest. `STAGED_DRAFT_PROOF.json` remains an earlier checkpoint. The writer must finalize the external evidence package after preserving this review.
- I verified local refs. Live GitHub equality was reported by the writer's contemporaneous check; it must be rechecked before any later authorized admission.

No further source change or Unity run is warranted solely by this documentation merge. Canonical remains `00b974c9...`; owner admission, push, activation, production, cleanup, and subsequent implementation remain separate dispositions. No files, branches, commits, or refs were changed during this review.
