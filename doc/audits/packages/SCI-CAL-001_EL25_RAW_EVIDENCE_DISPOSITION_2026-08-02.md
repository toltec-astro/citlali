# SCI-CAL-001 EL25 raw-evidence disposition — 2026-08-02

Record ID: `SCI-CAL-001-AM12-EL25-SALVAGE-001`

Status: coordinator recommendation; raw evidence salvageable; durable
preservation and replacement execution not authorized

Governing policies: `FRAMEWORK-NUM-001` and `FRAMEWORK-SALVAGE-001`

## Decision

The 672 completed full AM grids remain salvageable, individually SHA-bound raw
model evidence and should be preserved. The 1,281 admitted 225-GHz/EL80
scale-search anchor outputs and their 13 complete traces should also be
preserved. Their current state is
`salvageable_raw_pending_preservation_and_admission`: the only live copy found
is a writable 6.4-GiB tree under `/private/tmp`, so it is neither durable nor
immutable. Reuse in a successor result is conditional first on a byte-
preserving, digest-verified durable copy and then on independent sidecar,
warning, and reparse admission before scientific evaluation.

The failed confirmation result remains `invalid_evaluation`. It supplies no
candidate error, ranking, maximum-error result, band-integrated result,
operator adoption, operational domain, production authority, CAL repair, or
re-audit closure. Raw validity and evaluator validity are separate.

No AM execution is authorized by this disposition.

## Frozen provenance

- Audit package: `SCI-CAL-001`.
- Study: `SCI-CAL-001-AM12-EL25-CONFIRMATION-001`.
- Source task branch: `codex/sci-cal-001-atmosphere-operator`.
- Preregistration commit:
  `fe3b3a1f7885334c50337382d97a84121dbe57c0`.
- Failure-result commit:
  `5d1597ca2d18f5e35519f6e62b5a014aea736fad`.
- Repository-relative evidence root:
  `validation/sci_cal_001_atmosphere_operator_2026-08-01`.
- Local source worktree reviewed:
  `/Users/gwilson/.codex/worktrees/cdd5/citlali-refactor`.
- Current volatile, writable cache requiring durable preservation:
  `/private/tmp/sci_cal_001_am12_el25_confirmation_v1_20260802_root`.

Frozen control identities:

- preregistration JSON SHA-256:
  `66c9583d67c3696ac03d1edbd6eade95884dbdc77dd93ef890226594f210da70`;
- protocol SHA-256:
  `68674df82d0212826ad19bfbd1a6399e96f7dfd0fe99de984133c004b63971f1`;
- runner SHA-256:
  `bcc4bc9f59574424e1daab652ab0316f8a694998155d9c3daa246e1e6260fb22`;
- result-schema SHA-256:
  `a28e738970b2a462fd1fb68c78aad552e32cbd396f8f60956f8615e4be2a3965`;
- failure-report SHA-256:
  `4929efe11680c39532b3b9a74bc5d50bc2a9550a935573f55da3813f3ddc2e7a`;
- failure-decision JSON SHA-256:
  `6714994991eecade125eb3fd8e93b7795b41546e640cde3de962bba1cfa29b67`;
- execution-context SHA-256:
  `a867df7b05ea590c498e41932bb1b3f9520e635d2534f7c8fcc539cfd4a12ecf`;
  and
- AM 12.2 executable SHA-256:
  `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`.

Recorded aggregate identities for the currently available cache:

- evidence collection: 3,920 files, 2,249,667,308 bytes, SHA-256
  `25ee2a1b2f793f5273e714dc0094bb7e39ebc76615e1fe424d2d65a95013956d`;
- 1,953 raw outputs: SHA-256
  `e2d91f7b38c68b0c62651da5e391222b04206f49a5629a2344e75b0f70fa8370`;
- 1,953 matched execution sidecars: SHA-256
  `b2dd7b6d2b155795ca302402a8f8ce06241585c17bed5bb4bbb1b0b2f3804184`;
- 13 scale traces: SHA-256
  `775a491c99b3d02bd772d0d366909c360a5877a1ce829b1850bf3e11e5ace976`;
  and
- internal AM cache: 21,637 files in eight shards, SHA-256
  `9141c5fb61f8d6a7265ef6b6fd0d70b8a7ed113d0950a9f787dc62480bda087d`.

## Exact stop and governance failure

At case `q50_q75_trisect_2/LMT_DJF_50`, the scale search produced a 99-entry
trace and the parsed midpoint transmission exactly matched the frozen target.
The runner then reconstructed tau225 through a second binary64 expression and
compared it with a Decimal-to-binary64 construction:

- runtime path: `1.34988558021834626e-01`;
- Decimal-cast path: `1.34988558021834681e-01`;
- difference: `5.55111512312578270e-17`, two ULPs; and
- hidden threshold: `5.0e-17`.

The source-only comparison at frozen runner lines 1926--1927 was absent from
the protocol/register, had no arithmetic or ULP derivation, had no propagated
mapping to the final one-percent extinction-correction metric, and was fully
deterministic from frozen inputs. It therefore had no justified scientific or
invalid-evidence authority and should have been exercised across all 16 cases
before AM. It remains a Class D warning-only engineering consistency
diagnostic. Any future impact analysis that establishes non-warning authority
would require explicit reclassification as Class A, B, or C and satisfaction
of that class's derivation and approval rules.

## Validity matrix

| Layer | Disposition | Basis |
|---|---|---|
| Frozen scientific tuples, candidates, passbands, domain, and one-percent gate | `unchanged` | The incident concerned only a redundant construction-path comparison. |
| 672 full grids from the first 12 cases | `salvageable_raw_pending_preservation_and_admission` | All have matched sidecars and bound hashes; return-code 1 warning-bearing outputs passed the frozen admission structure. The downstream guard fired after their creation. The live cache is still writable and temporary. |
| 1,281 anchor outputs and 13 traces | `salvageable_raw_pending_preservation_and_admission` | Trace-to-anchor matching passed; there are no orphaned anchors; all anchor AM returns were zero. Durable copy and successor admission remain pending. |
| Case-13 scale trace (`q50_q75_trisect_2/LMT_DJF_50`) | `salvageable_raw_pending_preservation_and_admission` | The trace is complete with 99 evaluations; the guard fired after the accepted target match and before full-grid generation. Durable copy and successor admission remain pending. |
| Original evaluator/decision | `invalid_evaluation` | Only 12/16 cases and 672/896 grids existed; the frozen runner stopped before complete metrics. |
| Partial scientific metrics | `not_evaluated` | The failure record affirms no partial candidate errors, ranking, maximum-error inspection, band integration, or observational inference. |

## Permitted successor correction

A separately frozen successor may change scientific/numerical guard semantics
only for the redundant achieved-coordinate binary64 consistency check. It may
remove that check from invalidating authority or retain it as a registered
warning that records both operands and ULP distance. It may also add the
non-scientific framework plumbing required for stable condition IDs, one
dispatcher, model-free preflight, implemented-action reporting, and separate
raw/parser/evaluator validity records while preserving every other existing
guard's semantics. It may not substitute another universal epsilon.

The successor must preserve:

- all 16 scientific cases and their frozen tuple identities;
- every candidate, truth profile, passband authority, scientific domain, and
  elevation grid;
- the exact parsed 225-GHz target-transmission admission;
- AM executable/model identity and model-affecting execution conditions;
- return-code and warning admission, parsing, row completeness, sidecars,
  hashes, and provenance; and
- the one-percent representation-fidelity gate and its final-metric meaning.

The failed commits remain immutable. The current cache must not be treated as
immutable until a byte-preserving durable copy is rehashed and protected. The
successor receives new
protocol, runner, parser/evaluator, schema, result, register, preflight, and
readiness-certificate identities.

## Required salvage admission before metrics

An independent checker, not the corrected evaluator acting as its own
authority, must first verify a byte-preserving durable copy against the
recorded aggregate and then independently reparse and admit:

1. all 672 completed full grids and their 672 sidecars;
2. all 1,281 scale-search anchor outputs and sidecars;
3. all 13 scale traces and every trace-to-anchor binding;
4. execution-context, AM executable, tuple, profile, frequency/elevation,
   return-status, warning, parsed-row, and raw-payload identities; and
5. the statement that no partial decisive scientific metric was inspected.

Any mismatch invalidates only its causally affected artifact, tuple, or shard
unless the checker shows that the defect is collection-wide. A full rerun
requires a concrete model-input, executable, execution-condition,
completeness, warning-admission, parsing, provenance, or independence reason.

## Missing-only computation plan

If the salvage admission, guard preflight, independent review, and readiness
certificate all pass, the later execution plan should compute only:

- three unstarted scale searches for
  `q50_q75_trisect_2/LMT_DJF_75`,
  `q50_q75_trisect_2/LMT_annual_25`, and
  `q50_q75_trisect_2/LMT_MAM_25`; and
- 224 missing full grids: 56 elevations each for the completed-trace
  `q50_q75_trisect_2/LMT_DJF_50` case and those three unstarted cases.

No completed full grid or the completed case-13 scale search should be
repeated unless its independent salvage check fails. New evidence belongs in a
separate digest-bound delta cache. The frozen successor evaluator then processes
the verified union of all 896 grids in one pass; it must not inspect the 672
reused grids as a partial candidate result before the evaluator is frozen.

## Held boundaries

This disposition does not modify Citlali application code or TolTECA and does
not authorize CAL repair, Unity evidence, production, or re-audit. It does not
change the frozen scientific candidates, passband authority, opacity/elevation
domain, or one-percent gate.

The separate composition-closure framework decisions
`FRAMEWORK-COMP-D005` and `FRAMEWORK-COMP-D006` remain held. Nothing in this
salvage record supplies or authorizes their scientific closure.

## Exact next authorization

The next permissible action is owner authorization of a bounded, model-free
successor-protocol and evidence-preservation task. The coordinator may then
copy the current tree byte-for-byte to a named durable location, verify its
complete manifest and aggregate digest, protect the preserved copy, and create
the new register, corrected runner/evaluator design, full-16-case guard
preflight, salvage verifier, delta-computation manifest, and readiness
materials. It may not invoke AM or evaluate candidate metrics.

The recommended local preservation destination is
`/Users/gwilson/work_toltec/local_data/citlali-validation/v1/evidence/sci_cal_001_am12_el25_confirmation_5d1597ca`.
It is on the existing Citlali validation-evidence volume, which had 359 GiB
available during this review; the current cache occupies approximately 6.4
GiB. The owner may name a different durable destination before authorizing the
copy.

Only after independent review and a mechanically passing readiness certificate
may the audit manager request a **separate owner authorization** for the three
scale searches and 224 full grids. Replacement execution remains held now.
