# Timestream Successor D2 Native Measurement 001

Status: exact implementation candidate independently reviewed, owner accepted,
pushed, and passed its authoritative Unity GCC13 environment gate; this
documentation-only closure candidate remains pending independent exact-SHA
review and owner disposition; not canonically integrated, activated, or used
for representative science

Work-order identity: `TIMESTREAM-SUCCESSOR-D2-NATIVE-MEASUREMENT-001`

Owner: Citlali project owner

## Authorization and preflight

The owner approved one Timestream Successor spine increment based literally on
canonical `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538` and named branch
`codex/timestream-successor-d2-native-measurement`.

This milestone admits a native-axis D2 measurement carrier. It does not admit
or execute a substantive RTC operator. The application route remains inactive.

The owner further established these controlling validation boundaries:

1. VAL remains the sole semantic authority for sample and coordinate
   validation. D2 performs no independent scientific validation and owns no
   validation policy.
2. D2 owns derived residual `x/r` storage and mechanical presence/completeness
   state only.
3. Publication receives the applicable immutable `ValSnapshot` explicitly in
   the same input that supplies the residual payload. It never retrieves a
   current or ambient VAL state after residual realization.
4. Every residual realization binds its exact parent and realization-time VAL
   snapshot. Publication requires exact snapshot-handle equality. That equality
   is an in-memory admission invariant only, not durable cross-process identity.
5. Until VAL provides typed coordinate targeting, D2 exposes no local `x/r`
   validity, cause, or usability mask and no consumer may infer one from the
   bound snapshot through this product.
6. Source-mask and line-operator evidence remain separately bound processing
   evidence. D2 does not translate either into sample invalidity.

Risk tier: Tier 2. This adds a public typed product and publication boundary,
but no new scientific algorithm, route selection, or persistent product.

Applicable authority:

- `doc/governance/ENGINEERING_GOVERNANCE.md`;
- `doc/governance/TIMESTREAM_SUCCESSOR_GOVERNANCE.md`;
- `doc/governance/REVIEW_AND_CONFORMANCE.md`;
- `doc/ARCHITECTURE.md` and `doc/SCIENTIFIC_CONVENTIONS.md`;
- canonical Paired-D1, VAL, AST/ALIGN, and identity-route state through the
  exact base; and
- the owner's approved work order and subsequent VAL-ownership corrections.

## Exact implementation and disposition

The bounded implementation has these immutable identities:

- canonical base and sole parent:
  `4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538`;
- implementation candidate:
  `bb060947175523d6fc6a777ae4ad4606693e9e5f`;
- candidate tree: `87c9e3aee52e5e4119a484ba46dcd6fe493216b2`;
- branch: `codex/timestream-successor-d2-native-measurement`; and
- historical design source, not imported:
  `916fa07600cf6c5e9ea7317a396fdce160a6c419`.

Independent fresh-context, read-only Tier 2 review of exact candidate
`bb0609471...` returned PASS with no scientific or behavioral findings, no
architecture, ownership, or anti-bloat findings, and no repository or evidence
findings. The owner accepted that exact implementation and its review, then
reported the feature branch pushed. A read-only local-Mac `git ls-remote`
verification during closure preparation resolved the live GitHub feature ref
exactly to `bb060947175523d6fc6a777ae4ad4606693e9e5f`.

That acceptance and push apply only to the bounded implementation. They do not
authorize this later closure-record commit, canonical integration, activation,
production use, or a subsequent Timestream Successor increment.

## Bounded implementation

The candidate adds one isolated public header containing:

- route/profile identity and descriptive residual-realization identity;
- move-only residual `x/r` payloads with explicit absent or
  present-structurally-complete state;
- one per-network residual realization bound to the exact immutable Paired-D1
  parent, network-native detector/occurrence axes, contiguous-run identity,
  route/profile, realization identity, and `ValSnapshot` handle;
- zero-copy access to the parent's prefilter `x/r` matrices;
- separately retained, exact-parent source-mask and line-operator processing
  evidence;
- one explicit publication input containing the residual payload collection
  and applicable immutable `ValSnapshot` together;
- fail-closed structural admission for complete network inventories, exact
  parent, route/profile, network order, native axes, support shape,
  residual-realization identity, and exact snapshot handle; and
- memory evidence distinguishing owned residual numerics, identity text,
  referenced processing evidence, and referenced parent/snapshot/axis handles.

Residual values are not required to be finite. A non-finite residual remains a
mechanically present value whose semantic validation belongs to VAL. The
compile-only contract checks that the D2 product offers no `valid`, `x_usable`,
`r_usable`, or `causes` interface.

Line evidence only verifies a mechanically complete, deterministic description
of an already-applied operator: finite non-negative ordered frequency
intervals, unique line identity, non-overlap, a nonempty operator-evidence
identity, and an explicit statement that the operation was effective before
any later decimation. It does not design or execute a filter and has no
validity consequence.

Source-mask evidence calls its bits `excluded_from_processing`. The bits cover
the exact network-native cells when the mask was applied, or are absent for an
owner-approved not-applicable disposition. They are not sample-invalidity
bits.

Implementation candidate paths and their exact-candidate SHA-256 content
digests are:

- `c4426d5bda7b020aebdf27443446510d1ec1d38ed5eebf57a924cce2338431b7`
  — `include/citlali/core/pipeline/timestream_d2_native_measurement.h`;
- `a86d20350a790a9150b4bb3fc710743cd2a085ac6996952eadb1fed9702826fd`
  — `tests/timestream_d2_native_measurement_header.cpp`;
- `e60c113f83d6730455438190d5956c78d8d39af2d892b69b6e6a1813a7047680`
  — `tests/test_timestream_d2_native_measurement.cpp`; and
- `698ed1f1f2ac56b21517fd5b2f7785bae76736b31910395bb47ec3f49371c0e8`
  — `tests/CMakeLists.txt`.

The original pre-disposition form of this handoff at exact candidate
`bb0609471...` has SHA-256 content digest
`2513d764eea8aeccfec091fe0727c13a98aac572c0dfd7e91a78776b680f1aba`.
This later closure candidate changes only that handoff path and does not rewrite
the accepted implementation.

No application, `Engine`, CLI, YAML, MAP, CAL, PTC, AST, identity-route, VAL,
Paired-D1, scientific-contract, or dependency source is changed.

## Historical-source disposition

Historical commit `916fa07600cf6c5e9ea7317a396fdce160a6c419` was inspected as
design evidence only. It was not cherry-picked and no divergent historical
control/status stack was imported.

Retained concepts, adapted to the canonical architecture:

- residual `x/r` ownership;
- network-native sampling and detector-grid identity;
- descriptive cleaning-realization provenance;
- source-mask and line-operator evidence; and
- compact memory-accounting evidence.

Rejected historical concepts:

- D2-local residual validity or cause planes;
- D2-local `valid()` or coordinate-usability interpretation;
- treating non-finite residuals as a D2 validation decision;
- any common-grid, downsampling, filtering, factor-selection, RTC execution,
  route activation, or application wiring; and
- historical WP-7 control or status material.

## Validation

Pre-commit local validation of the bounded source passed:

- isolated header plus behavioral target built successfully;
- all 7 focused D2 tests passed;
- all 891 runnable CTest tests passed with zero failures; the established
  `citlali::MapFitterLifecycle.ExactProductSequence` remained disabled;
- configuration preflight passed 130/130, all four mode kits, 8/8 compact
  compatibility cases, 100% surface coverage, and every authority/boundary
  audit;
- baseline tools passed 207/207;
- the validation ledger reported 60 valid records; and
- the science-change ledger reported 3 valid changes and 5 valid integration
  commits.

The local C++ result is supplemental fallback evidence only. It uses the
available AppleClang/Homebrew dependency realization, and its CLI provenance
reports the pre-existing `kids 04088da-dirty` limitation. It is not Unity
GCC13/Spack evidence and makes no representative-science claim.

### Authoritative Unity GCC13 environment gate

The owner executed the pre-canonical environment gate at exact source
`bb060947175523d6fc6a777ae4ad4606693e9e5f` as Slurm job `63987273` on
2026-09-04. The job ran in partition `toltec-cpu` on `toltec-cpu001`, requested
6 CPUs and 64 GiB, and completed successfully with exit status `0:0` in
9 minutes 9 seconds. Its batch step reported peak RSS `22645480K`, approximately
21.6 GiB.

The build used:

- the accepted `unity-gcc13` Spack environment;
- cluster Python 3.12.3 at `/usr/bin/python`;
- six build jobs on six available CPUs;
- a job-local ext4 build and compiler-temporary tree under `/tmp`; and
- the Spack GCC compiler wrappers recorded in the accepted profile and CMake
  cache.

Stage results were:

- configure: PASS in 13 seconds;
- full default build: PASS in 8 minutes 4 seconds;
- explicit `citlali_safety_test` build: PASS in 5 seconds; and
- `check`: PASS in 46 seconds, including 33.29 seconds of CTest execution.

All 891 runnable tests passed. The only non-run test among the 892 registered
tests was the established disabled
`citlali::MapFitterLifecycle.ExactProductSequence`. Slurm stderr was empty, the
source remained detached and clean at the exact candidate before and after the
run, and all nine durable evidence files verified against their recorded
checksums.

The authoritative evidence root is:

`/work/toltec/citlali_dev/citlali_acceptance_evidence/TIMESTREAM-SUCCESSOR-D2-NATIVE-MEASUREMENT-001/bb060947175523d6fc6a777ae4ad4606693e9e5f/unity-gcc13-sbatch-63987273`

Its `final-status.txt` has SHA-256
`c36ea6953350875fc7adb28e4f06870061d6e64e3f827355a94d78cc6d699a28`.
Its `SHA256SUMS` has SHA-256
`01158279b8dd62a88f2f808c9a5b737b5d0a0bb1a65463f1375254f957464dbe`.

This was a clean build and complete CTest gate, not representative science or
an affected-mode reduction. No observational execution was required because
the increment adds an unactivated in-memory carrier without changing an
executable route or numerical operator.

### Build-environment reassessment

The owner reported that an earlier interactive attempt built on the shared NFS
`/work` filesystem with 8 compile jobs inside a 6-CPU allocation. It exhibited
long-lived existing test translation units, blocked tasks, disproportionate
system time, I/O wait, and eventual Slurm time-limit/OOM termination. The owner
also reported that a first attempt to move the build to local scratch exited
before configure because the command checked a build path before creating it.
Neither attempt completed the gate, and neither is acceptance evidence.

The owner's typical interactive `srun_bash` environment requests 16 GiB. The
authoritative six-job run later reported approximately 21.6 GiB peak RSS, so
16 GiB is insufficient for this clean full build. The successful run shows that
the exact D2 source builds under the corrected conditions; because memory,
concurrency, and storage placement changed together, it does not identify which
earlier condition caused the delay. The retained operational conclusion is to
use a resource-matched batch job and job-local build tree for long clean Unity
gates.

The unexpected build behavior triggered the Tier 2 environment/performance
reassessment. Its disposition is **continue after an in-scope gate-method
correction**: no source repair or scope expansion was needed.

## Completion and canonical sequencing

Scientific and behavioral conformance: PASS. The candidate implements the
approved mechanical D2 carrier and preserves VAL as the sole semantic
validation authority. It introduces no numerical operator or scientific-policy
change.

Architectural conformity and ownership: PASS. D2 owns residual payload and
mechanical completeness only, binds the explicit immutable VAL-authored
snapshot, and adds no `Engine`, orchestration, route, or speculative-framework
growth.

Repository, branch, and evidence hygiene: PASS for the implementation subject.
The exact branch, base, candidate, tree, path digests, review, owner acceptance,
remote feature ref, local and Unity environments, and durable evidence
identities are bound above.

During closure preparation, local `refs/heads/codex/refactor-mainline` had
independently advanced through map-space contract work to
`9f42d348298d76c5d5145aaf0c3eace1f3e154c1`, while the local cached
`refs/remotes/origin/codex/refactor-mainline` remained at the D2 base
`4d14d0dce8c80b6bc9d0d39c9a90a8f4b2504538`. During independent review on
2026-09-04, a read-only live `git ls-remote` check confirmed that GitHub
`refs/heads/codex/refactor-mainline` had also advanced to exact
`9f42d348298d76c5d5145aaf0c3eace1f3e154c1`. This moving-base trigger does not
invalidate the literally based D2 candidate, but it prevents treating later
canonical admission as the originally available simple fast-forward. This
closure therefore does not modify the older feature branch's
`doc/REFACTOR_STATUS.md`. Any canonical admission must be a separate,
owner-approved integration candidate constructed against the then-confirmed
canonical authority and independently reviewed at its exact SHA.

## Explicit exclusions and stop boundary

This candidate adds no filtering, filter design, factor selection,
downsampling, resampling, RTC or PTC processing, common-grid projection, AST
processing, route activation, ordinary-route change, persistent serialization,
MAP action, CAL action, generic framework, or representative-science claim.

Typed coordinate targeting and operator-specific relevance/invalidation policy
remain deferred VAL contract questions. They are not repaired or anticipated
inside D2. If future publication requires an immutable binding that the
admitted VAL interface cannot provide, work must stop at that dependency rather
than introduce D2-local validation semantics.

The implementation increment is complete, but this documentation-only closure
candidate must stop for independent fresh-context exact-SHA review and owner
disposition. Its containing commit identity and final clean-worktree state are
reported externally after commit so this record does not make a circular claim
about its own SHA. No closure-record push, canonical integration, route
activation, production use, cleanup, or subsequent Timestream Successor
implementation is authorized by this record.
