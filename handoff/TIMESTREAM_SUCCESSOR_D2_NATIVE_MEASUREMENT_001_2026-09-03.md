# Timestream Successor D2 Native Measurement 001

Status: bounded implementation candidate assembled locally; pending an
independent fresh-context exact-SHA review and owner disposition; not pushed,
integrated, activated, or used for representative science

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

Expected candidate paths:

- `include/citlali/core/pipeline/timestream_d2_native_measurement.h`;
- `tests/timestream_d2_native_measurement_header.cpp`;
- `tests/test_timestream_d2_native_measurement.cpp`;
- `tests/CMakeLists.txt`; and
- this record.

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

After the candidate has one exact full SHA and passes independent review, the
pre-canonical environment gate remains an owner-authorized, user-executed clean
Unity GCC13 full build and CTest run. No representative science execution is
authorized by this work order.

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

The candidate must stop for independent fresh-context exact-SHA review and
owner disposition. No push, canonical integration, route activation, Unity
execution, or subsequent Timestream Successor implementation is authorized by
this record.
