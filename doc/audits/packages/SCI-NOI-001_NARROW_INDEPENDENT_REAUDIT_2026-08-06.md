# SCI-NOI-001 narrow independent re-audit — 2026-08-06

Status: **re-audit complete; verdict `amend`; exact repair not ready for
application integration**. The application repair is conformant for F001,
the bounded current-mode part of F003, and the explicitly enabled F008
lifecycle. F002 and F005 remain open because the repository's active reduction
auditor rejects the exact provenance schema and disabled-state records emitted
by the repaired application.

This is a documentation-only independent re-audit of exact repair commit
`38ef72860743636f59d226c9e1ff5ff776d0e9c0`. It is not a repair, integration,
evidence request, production decision, or authorization to use Unity or run a
reduction.

## Exact entry and authority verification

- Audit worktree:
  `/Users/gwilson/.codex/worktrees/279d/citlali-refactor`.
- Audit branch: `codex/reaudit-sci-noi-001`, created only after local and
  remote branch absence and worktree safety were verified.
- Starting HEAD and exact pushed repair-branch tip:
  `38ef72860743636f59d226c9e1ff5ff776d0e9c0`.
- Exact repair parent:
  `d5015fe716971bf8ea617e8a187311bf5af05185`.
- Pushed coordination authority:
  `ac43fb4fc950d53e54f0b3ec424bd8b3ee0a6cd4`.
- The worktree was clean before inspection, remained clean through all
  read-only inspection and gates, and contained no pre-existing re-audit
  branch.

The frozen entry bytes and all manifest-listed Git objects were independently
hashed:

| Authority | Source | Verified SHA-256 |
| --- | --- | --- |
| re-audit prompt | `ac43fb4:doc/audits/prompts/SCI_NOI_001_REAUDIT_PROMPT.md` | `7448d748043cdcb7f942688766166b1d3a4ed112421fe0997155a3a10723546b` |
| re-audit authority manifest | `ac43fb4:doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REAUDIT_AUTHORITY_MANIFEST_2026-08-06.yaml` | `fbe4cf5c4c783f6bc878ef1037aae7806386294124af1689dd071e04d9abb49e` |
| frozen readiness record | `cfe45c7:doc/audits/packages/SCI-NOI-001_REAUDIT_DISPATCH_READINESS_2026-08-06.md` | `f0b64947246bb15fcf957caeaaff7491b95d35138d7a90b70d8c0b8cf4c4882d` |
| owner decision brief | `c21ff27:doc/audits/packages/SCI-NOI-001_COORDINATOR_DECISION_BRIEF_2026-08-06.md` | `a40b151e161d3d2e5721bf75f6ec126dafae8f275e28ee6efbf0e07f146de471` |
| final audit | `bf6de40:doc/audits/packages/SCI-NOI-001_SCIENTIFIC_CONTRACT_AUDIT.tex` | `eb38c7b282a95f792759ca71721fdb7731db5a4e392f8a95ac208c33754f7517` |
| independent core R3 | `5a027c9:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |
| repair prompt | `4fa99c6:doc/audits/prompts/SCI_NOI_001_REPAIR_PROMPT.md` | `83996bd6ae5ffbf609f14354d237d23465f3ecb1ec6a3d161447b162bfeaa024` |
| repair authority manifest | `4fa99c6:doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml` | `44c6ac459f058ba39e18d01f35e64bfb37bcfec9c116a12a2839f64beac71bae` |

The readiness record at `ac43fb4` hashes to
`5e40c7ba107d1cacd0c0f276c77e89fa381506e830511fd6a297ef7a657b3055`
because that coordination commit adds the live owner Beammap clarification.
The supplied frozen readiness bytes at its parent retain the required
`f0b649...` digest. Both were read with the frozen prompt and manifest left
unchanged.

The live clarification is honored exactly: the optional v4-compatible Beammap
jackknife/noise-map capability is retained; standard Beammap configurations
remain disabled with effective count zero and no realization/product work;
F008 assesses only deterministic lifecycle when explicitly enabled; and a
configured count while disabled is inert, not a requirement or evidence
request.

## Scope and method

Only F001, F002, F003, F005, and F008 were inspected, with F007 treated as held
evidence policy. F004, F006, NOI-002 estimator/consumer/count policy, FLT edge
work, residual FRUIT work, and RTC/PTC/MAP/JINC/FLT/FRUIT algorithm changes were
excluded.

The exact `d5015fe..38ef728` repair diff contains 23 files and 1,386 insertions
with 138 deletions. It is bounded to realization identity, noise admission and
lifecycle, the ordinary/Pointing/Beammap generation paths, compact sidecar and
FITS metadata, required-output handling, focused C++ tests, and the existing
noise-products config-boundary audit. It changes no runtime YAML configuration
and no excluded numerical algorithm. Standard Beammap YAML remains
`enabled: false` with the configured count left inert.

No delegation or subagent was used. No application, test, configuration,
canonical audit record, or original authority byte was modified. No local or
Unity reduction, astronomical evidence, external corpus, integration, push,
or production action occurred.

## Independent implementation assessment

### F001 — deterministic realization identity

The new public header defines versioned key and generator identities and a
canonical namespace containing the master seed, observation identity,
`source_imprinted_current` ensemble mode, conditioning iteration, named pass
and ordinal, coherence policy, channel policy, and channel-randomization mode
(`include/citlali/core/pipeline/noise_realization_identity.h:22-138`). Each sign
then adds realization, observation-scoped coherence-unit ordinal, and stable
observation-scoped channel ordinal to the namespace-derived counter
(`noise_realization_identity.h:148-208`).

Ordinary and Pointing paths construct one context per observation/conditioning
iteration before their scan work and record the completed compact assignment
afterward (`lali_setup_pipeline_impl.h:30-89` and
`pointing_pipeline_impl.h:24-81`). The sign matrix is populated from explicit
indices, not mutable generator draw order
(`timestream_scan_generation.h:20-49`). The focused test varies traversal,
OpenMP dynamic scheduling, observation identity, every key dimension, and
channel-shape growth. Static tracing and repeated tests support closure.

### F002 — compact provenance and persisted joins

The application-side design is compact and reconstructible. Each assignment
record carries policy/generator versions, observation, mode, iteration/pass,
partition sizes, ordering policy, completed realization IDs, and namespace,
partition, and reconstruction digests. Records are sorted before their summary
digest; observation products select the matching assignment and coadds use a
stable sorted join (`noise_realization_identity.h:247-449`). No sign vector,
per-sample identity array, or dense correlation matrix is persisted.

The sidecar schema is intentionally advanced to
`citlali-noise-products-provenance-v2`, with an assignment policy, compact
records, summary digest, and requested/effective/realized state
(`noise_provenance.h:12-44`; `noise_config_serialization.h:23-136`). FITS
realization HDUs carry mode, key version, realization ID, product scope,
assignment digest, product digest join, diagnostic restriction, signal-state,
and negative-source policy (`fits_image_metadata_keys.h:118-138`). Required
writer failures throw an output-class exception and cannot be logged away
(`required_output_failure.h:10-15`); run completion and the atomic sidecar
write occur only after output execution (`reduction_execution.h:260-301`).

This application surface is internally conformant, but the end-to-end
repository boundary is not. The active reduction auditor still declares only
`citlali-noise-products-provenance-v1`
(`tools/baseline/audit_reduction_run.py:176-194`), requires the retired
`boost::random::mt19937`/fixed-default/invocation-scoped identity
(`audit_reduction_run.py:1101-1112`), and has no schema-specific v2 semantic
path. A direct in-memory probe of a v2 record returned `schema_ok: false` and
`noise randomization identity is inconsistent` for both enabled and disabled
modes. Therefore any exact-`38ef728` run audited with required noise-products
provenance is rejected even when its new application output is correct.

The persisted FITS join also remains invisible to the standard baseline
manifest/comparison inventories: the intersection of the nine new identity
cards with both `summarize_outputs.FITS_HEADER_EXACT_KEYS` and
`compare_reduction_products.FITS_HEADER_KEYS` is empty. The 22-test FITS suite
does directly verify the application writer, but the standard evidence tooling
cannot retain or compare those identities.

F002 therefore remains open. This is a bounded validation/provenance
integration defect, not a request for dense products, a new framework, or an
astronomical run.

### F003 — truthful current-mode identity only

The sole implemented ensemble identity is `source_imprinted_current`.
Sidecar policy labels it `restricted_diagnostic_only`, records that
deterministic signal may remain, and permits negative-source realizations
(`noise_config_serialization.h:55-72`). The same identity and restrictions are
persisted in realization FITS headers. No residual ensemble mode or source-free
noise claim was introduced. The bounded current-mode portion is technically
ready for closure; any future `final_pre_readdition_residual` mode remains
prohibited until its separate SCI-FRUIT-001 contract and NOI-002 consumer
dispositions exist.

### F005 — positive enabled count and disabled zero work

Both config validation and config reading require at least one realization
when enabled and allow zero only when disabled
(`noise_config_validation.h:8-11`; `noise_config_read.h:14-20`). The execution
plan independently rejects enabled counts below one; when disabled directly or
by mapmaking, it forces the effective count and all output/weight toggles off
(`noise_execution_plan.h:57-90`). Disabled completion records all eight
realized cardinalities as available zero, `generation_executed: false`,
`zero_work: true`, `outputs_promised: false`, and no assignment records
(`noise_execution_plan.h:155-189`). Effective-config accessors then prevent
generation, allocation, and output work. The shared Pointing/OOF fixture and
static call paths confirm this behavior.

The active reduction auditor encodes the opposite historical representation:
it requires every disabled realized count to be unavailable
(`audit_reduction_run.py:1126-1145`). A direct probe using the repaired
available-zero representation produced eight rejection messages, one for each
required zero cardinality. Its v1 schema and retired randomization check reject
the record earlier in the normal path. Consequently F005's application logic
conforms, but the required repository validation boundary cannot accept or
verify that contract. F005 remains open with F002 pending one bounded validator
repair and fresh re-audit.

### F008 — explicitly enabled Beammap lifecycle

For explicit opt-in, Beammap constructs one named context for each iteration
and pass (`beammap_primary` or `beammap_scan_band_rebuild`), populates all PTC
sign matrices once before map-buffer reset, and records that pass after
mapmaking (`beammap_mapmaking_pass_impl.h:42-112`). Buffer reset no longer
generates or overwrites signs and touches only active map buffers
(`beammap_mapmaking_policy.h:74-116`). The focused permutation test verifies
active-map ordering and prior-history invariance. This bounded F008 lifecycle
is technically ready for closure, with any JINC scientific conclusion still
conditioned on SCI-MAP-002.

Standard Beammap configurations remain disabled
(`config/tolteca/beammap/70_pipeline.yaml:194-197` and
`config/tolteca/v2/beammap/60_beammap_internal_policy.yaml:195-198`). Their
configured count of 10 is inert and resolves to effective/realized zero. It is
not used here as a minimum, adequacy threshold, operational expectation, or
evidence request.

## Focused finding dispositions

| Finding | Proposed disposition | Re-audit basis |
| --- | --- | --- |
| SCI-NOI-001-F001 | **close after coordinator review** | Versioned observation/pass/realization/coherence/channel identity is draw-order independent and passes sequential/OpenMP/scheduling, namespace, and shape-growth fixtures. |
| SCI-NOI-001-F002 | **remain open** | Compact application provenance conforms, but the active required-provenance auditor rejects schema v2 and its randomization identity; standard FITS inventories omit every new join card. |
| SCI-NOI-001-F003 | **close in bounded current-mode scope** | `source_imprinted_current` and its restricted, signal-imprinted, negative-source-permitted meaning are truthful in sidecar and FITS metadata; no residual mode was implemented or authorized. |
| SCI-NOI-001-F005 | **remain open** | Application admission/zero-work behavior conforms, but the active auditor rejects the required available-zero disabled cardinalities and cannot validate an exact repaired run. |
| SCI-NOI-001-F008 | **close in explicit-opt-in lifecycle scope** | Signs are generated once per named Beammap pass/iteration before active-buffer reset and are invariant to active-map order/history; disabled-mode count is deliberately ignored. |
| SCI-NOI-001-F007 | **remain open, held policy** | No runtime, astronomical, Unity, or reduction evidence was requested or executed. The exact-d501 design remains a reference only; later work still requires an exact repaired SHA, concrete question, preapproved comparison/cost/acceptance policy, and FRAMEWORK-NUM-001 admission. |

F004 and F006 were excluded and remain open without re-audit disposition. No
new cross-package handoff is supported or proposed.

## Independent gates at exact `38ef728`

| Gate | Result |
| --- | --- |
| clean entry, exact repair branch tip, HEAD, and parent | pass |
| frozen prompt, manifest, readiness, and five manifest-listed authority digests | pass |
| fresh `BUILD_TESTS=ON` configure | pass after approved dependency-network retry |
| `citlali_cli`, `citlali_test`, and FITS-test builds | pass |
| CLI identity | `v4.0.0-3634-g38ef72860` |
| complete core C++ suite | 567/567 pass; one unrelated disabled lifecycle test reported |
| complete production FITS suite | 22/22 pass |
| focused realization contract rerun | 7/7 pass |
| relevant CTest selection | 15/15 pass |
| Python noise-boundary tests | 10/10 pass |
| full config preflight | 127/127 unit tests; four mode kits; 8/8 compatibility cases; 100% compact coverage; all boundary audits pass |
| new public-header isolation compile | pass |
| active baseline-auditor unit suite | 63/63 pass, but fixtures cover the historical v1 contract only |
| exact v2 baseline compatibility probe | **fail as finding**: schema rejected; new generator rejected; repaired disabled zero counts rejected |
| baseline FITS identity inventory probe | **fail as finding**: 0/9 new cards captured by either standard key set |
| final pre-artifact worktree state | clean |

The supplied worktree initially had no configured `build/` directory, so the
first mandated build command could not execute. A fresh configure initially
encountered sandboxed dependency-network denial; the approved network retry
completed and all builds/tests above then passed. These were setup conditions,
not application test failures. The unrelated disabled external-corpus test was
not treated as success evidence or a blocker.

## Verdict, bounded successor, and stop

The exact repair is **not accepted as the complete F001/F002/F003/F005/F008
closure set**. Proposed package axes remain contract `proposed`, implementation
`nonconformant`, validation `failed` for this narrow repair boundary, production
`existing_use_only`, and verdict `amend`.

A bounded successor needs no scientific-policy choice and no numerical or
astronomical work. It should update the existing baseline run auditor and its
tests to:

1. admit and semantically validate v2 while preserving any intentionally
   supported historical v1 lane;
2. verify the versioned generator/key policy, compact assignments, completion
   IDs, ensemble mode, partition/order, and digest joins;
3. accept and require the approved disabled available-zero cardinalities,
   suppressed promises, and zero-work state; and
4. retain/compare the new realization identity cards in the existing FITS
   summary/comparison boundary.

That successor should rerun the same bounded local gates and return to a fresh
narrow re-audit. It must not change application algorithms, choose a count or
estimator, execute evidence, use Unity, run a reduction, integrate, push, or
change production status.

Only the coordinator may integrate the accompanying ledger proposal, launch a
successor repair/re-audit, change canonical finding status, or authorize
application integration. This re-audit stops at its documentation-only commit.
