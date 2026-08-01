# SCI-ALIGN-001 sample-alignment scientific-contract audit

## Assignment and immutable dispatch

Conduct one Tier A scientific-contract audit of `SCI-ALIGN-001`, **Sample
alignment, scan slicing, and gap interpolation**.

- Canonical repository: `/Users/gwilson/GitHub/citlali-refactor`.
- Authority branch and governing source:
  `codex/refactor-mainline` at
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Required audit branch: `codex/audit-sci-align-001`.
- Coordinator registry snapshot:
  `99bfa3562ecfa05c0c10de24a255a6a18ff313d2`.
- Frozen inbound-handoff manifest:
  `doc/audits/handoffs/SCI-ALIGN-001/SCI-ALIGN-001_INBOX_MANIFEST_2026-07-31.yaml`.
- Manifest SHA-256:
  `4bad1e6b853a1176c40dfac405fe2a88214b129d7fba4eed9261f56725a17d89`.
- Pre-core authority handoffs: none.
- Post-core evidence handoffs: none.

The canonical registry contains no integrated record addressed to
`SCI-ALIGN-001` at the snapshot above. Records addressed to CAL, MAP, RTC,
PTC, VAL, FLT, MODE, or BEAM are not part of this inbox and must not be opened
as though they were ALIGN evidence.

The Codex app supplies the isolated task worktree. Before any source
inspection or edit, verify that its initial HEAD is exactly the governing SHA,
that the worktree is clean, and that no `codex/audit-sci-align-001` branch or
other ALIGN audit worktree already exists. Create the required audit branch at
that exact SHA in the supplied worktree. If any identity or existing-state
check fails, stop without overwriting, moving, deleting, or reusing uncertain
state.

Treat `/private/tmp/citlali-scientific-audit-framework` as read-only
coordination authority. Verify that the registry commit named above is an
ancestor of its clean HEAD and verify the manifest bytes and digest before
derivation. Do not merge or cherry-pick the coordination branch.

## Included scientific scope

Audit the complete construction of the common detector/telescope/optional-HWPR
sample identity before scientific conditioning, including:

- native detector, telescope, and optional HWPR stream identities, clock or
  timestamp origins, cadence, ordering, sample duration, acquisition bounds,
  and missing/non-finite state;
- the sign, unit, reference interface, rounding/subsample policy, and
  application stage of every interface synchronization offset;
- definition of the common time axis and the exact mapping, selection,
  interpolation, or resampling operator from every native stream onto it;
- telescope-variable interpolation, including scalar versus angular/circular
  quantities, coordinate/frame identity, endpoint behavior, and validity;
- scan construction and slicing: start/stop convention, half-open versus
  closed bounds, overlap, truncation, empty/short scans, final samples,
  observation boundaries, and stable scan/sample indexing;
- timing-gap detection, gap extent and identity, fill/interpolation operator,
  edge gaps, consecutive gaps, long-gap policy, and the distinction between
  original, synthesized, unavailable, and invalid samples;
- propagation of formal covariance, cross-sample correlation, interpolation
  uncertainty, response/transfer, and duration/exposure consequences;
- requested, effective, observation-resolved, and realized alignment state,
  provenance, diagnostics, products, metadata, failure policy, and lifecycle;
- sequential/OpenMP or other alternate paths and deterministic equivalence;
  and
- downstream contracts consumed by CAL, AST, RTC, and VAL, with interfaces
  traced without absorbing their estimators.

Explicit exclusions belong to their owning packages: detector calibration,
extinction, and target-unit transfer (`SCI-CAL-001`); pointing corrections,
coordinate construction, astrometric response, and WCS (`SCI-AST-001`);
RTC/PTC filtering or cleaning mathematics (`SCI-RTC-001` and `SCI-PTC-001`);
cross-stage science eligibility policy beyond ALIGN's own original/synthesized/
invalid facts (`SCI-VAL-001`); mapmaking, noise estimation, map filtering,
source or mode fitting, Beammap inference, and fruit-loop feedback. HWPR
loading/alignment identity is in scope, but polarization demodulation and any
scientific polarization interpretation are excluded and remain unavailable
without a separately approved contract.

## Approved pre-core abstractions

ALIGN has no upstream audit-package dependency. Before the independent core is
frozen, use only explicit abstract native streams:

> Each detector, telescope, and optional HWPR input has an observation and
> interface identity, ordered native sample axis, timestamp or clock origin,
> cadence, acquisition bounds, and explicit missing/non-finite state. Each
> requested interface offset has a sign, unit, reference interface, and
> application stage. The audit must derive the contract for resolving these
> inputs; their realization in the governing source is not assumed correct.

The approved `SCI-CAL-001` contract at owner-decision commit
`e8bd929008140e2ea8b44bfdc80b0a531b488765` supplies one downstream consumer
requirement that may constrain ALIGN output before the freeze:

> CAL receives an ordered detector and telescope sample identity, common time
> axis, timestamps, aligned elevation, applicable sample duration, timing-gap/
> interpolation state, and exact eligibility of original versus synthesized
> samples.

This is an approved consumer requirement, not evidence that ALIGN implements
it and not authority for ALIGN's estimator. AST, RTC, and VAL remain
unaudited; before the core freeze, treat their needs only as provisional
consumer interfaces stated in the package inventory. Do not inspect ALIGN
implementation or invent external clock/header semantics to satisfy a
consumer.

## Independence quarantine and freeze

Read and follow `AGENTS.md`, the TolTEC context skill, and the project-level
architecture/scientific-convention authorities in the audit worktree. Read
the audit process, generic prompt, and LaTeX template only from the read-only
coordination worktree at:

- `/private/tmp/citlali-scientific-audit-framework/doc/audits/README.md`;
- `/private/tmp/citlali-scientific-audit-framework/doc/audits/templates/PACKAGE_AUDIT_PROMPT_TEMPLATE.md`; and
- `/private/tmp/citlali-scientific-audit-framework/doc/audits/templates/SCIENTIFIC_CONTRACT_AUDIT_TEMPLATE.tex`.

Before the core freeze, do **not** inspect contents, history, blame, diffs,
tests, or generated references for these quarantined implementation areas:

- `include/citlali/core/engine/detail/todproc_alignment_impl.h`;
- `include/citlali/core/engine/detail/kidsproc_gaps_impl.h`;
- `include/citlali/core/pipeline/timestream_scan_generation.h`;
- `include/citlali/core/pipeline/telescope_timestream_alignment.h`;
- `include/citlali/core/pipeline/timestream_alignment_helpers.h`;
- `include/citlali/core/pipeline/timestream_alignment_state.h`;
- `include/citlali/core/pipeline/scan_indices.h`;
- `include/citlali/core/pipeline/timing_gap_policy.h`;
- `include/citlali/core/pipeline/timing_gap_output.h`;
- `include/citlali/core/pipeline/timing_gap_log_file.h`;
- `include/citlali/core/config/interface_sync_config.h`;
- `include/citlali/core/config/interface_sync_config_validation.h`;
- `include/citlali/core/pipeline/interface_sync_config_adapter.h`;
- `include/citlali/core/pipeline/interface_sync_state.h`;
- `include/citlali/core/pipeline/citlali_config_read_sync_offsets.h`;
- `include/citlali/core/pipeline/observation_timing.h`;
- `include/citlali/core/pipeline/telescope_data_loading.h`;
- `include/citlali/core/pipeline/hwpr_loading.h`;
- `include/citlali/core/pipeline/hwpr_policy.h`;
- `include/citlali/core/pipeline/hwpr_state.h`;
- `include/citlali/core/engine/telescope.h`;
- `src/citlali/core/engine/telescope.cpp`; or
- other ALIGN-specific source, test, config, writer, metadata, provenance, or
  consumer code discovered by filename-only searches.

Filename-only discovery needed to define the quarantine is allowed; content
inspection is not. Record any unavoidable prior exposure.

Create
`doc/audits/packages/SCI-ALIGN-001_INDEPENDENT_CORE.tex`. Before opening any
quarantined source, it must independently state the native and common sample
identities, units/frames, shapes/indexing, variable classification, alignment/
interpolation operator, scan-window convention, response, formal and full
covariance, interpolation/systematic uncertainty, duration/exposure meaning,
validity/non-finite/gap policy, state lifecycle, consumer restrictions,
analytic limits, and pre-registered tests. Give a specific rationale for each
standard validation method marked not applicable.

Compile and inspect the independent core with already available offline tools,
then freeze its exact bytes in a dedicated commit. Record its SHA-256, freeze
commit and timestamp, and the exact first implementation-inspection event.
Later corrections create successor bytes; do not rewrite the frozen core.

## Post-freeze implementation audit

Only after the freeze, inspect the governing source and trace the complete
ALIGN path through raw input/header authority, sync-offset resolution, scan
construction, gap detection/fill, telescope and HWPR alignment, sequential/
parallel paths, flags and validity, configuration/state resolution, failure
propagation, covariance and simulation paths, products, metadata/provenance,
and every downstream consumer. Tie source operations to numbered independent
equations.

After the core freeze, inspect the approved CAL decision only for the bounded
consumer requirement above and inspect any ALIGN manifestations in downstream
code as evidence, not as authority for the ALIGN estimator. The dispatch
manifest has no handoff dispositions to perform. Any handoff integrated after
the registry snapshot is a late arrival and must be held for coordinator
disposition rather than silently opened.

Separate implementation defects, contract gaps, scientific-policy decisions,
evidence gaps, and dependency gaps. Do not treat historical behavior,
successful reductions, metadata wording, or plausible interpolation as
scientific authority. Conditional conformity requires an exact external-input
assumption and a falsifiable test; a known mismatch is nonconformant.

Create and offline-render
`doc/audits/packages/SCI-ALIGN-001_SCIENTIFIC_CONTRACT_AUDIT.tex`. Include an
exact human-run Unity evidence request where needed, but do not connect to
Unity. Propose one bounded handoff record per affected target package on the
audit branch; do not edit the canonical ledger or handoff registry.

Commit audit artifacts coherently and stop with the exact branch/commit,
clean state, frozen-core identity, source/equation trace, findings, unresolved
owner decisions, dependencies, status axes, verdict, validation evidence,
consumer restrictions, proposed ledger patch, and outbound handoffs.

Do not modify application code, tests, build files, production documentation,
other branches/worktrees, or the coordination line. Do not repair, merge,
rebase, push, install/download software, use the network, launch another
audit, or claim production authorization.
