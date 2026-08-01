# SCI-CAL-001 detector-calibration scientific-contract audit

## Assignment and immutable dispatch

Conduct one Tier A scientific-contract audit of `SCI-CAL-001`, **Detector
calibration, extinction, and map-unit transfer**.

- Canonical repository: `/Users/gwilson/GitHub/citlali-refactor`.
- Authority branch and governing source:
  `codex/refactor-mainline` at
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Required audit branch: `codex/audit-sci-cal-001`.
- Coordinator registry snapshot:
  `88741d085a08431790c6eee719b6f20052b459d8`.
- Frozen inbound-handoff manifest:
  `doc/audits/handoffs/SCI-CAL-001/SCI-CAL-001_INBOX_MANIFEST_2026-07-31.yaml`.
- Manifest SHA-256:
  `00582bb2dc19da7b3380502cdb5f11a3dd35f64c76e879b657ec2573914222b2`.
- Pre-core authority handoffs: none.
- Post-core evidence handoffs: none.

The Codex app supplies the isolated task worktree. Before any source
inspection or edit, verify that its initial HEAD is exactly the governing SHA,
that the worktree is clean, and that no `codex/audit-sci-cal-001` branch or
other CAL audit worktree already exists. Create the required audit branch at
that exact SHA in the supplied worktree. If any identity or existing-state
check fails, stop without overwriting or reusing uncertain state.

Treat `/private/tmp/citlali-scientific-audit-framework` as read-only
coordination authority. Verify that the registry commit named above is an
ancestor of its clean HEAD and verify the manifest bytes and digest before
derivation. Do not merge or cherry-pick the coordination branch.

## Included scientific scope

Audit the complete calibration transformation from raw detector values and
declared calibration inputs to calibrated TOD and its unit/uncertainty
identity, including:

- detector/APT calibration coefficient identity, normalization, units,
  indexing, selection, and lifecycle;
- application of calibration to detector samples and kernels or response
  companions;
- atmospheric extinction using tau, elevation/airmass, observation identity,
  and any data-derived or fitted state;
- target map-unit conversion, beam/template identity, and the distinction
  among detector-signal, surface-brightness, per-beam, point-source, and
  integrated-flux meanings;
- propagation of statistical calibration uncertainty, correlated/common
  calibration terms, extinction uncertainty, beam/template uncertainty, and
  systematic terms;
- requested, effective, observation-resolved, and realized calibration state,
  provenance, products, metadata, failure policy, and downstream consumers;
  and
- analytic limits, deterministic fixtures, injections or standard-source
  recovery, same-SHA external evidence, and falsifiable acceptance gates.

Explicit exclusions belong to their owning packages: common sample-axis and
gap/interpolation construction (`SCI-ALIGN-001`), pointing/WCS and astrometric
response (`SCI-AST-001`), RTC/PTC filtering mathematics (`SCI-RTC-001` and
`SCI-PTC-001`), map gridding/coaddition (`SCI-MAP-001`), noise estimators
(`SCI-NOI-001/002`), map filtering (`SCI-FLT-001/002`), source fitting,
Pointing/OOF, Beammap inference, and fruit-loop feedback. Trace interfaces and
consumer assumptions across those boundaries, but do not absorb their
estimators or repair them.

## Approved pre-core dependency abstraction

`SCI-ALIGN-001` remains open. Before the independent core is frozen, CAL may
use only this coordinator-approved abstract input fact from the registry
snapshot:

> Calibration receives an ordered detector and telescope sample identity,
> common time axis, timestamps, aligned elevation, applicable sample duration,
> timing-gap/interpolation state, and exact eligibility of original versus
> synthesized samples.

CAL may derive equations against those quantities but must condition every
conclusion that depends on their correctness. Do not inspect ALIGN
implementation or invent its policy. Per-sample calibration identity,
time/elevation response, propagated uncertainty, and conclusions involving
interpolated or misaligned samples remain conditioned on `SCI-ALIGN-001`.

## Independence quarantine and freeze

Read and follow `AGENTS.md` and the project-level architecture/scientific-
convention authorities in the audit worktree. Read the audit process,
generic prompt, and LaTeX template only from the read-only coordination
worktree at:

- `/private/tmp/citlali-scientific-audit-framework/doc/audits/README.md`;
- `/private/tmp/citlali-scientific-audit-framework/doc/audits/templates/PACKAGE_AUDIT_PROMPT_TEMPLATE.md`; and
- `/private/tmp/citlali-scientific-audit-framework/doc/audits/templates/SCIENTIFIC_CONTRACT_AUDIT_TEMPLATE.tex`.

Before the core freeze, do **not** inspect the contents, history, blame,
diffs, tests, or generated references for these quarantined implementation
areas:

- `include/citlali/core/engine/calib.h`;
- `src/citlali/core/engine/calib.cpp`;
- `include/citlali/core/timestream/rtc/calibrate.h`;
- `include/citlali/core/pipeline/flux_calibration.h`; or
- other CAL-specific implementation, test, writer, metadata, configuration,
  provenance, or consumer code discovered by filename/search results.

Filename-only discovery needed to define the quarantine is allowed; content
inspection is not. Record any unavoidable prior exposure.

Create
`doc/audits/packages/SCI-CAL-001_INDEPENDENT_CORE.tex`. Before opening any
quarantined source, it must independently state the estimator, identities,
units, frames, shapes/indexing, variable classification, response, formal and
full covariance, systematic uncertainty, validity/non-finite policy, state
lifecycle, consumer restrictions, analytic limits, and pre-registered tests.
Give a specific rationale for every standard validation method marked not
applicable.

Compile and inspect the independent core with already available offline tools,
then freeze its exact bytes in a dedicated commit. Record its SHA-256,
freeze commit and timestamp, and the exact first implementation-inspection
event. Later corrections create successor bytes; do not rewrite the frozen
core.

## Post-freeze implementation audit

Only after the freeze, inspect the governing source and trace the complete
calibration path through sequential/parallel operators, flags and selection,
configuration and state resolution, failure propagation, uncertainty and
simulation paths, products, metadata/provenance, and every downstream
consumer. Tie source operations to numbered independent equations.

Separate implementation defects, contract gaps, scientific-policy decisions,
evidence gaps, and dependency gaps. Do not treat historical behavior,
successful reduction, metadata wording, or plausible fluxes as scientific
authority. Conditional conformity requires an exact upstream assumption and
a falsifiable test; a known mismatch is nonconformant.

Create and offline-render
`doc/audits/packages/SCI-CAL-001_SCIENTIFIC_CONTRACT_AUDIT.tex`. Include an
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
