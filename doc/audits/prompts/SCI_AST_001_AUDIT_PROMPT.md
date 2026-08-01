# SCI-AST-001 pointing, astrometry, detector-coordinate, and WCS audit

## Assignment and immutable identities

Act as the dedicated independent scientific-contract auditor for
`SCI-AST-001`, **Pointing, astrometry, detector coordinates, and WCS**. This is
one Tier A audit-only task at Ultra effort. It is not implementation, repair,
re-audit, integration, production authorization, or Unity work.

- Canonical repository: `/Users/gwilson/GitHub/citlali-refactor`.
- Authority ref: `codex/refactor-mainline`.
- Exact governing application SHA:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Required audit branch: `codex/audit-sci-ast-001`.
- Read-only coordination framework:
  `/private/tmp/citlali-scientific-audit-framework`.
- Coordinator registry snapshot recorded by the frozen manifest:
  `65a0c50ad49f0602a4e20bb39e3f781843ab872d`.
- Frozen inbox manifest:
  `doc/audits/handoffs/SCI-AST-001/SCI-AST-001_INBOX_MANIFEST_2026-08-01.yaml`.
- Frozen manifest SHA-256:
  `369fcbd398437b2fe0c60162577faf41d397c7b32afb45c3c3169ac8c71b8e1c`.
- Pre-core authority handoff IDs: none.
- Post-core evidence handoff IDs: `SCI-AST-001-XAUD-001` only.

Before any source inspection or edit, verify the app-supplied worktree is
clean and detached at the exact governing SHA and that `codex/refactor-mainline`
still resolves to it. Verify no `codex/audit-sci-ast-001` branch or other AST
audit worktree exists. Create only that audit branch in the app-supplied
worktree. If any identity or state differs, stop safely; do not delete, reset,
move, overwrite, or reuse uncertain state.

Use the TolTEC context skill and obey the repository `AGENTS.md`. Verify the
coordination worktree is clean, the registry snapshot is an ancestor of its
HEAD, and the prompt and manifest bytes match their recorded SHA-256 values
before relying on them.

## Bounded scientific scope

Audit the observation-local transformation from approved aligned telescope
and detector inputs plus a requested astrometric plan to realized
per-sample coordinates and persisted WCS identity. Include:

- validation and application of supplied azimuth/altitude pointing offsets,
  their constant or support-interpolated realization, sign, unit, support,
  uncertainty, and observation lifecycle;
- requested, effective, observation-resolved, and realized astrometry state;
- telescope boresight coordinates, detector focal-plane offsets, and their
  composition into per-detector/per-sample coordinates;
- coordinate frames, epochs, wrap/sign conventions, angular topology,
  units, ordering, indexing, missing/non-finite policy, and support;
- AltAz tangent-plane point/OOF and Beammap coordinates, equatorial J2000 TAN
  science coordinates, projection center, pixelization, FITS WCS,
  handedness, and one-based FITS versus zero-based memory indexing;
- mean, formal covariance, systematic/calibration uncertainty, interpolation
  and projection response, selection/availability uncertainty, and any
  approximations or unavailable terms;
- observation lifecycle, simulation/alternate paths, sequential/parallel
  equivalence, product writing, metadata, provenance, and required-failure
  propagation; and
- exact boundary contracts presented to RTC, PTC, MAP, source, Pointing/OOF,
  and Beammap consumers.

Treat TolTECA selection of pointing-support records as an upstream authority
boundary. Citlali owns validation and application of the supplied records; it
does not select calibrators or backfill missing upstream authority.

## Explicit exclusions and dependencies

Do not audit or repair:

- native detector/telescope/HWPR clock alignment, scan slicing, or gap
  synthesis (`SCI-ALIGN-001`);
- photometric calibration, extinction, flux scale, or map units
  (`SCI-CAL-001`);
- RTC/PTC filtering and correlated-mode estimators (`SCI-RTC-001`,
  `SCI-PTC-001`);
- map accumulation, gridding, coaddition, or map-validity policy
  (`SCI-MAP-001`, `SCI-MAP-002`, `SCI-VAL-001`);
- map-domain source finding/fitting or Pointing/OOF fit estimators
  (`SCI-SRC-001`, `SCI-MODE-001`);
- Beammap fitting, detector calibration/APT production, or sensitivity
  estimation (`SCI-BEAM-001`); or
- enabled polarimetry/HWPR science.

`SCI-ALIGN-001` is an open implementation dependency. Before the independent
freeze, you may read only its approved owner contract in
`SCI-ALIGN-001_COORDINATOR_DECISION_2026-08-01.md` at decision commit
`4f905f4f353e91847a303f4f3959654f3f03c302`, together with the canonical
ledger state. Abstract inputs may therefore declare field identity, topology,
unit/frame, native-source mapping, timing residual, origin, validity, and
original/synthesized eligibility. Do not assume the governing implementation
supplies them correctly. ALIGN remains implementation-nonconformant,
validation-in-progress, `existing_use_only`, and requires repair and re-audit.

The post-core handoff `SCI-AST-001-XAUD-001` reports a manifestation of that
dependency. Do not open it before freezing the independent AST core. Its
evidence may sharpen AST dependency tests and restrictions, but cannot define
the independent AST estimator or authorize repair.

## Independence quarantine and known implementation paths

Before freezing the independent core, do not inspect the contents, diffs,
history, tests, or generated products of the package implementation surface,
including at least:

- `include/citlali/core/pipeline/astrometry_execution_plan.h`;
- `include/citlali/core/pipeline/observation_calibration_config.h`;
- `include/citlali/core/pipeline/telescope_pointing_operations.h`;
- `include/citlali/core/pipeline/pointing_offset_state.h`;
- `include/citlali/core/pipeline/astrometry_provenance.h`;
- `include/citlali/core/pipeline/astrometry_config_serialization.h`;
- `include/citlali/core/pipeline/telescope_pointing.h`;
- `include/citlali/core/pipeline/fits_image_hdu_names_wcs.h`;
- `include/citlali/core/engine/detail/astrometry_config_impl.h`;
- `include/citlali/core/engine/detail/todproc_pointing_impl.h`;
- `include/citlali/core/engine/pointing.h` and
  `src/citlali/core/engine/pointing.cpp`;
- relevant AST/pointing/WCS tests, product contracts, validation evidence,
  writers, and downstream consumers.

Repository architecture, scientific conventions, controlled vocabulary,
approved upstream abstractions, and product intentions may be read first.
Record unavoidable prior exposure rather than concealing it.

## Phase 1 — independently frozen scientific core

Create
`doc/audits/packages/SCI-AST-001_INDEPENDENT_CORE.tex` before opening the
quarantined sources or post-core handoff. Derive the claimed transformation
from physical coordinate identities and approved abstract inputs, not from
current code. At minimum define numbered equations and falsification tests for:

- native aligned boresight and detector-offset composition;
- pointing-offset sign and constant/two-support interpolation in MJD or
  explicitly approved observation-span fallback, with no extrapolation;
- spherical/circular coordinate handling and RA/azimuth wrap;
- AltAz tangent-plane and equatorial J2000 TAN projection, inverse mapping,
  center/reference pixels, handedness, and index conventions;
- fixed versus fitted/selected/data-derived variables;
- Jacobian/response and covariance propagation through correction,
  coordinate composition, projection, pixelization, and WCS;
- missing-support, non-finite, ambiguous topology, out-of-domain, singular
  projection, and unavailable-uncertainty behavior;
- observation reset and no cross-observation state leakage;
- requested/effective/resolved/realized identity and lossless provenance;
- analytic zero-offset, constant, wrap-boundary, pole/center, round-trip,
  dither, simulation, and sequential/parallel limits; and
- provisional downstream allowlist and fail-closed claims.

Explicitly distinguish astrometric precision, projection response, pixel
support, validity, coverage, source-fit uncertainty, pointing-calibrator
uncertainty, and map-domain significance. For every standard validation method
omitted, state a specific `not_applicable` rationale.

Freeze the exact independent-core bytes in their own commit, record SHA-256
and timestamp, and only then record the first source-inspection and post-core
handoff-opening events. A correction after freeze is a successor revision and
must preserve the original bytes and trigger.

## Phase 2 — exact-source audit

After the freeze, open `SCI-AST-001-XAUD-001` and inspect the exact governing
source. Trace every signal, coordinate, covariance/uncertainty, validity,
configuration, lifecycle, product, and consumer path. Compare each operation
to the numbered independent equations. Test or derive:

- offset sign, support choice, fallback behavior, and extrapolation rejection;
- frame/unit/epoch/topology conversions and wrapping;
- detector identity/order and focal-plane offset composition;
- projection and inverse-WCS round trips, pixel-center/index conventions,
  axis signs, handedness, boundaries, and singularities;
- observation replacement/reset, absent optional state, simulation, repeated
  runs, and compiled execution-path equivalence;
- covariance/Jacobian availability, response, and missing-term semantics;
- exact requested/effective/resolved/realized provenance and required product
  failures; and
- source-crossing times, centroids, and PSF widths as compatibility evidence,
  without turning composite source-fit recovery into AST authority.

Classify findings independently as implementation defects, contract gaps,
scientific policy decisions, evidence gaps, or dependency gaps. Use separate
priority, evidence-basis, confidence, owner, and falsifiable closure gate.
Disposition the one inbound handoff and propose bounded outgoing handoffs for
facts needed by stable consumer packages.

## Required artifact and evidence request

Create and commit only audit artifacts on `codex/audit-sci-ast-001`:

- `doc/audits/packages/SCI-AST-001_INDEPENDENT_CORE.tex`;
- `doc/audits/packages/SCI-AST-001_SCIENTIFIC_CONTRACT_AUDIT.tex`;
- proportional audit-specific manifests/evidence; and
- proposed cross-audit handoffs outside the canonical registry.

Compile the TeX offline and visually inspect the rendered PDF. The final audit
must include exact identities and exposure chronology; estimator and
source/equation trace; frame/unit/index/response/uncertainty/product matrices;
all validation methods or N/A rationales; finding and decision tables; inbound
handoff disposition; downstream restrictions; four independent status axes
and one verdict; and a machine-readable ledger proposal.

Prepare an exact human-run `SCI-AST-001-UNITY-001` request using SSH alias
`unity_toltec` only. It must bind a future exact source SHA, build/dependency
identities, raw/config bytes, representative point/OOF/science/Beammap cases,
source-crossing/centroid/PSF and WCS round trips, edge/wrap/support cases,
sequential/compiled-path equivalence, and required-product outcomes. Do not
connect to Unity or claim external evidence exists.

## Stop rules

Do not modify application code, tests, production documentation, the canonical
ledger/inbox, another branch/worktree, or any external system. Do not merge,
rebase, cherry-pick, push, install/download software, use the network, request
production, repair a finding, or launch another task. Stop after committing a
clean audit report. Report exact commits/digests, findings, unresolved owner
decisions, open dependencies, allowed/restricted consumers, and the proposed
repair/re-audit sequence without claiming conformity beyond the evidence.
