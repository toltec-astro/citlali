# Citlali Integration Ledger

This is the compact authority for concurrent Citlali workstreams. Update it
when a workstream starts, changes authority, reaches a gate, or integrates.
Detailed scientific and build evidence remains in the linked plans, handoffs,
ADRs, and validation records.

## Branch Authorities

| Workstream | Authority | Purpose | Integration rule | State |
| --- | --- | --- | --- | --- |
| Refactored application | `codex/refactor-mainline` | Canonical source, tests, configuration, operational behavior, and validation history | Normal application changes land here after their affected gates | Active |
| SCI-MAP-001 bounded repair | `codex/repair-sci-map-001` from `9aae0e669384c5c0c0dda93debc194d6b8dac787` | Isolated bounded implementation, product/provenance contract, local truth suite, owner-amended F005/output-persistence cycle, and reconciled seven-case external corpus | Fresh exact-repair-SHA re-audit precedes any application-mainline integration; do not merge audit, coordination, or convolve/noise candidates into this lane | Second bounded repair locally verified; re-audit pending; not application authority |
| Conan 2 adaptation | `codex/conan2-adaptation` | Port the full application into the accepted Tula/Conan 2 architecture | Incorporate mainline regularly; return only after all build-integration gates pass | Active in `/Users/gwilson/GitHub/citlali-refactor-conan2` |
| Historical structural refactor | `codex/structural-refactor` at `171487196` | Forensic pointer to the pre-follow-on integration tree | Do not resume as an application authority | Frozen |
| Historical fruit-loop/raw-IQ topic | `codex/fruit-loop-calibration-reference` at `b02fef613` | Forensic pointer to the topic-named branch before mainline normalization | Do not resume as an application authority | Frozen locally; its existing remote remains historical |

## External Build Inputs

Latest isolated review completed 2026-07-31:

| Repository | Branch | Reviewed commit | Disposition |
| --- | --- | --- | --- |
| `tula_cmake` | `v3.x_conan2` | `998c229dba3abc178ffae5d45c777bb2e371304f` | Accepted foundation with profile and test-harness follow-up |
| `tula` | `v3.x` | `04aef8a02c0d29b9baa0a3a7f5262c0d4a38597e` | Builds and passes C++ package tests under exact LLVM 20 |
| `kidscpp` | `v3.x` | `6d256fb8f266ee20268d49ade4a2c6728da2c6f7` | Builds; synthetic tests pass; real TolTEC fixture lane remains required |
| `citlali` | `v4.x_conan2` | `ec8965b9075a23ad6d23317db5a9a7fd9672b014` | Full milestone CLI builds and in-tree tests pass; external package consumer is blocked by NetCDF C++ propagation |

Branch movement does not update this table automatically. Re-review and record
new exact commits before importing subsequent upstream work.

## Current Gates

### Application Mainline

- Preserve the existing local build and Unity workflow.
- Run the focused tests, complete CTest/config gates, and affected-mode Unity
  validation required by the behavior touched.
- Record intentional scientific changes separately from refactoring and build
  integration.

### SCI-MAP-001 Bounded Repair

- Preserve ordinary compatible numerical behavior and the exact approved
  F009/F010 contract in ADR 0009; do not broaden the prohibited algorithms or
  production defaults.
- Preserve the frozen raw-parent snapshot and matching `RAWPDGST` carriage for
  filtered successor products. Keep unsupported profiles on their explicit
  pre-successor legacy arithmetic lane without an F009/F010 claim.
- Pass the contract-derived focused truth suite, affected CTests, baseline
  tools, complete config preflight, and touched provenance/output validators at
  one exact committed repair SHA with no required-data skip or unexpected
  error-level record.
- Keep F009 and F010 `addressed_pending_reaudit`, package verdict `amend`, and
  production `existing_use_only`; this lane cannot declare conformance or
  close findings.
- Apply the 2026-08-05 owner amendment only to F005 aggregate/index
  fail-closed safety, coadd-enabled observation-realization persistence, and
  the specified production WCS/card/output tests. Keep normal finite-domain
  arithmetic, WCS policy, defaults, and prohibited algorithms unchanged.
- The human owner completed all seven `SCI-MAP-001-UNITY-001` cases for exact
  candidate `ed28dafb37f9113c0d3c95297148157129a90886`; do not issue or run a
  duplicate campaign. The read-only local reconciliation is recorded in
  `SCI-MAP-001_EXISTING_CORPUS_CLOSEOUT_2026-08-05.md`. F012 is owner-accepted
  only for the bounded external product/execution/SEQ-OMP claims in
  `SCI-MAP-001_OWNER_SCOPE_EVIDENCE_AMENDMENT_2026-08-05.md`; missing lanes
  remain limitations. Codex did not access Unity.
- The versioned local owner package is
  `validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb/`. It prepares
  the exact `ed28dafb37f9113c0d3c95297148157129a90886` campaign and now points
  to the reconciled external corpus closeout; multi-gigabyte products remain
  outside the repository. Its manifest records ALIGN-OD1 through ALIGN-OD8 and
  ALIGN-C001 as owner-approved at record commit
  `4f905f4f353e91847a303f4f3959654f3f03c302`, with its expanded-identity
  correction at `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`, the bounded handoff at
  `0309fd48a973a6e7e136224906ac49c02f0171be`, and coordination-ledger HEAD
  `846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. ALIGN implementation remains
  nonconformant, validation is in progress, production remains
  `existing_use_only`, and MAP evidence closes none of ALIGN, CAL, AST, PTC,
  or VAL.
- Keep F013 conditioned on `SCI-ALIGN-001`, `SCI-CAL-001`, `SCI-AST-001`,
  `SCI-PTC-001`, and `SCI-VAL-001`. A fresh `codex/reaudit-sci-map-001`
  worktree assesses F004, F005, F007, F010, and F011 against the amendment and
  exact repair SHA, records bounded F012 acceptance and limitations, preserves
  F013 dependencies, and issues the next disposition.

### Conan 2 Adaptation Entry

- Fix or explicitly resolve exported NetCDF C++ headers/library metadata so
  Citlali's installed-package consumer compiles.
- Make the macOS profile select or validate LLVM major version 20 rather than
  relying on an unversioned `brew --prefix llvm`.
- Correct the Tula CMake workflow test so both a `conan` executable and
  `python -m conan` are valid launch forms.
- Provide a real TolTEC fixture lane for Kidscpp/Citlali raw-reader tests.

### Conan 2 Adaptation Exit

The exit gates remain those in the build integration review:

1. Full refactored library and CLI targets build through Conan 2.
2. Installed library and CLI package consumers pass.
3. Complete local CTest, configuration, baseline, ledger, and exit-policy gates
   pass without required-data skips.
4. Source, dependency, compiler, configuration, and dirty-state provenance are
   truthful.
5. The exact commit builds on Unity without source edits and passes a point
   smoke reduction.
6. The frozen same-SHA point, OOF, science, and Beammap matrix passes.
7. The existing build remains available until a separate retirement decision.

## Import Policy

- Do not merge `citlali/v4.x_conan2` wholesale.
- Import build definitions and infrastructure in coherent, reviewable changes.
- Consume Tula and Kidscpp through explicit versioned package identities.
- Review upstream Citlali source changes individually against mainline; port
  only changes that remain applicable.
- Do not resolve build conflicts by deleting mainline tests, generated config,
  provenance, CLI behavior, or validated application sources.
- Do not combine numerical algorithm changes with build integration commits.
- Do not import the isolated SCI-MAP-001 repair directly into the Conan 2 lane.
  It becomes eligible for ordinary synchronization only after the required
  independent disposition and application-mainline integration.

## Repository Hygiene

Existing committed evidence is retained without history rewriting. For new
studies, keep reusable tools, compact reports, manifests, checksums, and
essential fixtures in this repository. Prefer a versioned external evidence
archive for large generated matrices, plots, and reduction products. A later
housekeeping change may improve navigation, but it is not part of Conan 2
adaptation.
