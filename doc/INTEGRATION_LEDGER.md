# Citlali Integration Ledger

This is the compact authority for concurrent Citlali workstreams. Update it
when a workstream starts, changes authority, reaches a gate, or integrates.
Detailed scientific and build evidence remains in the linked plans, handoffs,
ADRs, and validation records.

## Branch Authorities

| Workstream | Authority | Purpose | Integration rule | State |
| --- | --- | --- | --- | --- |
| Refactored application | `codex/refactor-mainline` | Canonical source, tests, configuration, operational behavior, and validation history | Normal application changes land here after their affected gates | Active |
| SCI-MAP-001 application integration candidate | `codex/integrate-sci-map-001`, with exact application source `af0c849ce59a5f80e5efc8db435bb6662863052f` from `codex/repair-sci-map-001` followed only by the containing documentation-only integration commit | Accepted bounded implementation, product/provenance contract, truth suites, owner-amended evidence, and frozen campaign/closeout history | After owner/coordinator review, fast-forward `codex/refactor-mainline` only to the verified integration-candidate tip; do not merge audit, coordination, or convolve/noise branches | Committed and locally verified candidate; bounded MAP contract accepted; production remains `existing_use_only` |
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

### SCI-MAP-001 Application Integration

- Final independent re-audit
  `8fc716557ca78b0d220200a92be46fa3545797e9` and final canonical coordination
  candidate `c7bb0214edfd57fddf31165923f08784dfd1b8c9` establish the bounded
  package axes `approved`, `conformant`, `complete`, and
  `existing_use_only`, with bounded-contract verdict `accept`.
- The coordinator-directed 2026-08-05 task separately authorizes application
  integration. Exact application source remains
  `af0c849ce59a5f80e5efc8db435bb6662863052f`; its application tree is
  `47aa745554e47514398e72d579625484abdcb79e`. The branch-tip child is only the
  documentation/status commit containing the linked
  [integration handoff](../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md).
- F001--F011 are closed. F012 is
  `closed_bounded_owner_accepted`, retaining the absent raw/sample ledger,
  pre-normalization/commit-order, operational-chain, and historical same-case
  S-X observation-realization lanes as explicit limitations. F013 remains
  `open_conditioned` on ALIGN, CAL, AST, PTC, and VAL.
- Preserve ordinary compatible numerical behavior, the frozen raw-parent
  snapshot and `RAWPDGST` carriage, explicit unsupported-profile absence, and
  the exact ADR 0009 F009/F010 contract. Do not broaden mapmaking,
  noise-generation, coadd, WCS, threshold, output-routing, configuration
  defaults, or any other mature algorithm.
- The completed seven-case external campaign remains frozen historical
  evidence; do not repeat it. This application integration did not access
  Unity or the external corpus and closes no upstream dependency.
- Owner review may fast-forward `codex/refactor-mainline` to the verified
  integration-candidate tip. Production expansion and any subsequent Conan
  adaptation import remain separate decisions; production stays
  `existing_use_only`.

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
  It becomes eligible for ordinary synchronization only after the owner
  fast-forwards the verified application-integration candidate into
  `codex/refactor-mainline`; MAP closure alone does not update the Conan lane.

## Repository Hygiene

Existing committed evidence is retained without history rewriting. For new
studies, keep reusable tools, compact reports, manifests, checksums, and
essential fixtures in this repository. Prefer a versioned external evidence
archive for large generated matrices, plots, and reduction products. A later
housekeeping change may improve navigation, but it is not part of Conan 2
adaptation.
