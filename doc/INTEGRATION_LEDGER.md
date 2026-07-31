# Citlali Integration Ledger

This is the compact authority for concurrent Citlali workstreams. Update it
when a workstream starts, changes authority, reaches a gate, or integrates.
Detailed scientific and build evidence remains in the linked plans, handoffs,
ADRs, and validation records.

## Branch Authorities

| Workstream | Authority | Purpose | Integration rule | State |
| --- | --- | --- | --- | --- |
| Refactored application | `codex/refactor-mainline` | Canonical source, tests, configuration, operational behavior, and validation history | Normal application changes land here after their affected gates | Active |
| Successor build adaptation | `codex/build-adaptation` | Port the full application into the accepted Tula CMake/Spack architecture | Incorporate mainline regularly; return only after all build-integration gates pass | Active in `/Users/gwilson/GitHub/citlali-refactor-build` |
| Historical Conan 2 adaptation | `codex/conan2-adaptation` at `9aae0e669` | Preserved pointer to the superseded package-manager lane name | Do not resume; successor work moved without source changes | Frozen remotely |
| Historical structural refactor | `codex/structural-refactor` at `171487196` | Forensic pointer to the pre-follow-on integration tree | Do not resume as an application authority | Frozen |
| Historical fruit-loop/raw-IQ topic | `codex/fruit-loop-calibration-reference` at `b02fef613` | Forensic pointer to the topic-named branch before mainline normalization | Do not resume as an application authority | Frozen locally; its existing remote remains historical |

## External Build Inputs

Latest isolated review completed 2026-07-31:

| Repository | Branch | Reviewed commit | Disposition |
| --- | --- | --- | --- |
| `tula_cmake` | `v3.x_spack` | `1ea93f600055e14248b2dbfcf1c16c5487a7b757` | Accepted Spack/CMake foundation plus bounded Tlaloc ECSV matrix; installed fixture independently passes with LLVM 20 |
| `tula` | `v3.x_spack` | `61f862c9cc08f335e946a4f55c5aa5cf35401bb0` | Explicit component graph plus reviewed ECSV table-view lifetime repair and regression test |
| `kidscpp` | `v3.x_spack` | `e3c05ebc75da42151a450bbc8c1b27f1e2e5e61b` | Raw-reader and solver package tests reported against real TolTEC data |
| `citlali` | `v3.x_spack` | `8a1be68354d78110c0c3e0f1d4ee5fd3cea20864` | Installed upstream CLI and 123-scan real-data run reported; full refactor port remains required |

Branch movement does not update this table automatically. Re-review and record
new exact commits before importing subsequent upstream work.

## Current Gates

### Application Mainline

- Preserve the existing local build and Unity workflow.
- Run the focused tests, complete CTest/config gates, and affected-mode Unity
  validation required by the behavior touched.
- Record intentional scientific changes separately from refactoring and build
  integration.

### Spack Adaptation Entry

- The native macOS foundation environment is reproduced with exact Homebrew
  LLVM 20.1.8 and Spack 1.2.2. Its source-built Tula closure passes 14 package
  tests and an independent installed consumer. Local containers remain
  optional.
- Resolve the remaining OpenMP runtime edge and extend the proven foundation
  through Kidscpp and the full refactored Citlali application.
- Identify an accessible real-data fixture and publish an immutable manifest.
- Add a user-owned Unity environment based on user-supplied
  Spack/compiler/module facts.
- Make the remaining application dependency graph source-buildable or
  explicitly record environment-required externals such as CFITSIO.
- Define immutable release sources, a portable lock, and provenance identity.

### Spack Adaptation Exit

The exit gates remain those in the build integration review:

1. Full refactored library and CLI targets build through Spack.
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

- Do not merge `citlali/v3.x_spack` wholesale.
- Import build definitions and infrastructure in coherent, reviewable changes.
- Consume Tula and Kidscpp through explicit versioned package identities.
- Review upstream Citlali source changes individually against mainline; port
  only changes that remain applicable.
- Do not resolve build conflicts by deleting mainline tests, generated config,
  provenance, CLI behavior, or validated application sources.
- Do not combine numerical algorithm changes with build integration commits.

## Repository Hygiene

Existing committed evidence is retained without history rewriting. For new
studies, keep reusable tools, compact reports, manifests, checksums, and
essential fixtures in this repository. Prefer a versioned external evidence
archive for large generated matrices, plots, and reduction products. A later
housekeeping change may improve navigation, but it is not part of Conan 2 or
Spack adaptation.
