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

Latest isolated review completed 2026-08-10:

| Repository | Branch | Reviewed commit | Disposition |
| --- | --- | --- | --- |
| `tula_cmake` | `v3.x_spack` | `6433cdabe7010d0af2d0ba69da7af27510391b80` | Retains the normalized dependency adapters and separates artifact provenance from deployment provenance |
| `tula` | `v3.x_spack` | `aa16c853c6b596c04ccdc90dc3acc4ce2006d947` | Exports source, compiler, package, and concrete build identity through its installed interface |
| `kidscpp` | `v3.x_spack` | `498ece1113001ae2d42d96d9fc29152aea3eaaef` | Exports equivalent artifact provenance and retains portable versus optional real-data test separation |
| `citlali` | `v3.x_spack` | `ceb4335c3f00b52af58ce2c09093b863040434b6` | Reference implementation only; its provenance design is reviewed without replacing the refactored application |
| `tolteca_deploy` | `main` | `0a6b896` (`v0.1.1`) | Read-only deployment design evidence; not consumed or modified by this lane |

Branch movement does not update this table automatically. Re-review and record
new exact commits before importing subsequent upstream work. The three consumed
dependency revisions are also enforced by `spack/upstream-revisions.json`;
`tolteca_deploy` is neither modified nor used as an input to this lane.

The accepted `tula-netcdf-cxx4` adapter does not yet export NetCDF-C's include
directory to installed consumers. Citlali therefore retains a direct
`netcdf-c` dependency and target until that upstream interface is corrected;
this is a build-contract workaround, not a fork of the upstream package.
The 2026-08-10 upstream revision leaves that adapter interface unchanged, so
the workaround remains required despite the other release fixes in this round.
Tula's generated version header also still reports `cxx_standard=0` while the
Kidscpp header correctly reports C++23. Citlali's own build provenance records
the effective C++23 standard; the Tula metadata defect remains upstream debt.

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
- The native Kidscpp environment now includes an explicit Spack
  `llvm-openmp@20.1.8` edge. Kidscpp installs from source, its independent
  installed consumer passes, and a separate reader consumer opens a current
  raw TolTEC file and reads a two-sample I/Q slice. The historical upstream
  real-data fixture still needs an accessible immutable manifest.
- Citlali now consumes upstream-owned `tula-ccfits`, `tula-netcdf-cxx4`, and
  `tula-perflibs` targets. Its temporary package and CMake adapters have been
  removed, and general pipeline OpenMP is distinct from Wiener OpenMP.
- The full refactored Citlali library, production CLI, 539 enabled CTests,
  complete config preflight, installed CLI, and independent installed package
  consumer now pass natively under exact Homebrew LLVM 20. The CLI records its
  source dirty state and concrete Spack DAG identity. Managed launches also
  require their runtime environment lock to contain that exact root DAG and
  record the deployment profile and lock digest in product metadata.
- Identify an accessible real-data fixture and publish an immutable manifest.
- Add a user-owned Unity environment based on user-supplied
  Spack/compiler/module facts. The first inventory identifies Ubuntu 24.04,
  GCC/G++/GFortran 13.3, CMake 3.30, Python 3.12, no Ninja, and no
  user-callable Spack.
  A user-owned Spack 1.2.2 `unity-gcc13` profile and prerequisite gate are now
  prepared. Its earlier lock is superseded by the upstream-adapter graph;
  fresh concretization passed at lock SHA-256 `d2204524bc170cf9e9458a9f83f2730f4e44c78fe628eb01f7d9f8dee0c52f72`.
  Unity job `62888690` installed the full graph and passed all 539 enabled
  developer CTests at exact Citlali `0add18c24`, but stopped before installed
  CLI/consumer acceptance because the expected Linux Spack development build
  directory was not ignored and the clean-source gate rejected it. The
  hygiene rule is corrected; a fresh exact-SHA acceptance job and point run
  remain pending.
- The macOS graph builds CFITSIO 4.3.0 from source and declares Homebrew FFTW
  plus GCC 15 Fortran as checked host externals; all C/C++ application code is
  LLVM 20.
- Promote the checked development revision manifest into immutable release
  sources and a portable release lock.

### Spack Adaptation Exit

The exit gates remain those in the build integration review:

1. Full refactored library and CLI targets build through Spack.
2. Installed library and CLI package consumers pass.
3. Complete local CTest, configuration, baseline, ledger, and exit-policy gates
   pass without required-data skips.
4. Source, dependency, compiler, configuration, dirty-state, and managed
   deployment provenance are truthful and the runtime lock root matches the
   executable's compiled DAG.
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
