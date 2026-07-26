# TolTECA Conan 2 Build Integration Review

Date: 2026-07-26

Status: initial architecture review complete; **Adapt** path selected. The
reviewed implementation is a strong foundation for the successor build, but
its current Citlali target is a deliberately limited library milestone rather
than the full production reduction application.

## Reviewed Evidence

The review used isolated checkouts at these exact revisions:

| Repository | Branch | Commit |
| --- | --- | --- |
| `toltec-astro/tula_cmake` | `v3.x_conan2` | `e17e04600b2c02a45d20d58bf2b9ca1ff2e26054` |
| `toltec-astro/tula` | `v3.x` | `884436482799845a2ac589b561c8d309db334f2e` |
| `toltec-astro/kidscpp` | `v3.x` | `7dd96238dc3709eca1bfda66aec546384030bab6` |
| `toltec-astro/citlali` | `v4.x_conan2` | `00abcd20c5c98e75eb587b2b1540b5837307c99b` |

The Tula design records, recipes, build launcher, feature registry, CMake
targets, package tests, and Citlali milestone were compared with
`TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md` and the current
refactored source graph.

## Decision

Use the new Tula/Conan 2 architecture as the foundation, but do not replace the
validated Citlali source tree with `v4.x_conan2`.

The accepted direction is:

1. Conan 2 owns dependency graph resolution, package identity, profiles, and
   generated CMake presets.
2. CMake consumes an already-resolved graph and never downloads or invokes
   Conan.
3. Tula CMake owns the validated feature/provider registry and normalized
   dependency targets.
4. Tula, Kidscpp, and Citlali remain explicit first-party packages.
5. The full refactored Citlali library, CLI, generated provenance, tests, and
   operational behavior are carried into that model through a bounded
   adaptation.

This is an **Adapt** decision under the requirements document. Direct adoption
would silently discard most of the validated application.

## Architecture Worth Adopting

The implementation provides several material improvements over the current
FetchContent/Conan 1 arrangement:

- a clear Conan/CMake responsibility boundary;
- one thin `./build` launcher and generated CMake presets;
- explicit first-party package versions and dependency edges;
- typed, centrally validated feature/provider choices;
- normalized `tula::<feature>` targets across Conan, CPM, and system
  providers;
- checksummed CPM source archives;
- isolated package creation and downstream `test_package` consumers;
- explicit compiler profiles and feature-matrix tests;
- no hidden sibling-source lookup in normal package builds; and
- an appropriate static-library boundary without inventing a public Citlali
  ABI.

The implementation has reportedly passed GCC 14 on Ubuntu and LLVM 20 on
macOS. Its checked-in design also describes GCC 13, GCC 14, and Clang 20
profile gates. These are useful build-system results, but they are not yet
full Citlali CLI results.

## Current Citlali Milestone Boundary

The reviewed `v4.x_conan2` target:

- builds five existing implementation sources;
- packages `citlali::citlali` as a static library;
- installs 41 Citlali headers;
- runs the Gaussian-model test target; and
- deliberately excludes the historical CLI.

The refactored production tree:

- has 709 checked-in Citlali headers;
- builds the same five numerical sources plus typed-config, output-lease, and
  restart-checkpoint implementations;
- builds the operational `citlali` CLI;
- discovers more than 500 focused CTests;
- generates Git-version and embedded default-config headers; and
- supports the point, OOF, science, and Beammap reduction modes validated on
  Unity.

The new branch is therefore evidence that the package architecture works for a
vertical library slice. It is not yet evidence that the operational
application builds or runs.

## Requirement Assessment

| Area | Disposition | Evidence or gap |
| --- | --- | --- |
| Canonical operator command | Partial | `./build` is clear, but currently produces no production CLI. |
| Conan/CMake ownership | Pass | Conan installs and generates; CMake consumes the generated preset. |
| First-party package identity | Pass in principle | Tula 3.1.0, Kidscpp 3.1.0, and Citlali 4.0.0 form explicit graph nodes. |
| Third-party identity | Partial | Versions and CPM checksums are explicit; release remote/configuration is not deployed and no Citlali lockfile evidence was found. |
| Offline/repeatable resolution | Partial | Isolated package creation is tested, but the launcher fetches the CLI and normal resolution still depends on configured remotes unless a cache/lock discipline is supplied. |
| Supported compiler profiles | Pass for infrastructure slice | GCC 13/14 and Clang 20 profiles exist; the full CLI still requires a Unity lane. |
| Full Citlali library boundary | Fail at current milestone | Only five sources and 41 headers are packaged. |
| Production CLI | Fail at current milestone | Explicitly omitted. |
| Generated source identity | Fail at current milestone | No Citlali, Kidscpp, or Tula Git-revision contract equivalent to the production headers. |
| Embedded default config | Fail at current milestone | The CLI and its generated default-config header are absent. |
| Full test gates | Fail at current milestone | Infrastructure matrices are substantial, but Citlali runs only its small library-slice test. |
| TolTECA/Unity operation | Not demonstrated | No full CLI, cluster profile, executable snapshot, or reduction exists yet. |
| Build timing evidence | Not demonstrated | Clean, no-op, and representative incremental measurements are still required. |

## Dependency/API Compatibility Findings

Most Tula headers directly referenced by the active refactor remain available
in Tula v3. The generated `tula/config.h` is also provided by the package.

Kidscpp is the principal compatibility boundary. Kidscpp v3 intentionally
removes sweep finding/fitting and the TolTEC-specific raw-file adapter. The
active Citlali tree still includes:

- `kids/sweep/fitter.h`;
- `kids/toltec/toltec.h`;
- generated Kidscpp Git-version metadata; and
- the TolTEC `get_meta` and `read_data_slice` operations used to ingest raw
  detector files.

`KidsDataProc` constructs and reports a `SweepFitter`, but no production call
to that fitter was found. That dependency is a candidate for removal while
retaining compatibility parsing for existing low-level YAML.

The TolTEC raw-data adapter is not unused. It is required by current
production ingestion. Its long-term owner must be made explicit before the
full application can use Kidscpp v3. Acceptable outcomes include a focused
Kidscpp instrument-I/O component or an explicit Citlali input adapter. Copying
the legacy namespace into an incidental compatibility header is not an
acceptable permanent boundary.

The full application also directly uses HDF5 and Zlib APIs. They must be
declared dependencies rather than assumed transitive consequences of system
NetCDF:

- HDF5 is used to control low-level diagnostic reporting around raw-data I/O.
- Zlib implements compressed Citlali logs.

The successor dependency registry or Citlali recipe must model those direct
requirements explicitly.

## Generated Provenance Contract

The current outputs record source identity for Citlali, Kidscpp, and Tula in
logs, FITS headers, NetCDF metadata, product indices, and restart checkpoints.
Package semantic versions alone are not sufficient during active development.

The adapted build shall generate immutable build metadata containing:

- Citlali source commit and dirty-state policy;
- Kidscpp package version plus recipe/package revision or equivalent source
  identity;
- Tula package version plus recipe/package revision or equivalent source
  identity;
- build timestamp policy;
- compiler identity, build type, and relevant performance options; and
- dependency lock or graph identity.

`citlali --version` and published science products must continue to report
truthful provenance. The embedded default config must remain generated from
the checked-in YAML with a dependency edge that rebuilds it when the YAML
changes.

## Adaptation Projects

### A. Complete the dependency contract

1. Decide ownership of the TolTEC raw-data adapter with the Tula/Kidscpp build
   owner.
2. Remove the unused runtime `SweepFitter` dependency from Citlali while
   preserving accepted configuration compatibility.
3. Add explicit HDF5 and Zlib dependency features or recipe requirements.
4. Define a lockfile/package-revision policy for supported builds.
5. Add a Unity Release profile with the required GCC, GNU OpenMP, system
   NetCDF/HDF5 choices, and scheduler-independent paths.

### B. Port the full production targets

1. Carry the current static `citlali` target and all eight active compiled
   implementation sources into the Conan 2 project.
2. Build and package the operational `citlali_cli` executable as `citlali`.
3. Preserve the OMP Wiener implementation option as package/build identity.
4. Generate the default-config and provenance headers through declared build
   inputs.
5. Link every direct dependency through named imported targets.

### C. Restore the validation surface

1. Register both current C++ test executables and preserve test discovery.
2. Run the complete CTest set rather than only Gaussian-model tests.
3. Run config preflight, baseline-tool tests, validation-ledger checks,
   science-change checks, and session-exit audits.
4. Add a package consumer test for the library and a CLI smoke test for
   `--version` and default-config dumping.

### D. Verify developer and operator workflows

1. Measure clean configure/build.
2. Measure a true no-op build.
3. Measure one representative CLI/header incremental rebuild.
4. Avoid forcing `cmake --fresh` for every ordinary developer rebuild; retain
   it as the explicit clean-configure path.
5. Build the exact commit on Unity and run a point smoke reduction before
   freezing the final candidate.

## Stop Conditions

Do not remove the existing build path until all of the following are true:

1. the full CLI builds through Conan 2 locally;
2. every local test and configuration gate passes;
3. source and dependency provenance are truthful;
4. the Unity profile builds the same commit without source edits;
5. a Unity point reduction passes its contract and numerical comparison; and
6. rollback to the existing build remains straightforward.

Do not combine build integration with changes to RTC, PTC, mapmaking, Wiener,
JINC, fruit-loop, fitting, or calibration algorithms. Any compatibility source
change must be isolated, tested, and shown not to alter requested products.

## Questions For The Build Owner

1. Should the TolTEC raw-file adapter remain in Kidscpp as a focused optional
   component, or move to a Citlali-owned input boundary?
2. Is the removal of `SweepFitter` from the C++ package intentional for all
   current Citlali deployments?
3. What is the intended package/lockfile promotion model before the TolTEC
   Conan remote is deployed?
4. How should development source commits and package revisions be exposed to
   Citlali output provenance?
5. Is `./build` intended only for clean configure/build, with direct preset
   builds used for incremental development?
6. Which provider combination is intended for Unity Release builds,
   particularly NetCDF C/C++, HDF5, Zlib, OpenMP, FFTW, and CCfits?

These questions refine the adaptation. They do not invalidate the accepted
architecture.
