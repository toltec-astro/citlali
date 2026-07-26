# TolTECA Build Integration Requirements - 2026-07-23

## Purpose

Citlali v4.x build work is being developed separately with the stated goal of
supporting both the upstream code and this refactored tree. This document
defines what the refactor needs from that work. It deliberately does not
prescribe CMake structure, dependency technology, package manager, or
development tool.

The preferred outcome is one shared build direction. This project will not
create a competing build-system rewrite while the v4.x approach is being
prepared. When that implementation is available, it will be reviewed against
the requirements below and either adopted, adapted in a bounded way, or
deferred with the remaining limitations recorded.

## Operational Scope

Citlali is deployed as a command-line reduction engine on approximately four
collaborating clusters. It is not currently a general installed C++ SDK.
TolTECA constructs reduction inputs and invokes the executable; Citlali owns
execution and product publication from the generated low-level YAML onward.

The current physical build has:

- an internal static `citlali` target;
- one supported `citlali_cli` executable, emitted as `citlali`;
- a small number of compiled implementation files plus a still
  header-dominant numerical and orchestration graph;
- configure-time default-config and Git-version headers; and
- dependencies supplied through Kidscpp, Tula, CMake modules, system
  libraries, and the existing Conan-mediated cluster environment.

These are facts about the current tree, not mandatory implementation choices
for the successor build.

## Required Outcomes

### Supported Build Modes

The reviewed approach shall:

1. build the Citlali CLI from a clean checkout on at least one documented,
   supported cluster/compiler lane;
2. support the normal TolTECA deployment workflow on the collaborating
   clusters without machine-specific source edits;
3. state whether Citlali is a standalone project, an embedded dependency, or
   both, and test every mode claimed as supported;
4. retain an internal library/test boundary sufficient for focused C++ tests;
   and
5. provide one canonical build command or preset for operators.

Cluster environments need not be identical. Site-specific compiler,
scheduler, filesystem, and installed-library details may remain site
configuration, but the supported differences must be explicit rather than
encoded in an individual's shell state.

### Dependency Identity

The reviewed approach shall:

1. identify exact or bounded versions for direct dependencies;
2. identify the authoritative source of Kidscpp and Tula, including any local
   patches;
3. distinguish fetched, package-managed, and system-provided dependencies;
4. fail with an actionable message when a required dependency is unavailable;
5. avoid unrecorded network resolution during a nominal reproducible build;
   and
6. retain enough dependency identity in the build or validation record to
   interpret a reduction later.

A single package manager is not required. The requirement is a documented and
inspectable dependency resolution, not uniformity for its own sake.

### Generated Inputs And Version Identity

The reviewed approach shall:

1. regenerate default-config and Git-version headers from declared inputs;
2. rebuild them when those inputs change;
3. make `citlali --version` identify the source revision used for the binary;
4. keep generated build artifacts outside the source authority; and
5. demonstrate that a clean build and an incremental rebuild produce truthful
   version information.

### Test And Validation Gates

One documented supported lane shall be able to run:

```bash
ctest --test-dir build --output-on-failure -j 8
$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all
$HOME/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'
$HOME/tolteca/bin/python tools/baseline/validate_validation_ledger.py
$HOME/tolteca/bin/python tools/baseline/validate_science_change_ledger.py
$HOME/tolteca/bin/python tools/refactor/audit_session_exits.py --fail-on-growth
```

CI may use a practical representative environment rather than pretending to
reproduce every cluster. A cluster smoke build remains required before a
candidate is accepted for operations.

### Build Boundaries And Performance

The reviewed approach shall expose enough target and timing information to:

1. distinguish clean configure, clean build, no-op build, CLI-only rebuild,
   and test-build costs;
2. determine which headers are intended interfaces and which contextual
   implementation fragments are private;
3. measure whether moving a coherent cold implementation boundary into
   `.cpp` files reduces dependency or compile cost;
4. preserve established optimization, OpenMP, FFTW, and numerical-library
   behavior; and
5. avoid claiming an improvement without a representative measurement.

This requirement does not mandate C++ modules, shared libraries, unity builds,
precompiled headers, or a specific source split.

## Deliberate Non-Goals

The integration review shall not require:

- a stable public C++ API or ABI;
- install/export support without a real external client;
- identical dependency provisioning on all four clusters;
- a rewrite of mature RTC, PTC, JINC, Wiener, mapmaking, or fitting kernels;
- physical relocation of every historical header;
- activation of legacy or experimental executables;
- a generic distribution system for platforms outside the supported
  collaboration; or
- build-time optimization that worsens runtime behavior or scientific
  reproducibility.

The static library is currently an internal composition and testing boundary.
That remains the default unless a concrete external consumer is accepted.

## Evidence Requested From The V4.x Work

The review packet should contain:

1. repository/branch and exact commit;
2. intended standalone and TolTECA integration model;
3. canonical clean configure and build commands;
4. compiler, C++ standard, build type, and important optimization settings;
5. direct dependency sources and resolved versions;
6. one clean build log and elapsed time;
7. one no-op and one representative incremental build time;
8. discovered CTest count and results;
9. generated-version evidence from `citlali --version`;
10. known site-specific configuration points; and
11. limitations or unsupported combinations.

Evidence from one representative cluster is sufficient for the architectural
review. Other clusters can be added through bounded smoke builds rather than
requiring a full matrix before the approach can be discussed.

## Review Questions

The joint review must answer:

1. Does the v4.x approach satisfy the required outcomes without depending on
   undocumented developer state?
2. Can this refactored source tree adopt it without changing scientific
   behavior or weakening its validation gates?
3. Does it define a sensible public/private compilation boundary for the
   header-dominant tree?
4. Are dependency and generated-version identities strong enough for later
   reduction provenance?
5. Which current CMake options, cluster commands, or paths require migration?
6. Is any remaining build limitation material enough to block integration, or
   should it remain explicit retained debt?

## Adoption Paths

Choose one path after reviewing an actual v4.x implementation:

### Adopt

Use the v4.x approach substantially as written when it satisfies the required
outcomes and fits this tree without broad compatibility work.

### Adapt

Port a bounded part when the core model is sound but repository layout,
generated inputs, test registration, or cluster configuration differs. Keep
the adaptation as one coherent integration project rather than parallel build
systems.

### Defer

Retain the existing build temporarily when the proposal is incomplete or
incompatible. Record the failed requirements and the next owner/trigger; do
not replace a known working operational build with an unvalidated partial
modernization.

## Integration Sequence

Once an approach is selected:

1. preserve the exact validated pre-build tree and its history;
2. apply the build integration as a bounded change;
3. perform a clean local configure/build and the full local gate;
4. measure clean, no-op, and representative incremental builds;
5. compile the exact commit on Unity and run a point smoke reduction;
6. resolve any cluster-only issue at a new commit;
7. freeze one candidate SHA; and
8. run the point, OOF, science, and Beammap same-SHA validation matrix defined
   by the Phase 5 plan.

No documentation-only or unrelated code commits are added while that final
same-SHA matrix is running.

## Current Disposition

The first v4.x Conan 2 implementation became available on 2026-07-26 and was
reviewed at the exact revisions recorded in
`TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`. The project selected the
**Adapt** path. The Tula/Conan 2 architecture is accepted as the foundation,
but the current Citlali milestone builds only a five-source static library and
does not include the production CLI, full dependency/API surface, generated
provenance, or complete validation gates.

Existing build behavior remains unchanged while the bounded adaptation is
designed and tested. Compilation-dependent criteria 6, 7, and 10 in the Phase
4 closeout census are now active integration work rather than unavailable
external evidence; they remain open until the full application and Unity lane
pass the recorded gates.
