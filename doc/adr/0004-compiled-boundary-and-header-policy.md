# ADR 0004: Compiled Boundaries And Header Policy

- **Status:** Accepted
- **Recorded:** 2026-07-16
- **Decision owners:** Citlali project owner and engineering

## Context

Citlali remains header-dominant. Mature templates and numerical loops coexist
with many contextual `engine/detail` fragments, so the CLI compiles a broad
implementation graph. Earlier structural work also demonstrated that splitting
files by line count can increase navigation and include complexity without
creating ownership or reducing build time.

The first bounded compiled extraction moved timestream enum tables and
parse/format definitions from a 946-line header into
`src/citlali/core/config/timestream_enums.cpp`. The header fell to 712 lines,
but one immediate CLI compile pair was 62.4 versus 63.7 seconds. That is neutral
evidence, not a demonstrated speedup.

Compilation-side work is now deferred until TolTECA's revised C++ integration
approach is understood.

## Decision

Public boundary headers compile independently and link from multiple
translation units. Contextual implementation fragments are private mechanics
even where the transitional layout still places them below `include/`; they do
not define reusable public interfaces.

Keep templates and measured hot loops in headers when required. Move cold,
non-template implementation into `.cpp` only when the candidate forms a
coherent subsystem or materially reduces a measured dependency cost. Every
such move records before/after header closure or compile evidence, local
header/link tests, runtime impact, and affected-mode product validation.

Do not split files solely to reduce line count. A new file or boundary needs a
named owner, contract, test seam, dependency benefit, or measured build
benefit. Do not introduce allocation, filesystem access, YAML parsing, string
lookup, logging, type erasure, or virtual dispatch into established hot loops
without evidence.

No additional CMake, dependency, preset, CI-build, install/export, cluster
helper, full-header-matrix, or broad `.cpp` boundary work proceeds during the
current deferral.

## Consequences

- The enum extraction remains accepted as a coherent cold boundary with no
  product/runtime regression, but is not advertised as faster.
- The current header-heavy physical graph is explicit retained debt.
- Future compilation work must be designed against the intended TolTECA build
  topology rather than the current workstation configuration alone.
- Contextual fragments may be consolidated or privatized when that creates an
  enforceable boundary; textual churn is insufficient.
- Runtime and scientific validation remain mandatory for a build-only-looking
  extraction because template instantiation and linkage can alter behavior.

## Rejected Alternatives

- **Move all large headers to `.cpp` mechanically:** templates, private access,
  and hot loops make this risky and do not guarantee faster builds.
- **One file per helper:** increases fragmentation without ownership.
- **Treat one compile pair as a performance conclusion:** the observed pair did
  not show an improvement.
- **Modernize CMake before the TolTECA direction is known:** risks replacing
  infrastructure twice.

## Supersession

Review this ADR when the TolTECA integration model is available. A successor
may define the true public/private target graph, pinned dependencies, supported
CI lane, and install scope. Evidence-driven header and hot-path rules remain
unless that successor explicitly replaces them.

## Evidence

- [`../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md`](../PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md)
- [`../PHASE4_CLOSEOUT_CENSUS_2026-07-16.md`](../PHASE4_CLOSEOUT_CENSUS_2026-07-16.md)
- [`../ARCHITECTURE.md`](../ARCHITECTURE.md), header and compiled-code policy
- `src/citlali/core/config/timestream_enums.cpp`
