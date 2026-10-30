# POINT compatibility and table handoff — 2026-10-30

## Program adherence and prior-work recovery

Owner Grant explicitly authorized fixing pointing compatibility gaps and implementing
POINT table handoff on the combined candidate, preserving OG correction behavior,
keeping the wing study paused, and running no reductions. One active spine increment;
risk tier 2. Engineering, successor, review/conformance governance read; September 30
standing owner amendment applies. Canonical e32c4584dec07617d6caa857d39bf13047b6c4b0
unchanged. Clean base b090da0458271c618568802fde1cf5c39f5dd5d2; application merge
c83375dc030095821999a0e0ff435f13bf82b914. Branch codex/point-handoff-20261030;
worktree /Users/gwilson/.codex/worktrees/point-handoff-20261030/citlali-refactor.
Initial staged/unstaged/untracked state empty.

Adopt OG behavior recovered in pointing evidence commit
28061144d2018d2f57cd6c47ee95990bae186c19,
validation/pointing_og_compatibility_2026-10-30/REPORT.md, independently reviewed
closure 6969dcf8168bcffbdfdbf7ebe56a22730545e086. Cite connected-pilot plan
323f1fba1fead8fe3a422f0368d3362cc5a9ac7e for sequencing, not uncertainty authority.
Original fork 376e002238b1f49aeced8a3f33e8742db141634b; preserved OG
ffc6b9070f4744f9778f3db71cdc468846d1da89; V2 POINT
c31a60a0b74a7149d03d542966d6e35b77b8091c; V2 SCIENCE
187df04b21e942701cf41e6d9c50883922fd65aa. Reuse actual operational TolTECA
2791e6a1e6349ad1d3ac549a648f41cbc51b98c7 and TolProj
6309a5a852a0412cee867992aff01341a56b0ebf as read-only consumer fixtures.
Defer source-wing, temporal-change, photometric uncertainty and SCIENCE policy
studies. No fresh scientific derivation.

## Owned scope and boundaries

YAML adapter owns legacy normalization; CAL ingress owns typed pointing model
construction and OG epoch conversion; existing AST owns coordinates unchanged.
POINT publication owns one-way ECSV adapter for final accepted measurements,
observation metadata and explicit unavailable errors. TolTECA remains correction
composer: measured signed x/y arcsec, User/Paddle radians, observation and MJD.
No negated measurements, duplicate corrections or new Engine state. Require
qualified all-array fits before discoverable table; otherwise preserve maps and
explicit unavailability. Conditional sigma_A must not become amp_err, and no
photometric eligibility is granted. Actual calibration-reader rejection with
unavailable errors is required. If unsafe, stop affected publication and record
the sibling prerequisite. Already-applied POINT corrections must not silently
become uncorrected offsets: withhold unsupported contexts with a clear reason.

Paths: configured POINT config/input/product orchestration, CAL offset ingress,
focused tests, runtime inventory if needed, bounded docs/evidence. Preserve
physical equations, native schedule, weights, optical cut, FRUIT settings, grid,
original inputs, reductions and runtimes. No sibling edits, reductions, saved-map
refits, Unity, pushes, canonical integration, activation, extra studies or cleanup.

## Gates and stop conditions

Focused: single-date constant, observation-span, fractional-MJD OG parity,
invalid/collapsed support; signed synthetic all-array ECSV, metadata/units,
unavailable fits/errors, actual pinned TolTECA roundtrip and TolProj rejection.
Broad: configured Python suite, config preflight --require-all, baseline tests,
affected CTest. Local fallback build is supplemental; no supported Spack or
observational qualification claimed. Small metadata only; no detector-loop or
heavy provenance work. Stop affected work on new scientific meaning, unresolved
uncertainty/support or sibling dependency. One fresh-context exact-SHA review.
Coherent local commits authorized. Integration/push/activation/cleanup separate.

## In-scope reassessment

Observation-span input requires the realized common-grid endpoints, already owned
by processing preparation. Pass that existing support into CAL and save two
scalars plus the OG epoch-conversion marker with its receipt; saved MAP validates
that declaration. This is a bounded cross-stage representation of existing
pointing authority, not new sampling or extrapolation. Old variable-offset CAL
receipts are refused rather than silently retimed; old constant receipts remain
compatible. Tests exercise the concrete saved-receipt adapter. No sibling change
was needed: actual TolProj rejects unavailable amp_err.
