# Review stage 1 repairs — 2026-11-03

Status: bounded build correction prepared after an owner-run compilation
failure; corrected-candidate execution and independent review remain pending.
This is not an integration or qualification record.

## Work order and authority

Grant approved staged repairs following the independent static review of clean
mainline `fefdd6b7bde5b272ed664561cf996f805cf6852b`, then explicitly approved
one isolated branch/worktree after lifting the earlier review-only restriction:

- Branch: `codex/review-stage1-20261103`.
- Checkout: `/Users/gwilson/.codex/visualizations/2026/11/03/01a19edc-cf82-74b0-a34b-01547e7314a9/citlali-stage1-20261103/source`.
- Owner: Grant Wilson; implementation preparation: Codex.
- Execution owner: **Grant alone**, for every build, test, executable probe,
  benchmark, reduction and validation runner on Mac and Unity.
- WIP: one bounded repair increment; no scientific-module probe. Existing
  pointing/handoff work and legacy checkouts remain untouched.
- Risk: Tier 2 because the repair changes published fit metadata.

The engineering, successor and review/conformance governance documents were
read. Their accepted exact digests and incorporation identities are recorded
in [the integration ledger](INTEGRATION_LEDGER.md). Sequencing follows
[current state](CURRENT_STATE.md) and [refactor status](REFACTOR_STATUS.md).
The existing [architecture](ARCHITECTURE.md) and
[scientific conventions](SCIENTIFIC_CONVENTIONS.md) remain in force.

The two owner clarifications governing this repair are:

1. Repeated NetCDF header elements are valid. Read the first element, including
   when later values differ; do not require singleton shape or agreement.
2. Pointing coaddition is optional engineering/science use. Preserve the maps,
   mark an unfitted coadd's fit metadata invalid, and omit its fit measurements.
   Observation fit results remain unchanged.

## Included behavior and ownership

`citlali::cli::input_bindings::read_netcdf_scalar` remains the shared native
application input owner. A true scalar uses its scalar read; an array checks
that every dimension is nonempty and reads exactly its first element. Missing
and empty headers fail before a value is used. Existing numeric conversion,
units, header identities and caller-specific semantic validation are retained.
No array payload is allocated to read a repeated header.

The retained `Pointing` FITS writer now receives the existing `MapType` through
its output call chain. Raw/filtered observation products retain their fit
table and validity. Raw/filtered coadds preserve `POINTING.array`, publish
`POINTING.fit_enabled=0` and `POINTING.fit_valid=0`, and omit numerical fit
values, errors and significance fields. No observation fit table is allocated
for coadds, and the final observation's `fit_valid` is not read for them.
The fit flags describe the particular product; the requested fit configuration
is not mutated. No new fit, map arithmetic, public cross-stage state, field name
or generic framework is introduced.

## Gates and evidence limits

| Gate | Required evidence | Current state |
| --- | --- | --- |
| Static diff/call-site inspection | Scope, typed output routing, no numerical changes | Writer inspection completed; not independent acceptance |
| NetCDF input tests | Scalars, singleton, repeated/differing headers, multidimensional first value, empty/missing headers, actual timestamp-slice caller | Four new tests prepared; unexecuted |
| Pointing FITS tests | Persisted raw/filtered observation and coadd headers, invalid coadd flags, omitted measurements, retained array identity and unchanged image data | Four new tests prepared; unexecuted |
| Existing metadata/fit-metric tests | No regression to observation fit metrics or TOD metadata | Pending |
| Native Mac Spack build | Fresh candidate build and matching executable revision | First candidate failed compilation; corrected-candidate rerun pending |
| Broad local gates | Enabled CTest suite, baseline-tool tests, config preflight with `--require-all` | Pending |
| Independent review | Fresh-context read-only review of final exact candidate SHA and the three conformance categories | Pending |

The FITS tests call the production fit-header writer with real FITS persistence.
They do not run the entire retained reduction, accumulate observations, or
exercise the full `Pointing::output` diagnostic sequence. Whole-route coadd
product reproduction remains pending owner disposition before qualification.
Configured successor POINT acceptance is not established by these tests.

The intended local environment is the preserved Mac `macos-llvm20` Spack
realization, with a new candidate-owned build directory. Its existing lock is
`00964e0c57c1df96b0ddc3f6d214bef1c1ab8f59cc4e5c4da653cefbcb915238`.
No dependency installation, environment redesign, runtime replacement or Unity
access is part of this work. There is no hot-loop change or performance claim.
The first owner's configure/build logs were inspected as recorded below. No
successful application run or zero-unexpected-error outcome is claimed.

## First owner-run attempt and bounded reassessment

Grant ran `owner_run.py all` against clean candidate
`b2464ee6b3beebfc14c20def30cf8b8287b4555f`. The preserved attempt is
`../owner-run-20261103T131717.670695Z` relative to this checkout, with its original
`binding.json`, command receipts and logs. Configure exited 0; build exited 1
while compiling `tests/test_output_schema.cpp`. Version verification and all
test gates were not reached.

- `configure.log` SHA-256: `b9c4a3921c78ef163fc740c8257b83d2778b32dced0343f91b27820f10e533f8`.
- `build-and-version.log` SHA-256: `7d65d04661ed29110eb63c9ee2a9aec4c23c08f044b92bebecdf7ca6a5e1b346`.

The compilation trigger was `pointing.h` declaring two methods using
`FruitLoopWeightPolicy` before its defining header was included. The four
compiler errors are the two unknown-type errors and their consequent declaration
mismatches. The new FITS test exposed this existing include-order dependency.
The configure log also contains a CMake CMP0144 developer warning about
`BOOST_ROOT`; it is not the compile failure and is not suppressed or repaired here.

Disposition: **repair within scope**. Add the direct
`timestream_run_context.h` include to the existing Pointing owner and include
`pointing.h` first in the test translation unit, so caller include ordering
cannot mask the dependency. This satisfies the existing public-header rule;
it does not change scientific authority, numerical behavior, build realization,
or the stage-1 outcome. No other engine header or ownership area is changed.
The same owner-run gates remain required against the new exact candidate; no
prior pass transfers. Codex performs only static inspection and preparation.

## Conformance and remaining boundaries

- Scientific/behavioral: the prepared changes implement the two owner
  clarifications. Runtime conformance is pending. No scientific thresholds or
  algorithms change.
- Architecture/ownership: fixes remain in the shared input service and existing
  pointing product writer. Static writer assessment only; independent review
  is pending.
- Repository/evidence: one isolated candidate starts at the exact clean review
  base. Mainline, pointing work, Citlali 4.x/5.x, stashes, backups and qualified
  runtimes remain preserved. Final candidate SHA/tree, per-file digests and
  owner-run commands are bound in the external packet beside this checkout.

The NetCDF repair is a separate commit from the pointing metadata repair.
Stop and reassess if the base moves, unrelated dirt appears, scope needs a new
scientific policy, or another ownership area must change. Do not infer a pass
from prepared tests. Independent review and owner acceptance are required before
integration. There is no integration, push, deployment, activation or cleanup
authority.

Output naming/ownership at setup and configuration/reporting improvements are
later stages, pending the first stage's owner-run checks. The RTC-only/CAL
preparation question remains an investigation; it is not repaired here.
