# MAP successor first forward pass — 2026-09-21

## Work order and baseline

Owner: Grant Wilson. Current request authorizes implementation and testing of
one ordinary forward MAP pass from the existing observed-entry ALS PTC products.
FRUIT is deferred. This occupies the successor spine slot while the previous
timestream development is parked. Risk tier 2; read-only independent review is
required by the effective `doc/governance/REVIEW_AND_CONFORMANCE.md`.
All three governance documents were read. No integration, push, cleanup,
canonical ref movement, or Unity access is authorized.

Implementation branch: `codex/map-successor-first-forward-001`.
Worktree: `/private/tmp/citlali-map-successor-first-forward-001`.
Exact baseline: `686c280037762450b1870308c920e9191216dc16`, tree
`9aee4c03ec7ebd8cb526c6560c607956e091974f`.
The baseline is the parked reporting branch, not a new numerical baseline.
The 11-network Unity numerical runtime is
`5cd3e28396a08917986f3ec73314a69d3605bd0d`.
Canonical ref was locally observed at `e32c4584dec07617d6caa857d39bf13047b6c4b0`;
this is not a remote freshness claim. The new worktree began clean. The dirty
ALIGN checkout `/Users/gwilson/GitHub/citlali-refactor` is untouched.

## Authority and requirement-to-code-to-test worklist

Normative science is SCI-MAP v0.2/r0.4, accepted content
`c8081c4ee4dc1c5132c3030a351073205fc3142a`, shared-core SHA256
`3b81ea8180dcc3f5f37964ac1177feb417fef6a1cdcc590b33f9431b54bc1e31`.
The frozen common source and Engineering Conformance Specification live under
`doc/scientific_contracts/packages/SCI-MAP/v0.2/revisions/r0.4/`.
Later freeze, registration, and activation records govern historical candidate
labels: `SCIENTIFIC_OWNER_FREEZE_R0.4.md`,
`acceptance/SCI_MAP_PROFILE_ADMISSION_2026-09-08/REGISTRATION_AND_SOURCE_BINDING.md`,
and `acceptance/SCI_MAP_PROFILE_ACTIVATION_2026-09-09/ACTIVATION_RECORD.md`.
The five administrative generation identities were registered separately from
generic science and activated on September 9. Object facts/evaluations still
must be supplied. No candidate is accepted merely by accepting its author packet.

| Requirements | Settled method / producer | Implementation and evidence work |
|---|---|---|
| 002–006, 010 | Actual positive-rank PTC, exact occurrences, explicit PTC-owned coefficients; separate QC and MAP/VAL decision | Repair publication seam; reject legacy CAL and incomplete saved products. Existing ALS and comparator remain unchanged. |
| 005, 015, 044–045 | AST exact same RTC sample and full parent; MAP half-open one-hot placement | `forward_map` numerical placement and edge tests; actual frame/pointing publication is still a prerequisite. |
| 011–016, 018, 031–035 | N/Q signed signal, exact output rows, no centering; two support thresholds | Immutable `ForwardMapOperator`, hand calculations, constants/deltas, excluded nonfinite values, small-N selector, empty support and overflow tests. |
| 007, 025–030 | Distinct incidence/support and original-footprint seconds, deduplicated originals at their own AST coordinates | Separate original-footprint accumulation primitive and union/boundary tests. Missing original authority must remain unavailable, never sample-count exposure. |
| 008, 016–024, 049–051 | Known MAP-local operator; same operator for linear companions; upstream response/covariance may be unavailable | Preserve full numerical operator/domain and immutable parent. No beam-as-response, significance, noise substitute, or response-family conflation. |
| 009, 043, 046–048, 050 | Lifecycle, complete bundle, required publication/reload, truthful failure | Pending connected producer and bundle work; kernel tests are not bundle or package conformance. |
| 036–042 | Equal-observation compatible coadd with atomic admission | Deferred; not required for the first single observation. |
| 052 | Ordinary MAP/JINC/FLT/NOI/FRUIT and mode claims are distinct | Preserve explicit unavailable downstream roles, no feedback implementation. |

Owner selection during this task: **use the approved uniform family for this
pass**, `SCI-PTC:uniform_constant@draft-0.1`, dimensionless gamma=1. This is not
inverse variance, a mode default, or a uniform coadd coefficient. PTC's actual
publication and `uniform_coefficient_handoff@draft-0.1` decision are still needed.
The existing `SCI_PTC_OBSERVED_ENTRY_OWNER_BINDING_2026-09-18.md`, singleton-opacity
amendment, and partial-completion amendment remain upstream authority.

## Observed input gaps and reassessment

The September 21 result directory is a diagnostic report/manifest, not a PTC
numerical handoff. Its restart record explicitly says raw arrays/NPZ remain on
Unity. Earlier local full networks 0 and 7 have actual cleaned PTC files; bounded
network 11 products also exist. These are candidate input evidence, not admitted
maps. The full campaign covers 11 supplied networks, excludes absent networks
6/10, and preserves network-4 scan121 fit failure (387,422 eligible samples,
3,173.761 detector-seconds). A diagnostic PASS does not restore these samples.

Current PTC receipt v2 contains cleaned values, slots, causes, fits and file
hashes, but lacks a completed MAP coefficient/QC handoff. CAL/AST computes
same-RTC-grid pointing in memory but does not persist those coordinates. CAL's
frame label is only `J2000-radec-gnomonic-tangent`. The input adapter checks
Epoch=2000 and CoordSys=0; this does not establish ICRS versus FK5 or apparent
versus catalog coordinates. SCI-AST-041 and SCI-MAP-044 prohibit inference.
The owner was asked for the authoritative SourceRaAct/SourceDecAct definition.
Continue numerical implementation and independent input assessment while that
required authority remains pending; do not publish an invented WCS.

CAL uses `mJy/nominal-beam`, consistent with its accepted r0.5 Q03 definition.
The MAP label `mJy/beam` inherits that convention, but exact originating nominal
beam identity must still be recovered from the admitted APT source; no numerical
conversion is implied by spelling. Recovery remains pending.

## Downstream graph and deferred decisions

The existing horizontal audit graph and its accepted shared-convention repair
remain the dependency worklist. The September 8–9 MAP authority supersedes its
older MAP binding gates. PTC + same-sample AST -> base MAP -> independently
requested response/covariance roles or coadd. MAP and JINC are alternatives.
FLT-FIXED owns a same-grid deterministic transform; FLT-MATCHED owns a separately
normalized template-amplitude estimator. NOI owns detector-sign randomization,
conditional marginal second moment and standardized signal, never MAP coefficients
or significance. No suitable realized NOI ensemble/filter request is currently
supplied, and these are not base-role prerequisites.

FRUIT's numerical feedback model, source inference, recurrence and stopping remain
outside this task. The requested September 9 directive filename was not found in
known worktrees or tracked history. Independent review recovered the September 8
conditional generic freeze in sibling worktree `346d`, core v0.1/r0.4, preserving
numerical-method unavailability. This is a disclosed recovery gap, not permission
to invent missing FRUIT policy. Future interfaces must retain immutable upstream
parents and allow PTC relearning and replacement inference from reconstructed total.
No thresholds, damping, flux floors or terminal selection are chosen here.

## Gates and current disposition

Use the established supplemental AppleClang/Homebrew build locally, with frozen
baseline dependency source paths. It is not the accepted Unity/Spack environment.
Focused C++ operator/header tests, PTC publication/admission tests, full CTest,
baseline-tool tests and full config preflight are planned. Numerical kernel tests
use independent hand expected values; exact deterministic order is asserted for
permuted member input and concurrent read-only application. No real-data MAP,
complete bundle, observational fidelity or package conformance is claimed yet.

Task evidence root: `/private/tmp/citlali-map-forward-evidence-20260921`.
Build root: `/private/tmp/citlali-map-forward-build-20260921`.

## Recovery result and bounded stop

Independent preimplementation review confirms that SCI-MAP-007 permits missing
original-footprint exposure authority to be explicitly unavailable, and the base
role permits missing complete upstream response/covariance. They are not the
reason this first map is blocked. The missing complete AST frame and originating
nominal-beam identity are mandatory and cannot be responsibly supplied by code.
The later PTC observed-entry/nominal-beam unit spelling is consistent with CAL's
accepted Q03; no scale conversion is performed.

The selected APT manifest is SHA256
`cb83f1e1659454a1334b942b19e0b13b44ab757fc8722e13c3a3f0930a2e973c`.
Its baseline is beammap observation 148670, revision `v4.0.0-3753-gc31a60a0`.
The baseline runtime-provenance reference was recovered at
`/Users/gwilson/work_toltec/local_data/citlali-validation/v2/beammaps/3c273/reduced/redu00/runtime_provenance.yaml`,
SHA256 `0e91a375ef192786c04254d6bb84d100c7273e6b4d2892746abc0191ed87f366`.
It and adjacent `beammap_provenance.yaml` supply execution/flux information,
not a fixed nominal-beam identity. Fitted APT beam widths are not a substitute.

The saved effective configuration, SHA256
`69e93f75b71209772ff080b074c4f3f91f99b2d6397b2f3f3ed24852f084b1e2`,
requests JINC, 2-arcsecond pixels, coverage cut 0.5, radec, with zero placeholders
for shape/reference coordinates. The user's current work order selects ordinary
MAP; it does not alter the saved upstream request. Reuse pixel scale and coverage
cut in a separately identified MAP request after AST resolves an actual grid.
These values are recorded source choices, not a new universal default.

**Completed changed scope:** compiled `mapmaking/forward_map.h/.cpp`, independent
numerical tests and header isolation, both production source graphs, and a
bounded saved-PTC input inspector with negative tests. The numerical primitive
accepts a caller-established exact admitted input domain; it is deliberately
not a PTC/VAL admission issuer or complete map bundle. Its unequal-coefficient
fixtures test general positive arithmetic and do not register another science
coefficient family. Concurrent read-only applications are supported; shared
mutable parallel accumulation is not introduced.

**Not complete:** first-pass successor MAP CLI route, typed PTC uniform
publication/QC, MAP/VAL object decisions, actual AST/WCS carrier publication,
logical/persisted MAP bundle and reload, real maps/coverage images, source response,
MAP runtime/memory costs, coadd and package-wide conformance. No science map is
claimed from the numerical core or the real input inspection. The earlier plan
to publish the new PTC handoff remains a next implementation item, not a completed
artifact. No source changes were made to upstream numerical owners.

## Reproduction and next implementation steps

Use the task's established supplemental build and local venv:

```sh
cmake --build /private/tmp/citlali-map-forward-build-20260921 --target citlali_cli citlali_forward_map_test citlali_safety_test -j 8
ctest --test-dir /private/tmp/citlali-map-forward-build-20260921 -R forward_map --output-on-failure
cd /private/tmp/citlali-map-successor-first-forward-001
/Users/gwilson/tolteca/bin/python -m unittest discover -s tools/timestream_successor -p test_inspect_map_inputs.py -v
/Users/gwilson/tolteca/bin/python tools/config/run_config_preflight.py --require-all
/Users/gwilson/tolteca/bin/python -m unittest discover -s tools/baseline -p 'test_*.py'
```

The build directory is linked as this worktree's untracked `build` locator for
existing tests that require `build/bin/citlali`. Configure arguments and actual
compiler/dependency cache are in the evidence/build roots. The initial configure
attempts omitted old dependency-selection switches and the CMake compatibility
floor; their failures are retained. Final configuration uses the existing frozen
dependency sources, Homebrew/AppleClang and CMake policy minimum 3.5; no dependency
source or build-system policy was changed. A fixture initializer-list type issue
was fixed before focused tests passed. The first whole `check` attempt exposed
its existing missing `citlali_safety_test` build dependency; build that target
explicitly before rerunning CTest. Required missing tests are not counted as pass.

Reproduce a real input inspection with a new output filename:

```sh
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-map-successor-first-forward-001/tools/timestream_successor/inspect_map_inputs.py /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-ready-ae4842e-20260919/local-evidence/network0-full/products/network0 --output /private/tmp/map-network0-new-inspection.json
```

Expected exit is **2**, successful numerical inspection with **MAP readiness
blocked**. It creates JSON evidence, never a MAP product. It does not rerun RTC,
CAL, PTC, choose modes, estimate noise or connect to Unity. No reproducible MAP
execution command is available yet; inventing one would hide the missing route.

Resume after the two actual source facts are supplied:

1. Record authoritative SourceRaAct/SourceDecAct frame semantics and originating
   nominal-beam identity through their AST/CAL owners, preserving numerical values.
2. Publish exact PTC completed segments and selected uniform scalar/support
   realization, with faithful PTC/VAL handoff. Preserve rejected intervals and
   absent networks; do not infer publication from diagnostic reports.
3. Carry same-sample AST coordinates and resolve the target grid/complete frame;
   leave unavailable original-footprint exposure explicitly unavailable unless
   its separate exact original-coordinate carrier is supplied.
4. Implement MAP-authored admission through VAL, then call the tested numerical
   owner; publish its full role-bound bundle/operator and test serialization/reload.
5. Exercise the verified local networks as a smoke pass and the available
   observation networks separately by array. If Unity products are needed, prepare
   a bounded owner-run MAP-only packet. Do not repeat upstream science unnecessarily.
6. Visually inspect actual maps and coverage, measure MAP costs and source response,
   obtain exact-candidate review, and only then return grounded FRUIT decisions.

The two current owner questions are source facts required by MAP, not FRUIT
policy. FRUIT thresholds, inference and recurrence discussion remains deferred
until actual maps exist, as requested.

### Actual inspected input coverage and cost

Source network identities, not directory nicknames, govern these results.
The full a2000 input is **network 12**, recovered from its producer receipt;
the separate failed bounded fixture is network 11. Network 12 was inspected
because this saved full a2000 product exists, not because of its 11-Hz residual.
The numerical source revisions remain attached to each individual report.

| Saved source | Array | Retained entries | Failed fits | Final warm inspection wall |
|---|---|---:|---:|---:|
| full network 0 | a1100 | 74,946,311 | 0 | 3.41 s |
| full network 7 | a1400 | 27,672,182 | 0 | 2.28 s |
| full network 12 | a2000 | 27,157,906 | 0 | 2.04 s |
| bounded network 11 test | a2000 | 0 | 124 | 0.45 s |

These are **input-inspection costs**, not MAP costs or repeated reductions.
The first inspections took 5.4–7.0 seconds for the full inputs; the faster final
reads reused the filesystem cache. Process peak RSS, CPU time, segment working
arrays, exact sources and report digests are in
`handoff/evidence/MAP_SUCCESSOR_FIRST_FORWARD_001_2026-09-21.json`.
All three full inputs remain MAP-blocked despite intact PTC components.

### Verified implementation gates

Final supplemental CLI/library build and public-header isolation pass. Final
CTest: **1,557 runnable tests pass / 1,558 registered**, with the previously
established `MapFitterLifecycle.ExactProductSequence` disabled; no newly skipped
required test and no test failure. This includes all eleven new numerical cases.
Eight input-inspection tests, 207 baseline-tool tests and the full config
preflight pass. Final logs/digests are recorded in the compact JSON evidence.
The native source was repaired to reject coefficient-product or normalization
underflow to false measured zero as well as overflow; the final CTest run
includes the explicit tiny-coefficient fixture.

Scope-level disposition: numerical arithmetic and negative input-inspection
controls pass their implemented tests; complete SCI-MAP conformance and real-data
MAP are **blocked/not assessed**, not pass. Architecture keeps the owner below
orchestration, with no Engine/legacy/upstream changes. Repository work is confined
to one owned branch/worktree; generated build locator is untracked. Exact-candidate
independent review follows the implementation commit and is separately bound.


### Initial independent review and bounded repair

Independent fresh-context review of implementation candidate
`6fd2d8079867bae14faded35e650dd6505813f2a`, tree
`e6e787626aa7a9e58b626207b9549ce6c38881c1`, required a numerical repair:
nonzero underflow rounding could change a constant (coefficient `denorm_min()`,
input 0.75 returned 1). The approved uniform family was unaffected, but the
primitive's declared general positive-coefficient domain was affected.
Architecture/ownership and repository/evidence dispositions passed with the
recorded partial-scope limitations. The reviewer reproduced all then-current
9 numerical and 8 inspector cases and verified the saved gate/report hashes.

The repair scales the product relation to normal mantissas and checks its FMA
residual whenever a product or quotient is at/below the minimum normal value.
It detects inexact subnormal rounding, including rounding upward to minimum
normal, without relying on wider `long double`. The same guard covers weighted
signal, normalized signal and support thresholds. Exact subnormal arithmetic
and true zero remain valid. Two added cases cover signed constant failures,
normalization, support thresholds and exact-value preservation. Rebuilt CLI,
public header and all 1,557 runnable CTests pass; the established lifecycle test
remains disabled. Exact-SHA review of this repaired candidate follows its commit.
