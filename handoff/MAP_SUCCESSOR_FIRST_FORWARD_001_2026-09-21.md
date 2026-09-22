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


### Final independent numerical-candidate disposition

Repaired implementation candidate **`b319544483645b3591fee70aa5d4ea2ea12e3569`**,
tree `48bea1175ca707e599b0976d351c0b9419fdd1e2`, received independent
**pass with recorded limitations**. The prior major finding is resolved;
no remaining blocker, major or minor finding was identified in the implemented
scope. Scientific/behavioral disposition passes the numerical primitives and
bounded inspector. Architecture/ownership and repository/evidence dispositions
pass with the existing partial-scope limitations. The reviewer independently
reproduced 11 numerical and 8 inspector tests, verified repair-log hashes, and
confirmed the refreshed CLI revision `gb31954448` matches this exact candidate.
The recorded supplemental dependency identity remains `kids 04088da-dirty`;
this is not a Spack or Unity validation claim.

This review does not approve a complete MAP path, missing PTC/QC publication,
MAP/VAL admission, frame/beam authority, bundle serialization/reload, real maps,
response measurements, production readiness or integration. The subsequent
closeout record changes documentation/evidence only; numerical implementation
and test sources remain those of the reviewed candidate. Final closeout SHA,
CLI identity and read-only documentation verification are also retained in
`/private/tmp/citlali-map-forward-evidence-20260921/final-closeout.json`.

The milestone remains blocked and incomplete. Required owner/source facts are
(1) the authoritative celestial reference system and catalog/apparent semantics
of `SourceRaAct/SourceDecAct`, and (2) the originating fixed nominal-beam
definition for the APT calibration inherited from beammap 148670/redu00,
revision `c31a60a0`. The explicit uniform-family selection is already settled.
After those facts are bound through AST/CAL, the producer/admission/bundle work
listed above and actual map execution remain required. No FRUIT decisions are
requested before maps exist.


### Owner follow-up: nominal beams and J2000 direction — 2026-09-21

The owner supplied this explicit direction after the reviewed increment:

> The celestial frame should be J2000. Take the fixed nominal beams to be 5,
> 7.5, and 10 arcseconds FWHM for a1100, a1400, and a2000 respectively.
> Telescope deformations and focus will lead to different values measured,
> of course.

For this first forward pass the selected fixed nominal calibration convention
is therefore a1100 = 5 arcsec FWHM, a1400 = 7.5 arcsec FWHM, and a2000 =
10 arcsec FWHM. This owner declaration resolves the missing nominal-beam
selection/definition for the pass; it is not a claim that these values were
recovered from the historical APT metadata. Measured widths, focus/deformation
effects, and realized processing response remain separate. No signal rescaling,
PSF measurement, Gaussian response model, or transfer-function qualification
is implied. The upstream calibration values and PTC numerical treatment remain
unchanged. Binding this convention into the CAL/MAP product metadata remains
engineering work; the old receipts and their original inspection findings
remain immutable.

The owner also explicitly selected J2000. A narrow clarification is pending
for its machine-readable WCS identity: whether this means FK5/equinox J2000.0
and whether SourceRaAct/SourceDecAct already carry that frame, rather than ICRS
or apparent place. This does not reopen the chosen J2000 direction. The exact
remaining distinction follows SCI-AST-REQ-041: “AST shall not equate J2000,
FK5, ICRS, apparent, observed, or unrefracted frames by name or default.” No
RADESYS value or coordinate transformation has yet been invented. The question
was sent while the PTC coefficient/publication boundary was inspected.

Reassessment: continue within the same owned worktree and original MAP work
order. The three governance documents were reread; this follow-up records owner
authority only and changes no numerical source. The uniform family remains
explicitly selected. Required implementation work and real-data gates remain as
listed above. The earlier “two source facts missing” disposition is historical:
nominal beams are now selected, with only the narrower frame identity question
remaining as an owner clarification. No actual MAP product exists yet.


### Owner frame confirmation and resumed implementation — 2026-09-22

The owner answered “yes, that's correct” to the explicit FK5/equinox J2000.0
question, including that SourceRaAct/SourceDecAct already carry this frame.
This resolves the remaining scientific input clarification; no frame transform
is requested. The nominal beams and uniform coefficient selection above remain
authoritative. Resume the original MAP work order on the same isolated branch.
The resumed base is `fd3c1ff151c766cc44cafec247f99dc930447f66`; only the owned
untracked build locator was present. No new branch, upstream treatment change,
Unity access, push, or FRUIT work is introduced.

The three governing engineering/review documents have been read. This remains
the Tier-2 spine increment: PTC owns completed segment/coefficient publication;
AST owns exact RTC-sample pointing; MAP owns admission, plan, arithmetic and
bundle; application adapters own file parsing and orchestration. Saved-product
reuse must prove the existing sample relation instead of matching timestamps.
New schema/publication seams require focused negative and round-trip tests,
whole active local gates, real-data measurements, and independent exact-SHA
review. Local AppleClang/Homebrew evidence remains supplemental.


### Connected saved-PTC implementation candidate — 2026-09-22

The original owner work order now has an actual native `terminal: map` route.
`prepare_forward_map.py` binds immutable original request and completed PTC
receipt digests, explicit owner selection, pixel scale and coverage cut. The
compiled application adapter verifies final/learning/RTC apply schedules and
PTC→CAL→RTC ancestry, then publishes complete segment PTC and selected uniform
coefficient handoffs. PTC owns gamma=1; VAL owns its named-use QC and separate
sealed MAP decisions; MAP authors admission predicates; AST owns reconstructed
same-slot detector pointing and WCS. The existing AST evaluator is shared with
the live route. No Engine state or new upstream treatment is introduced.

Every plan column block binds publication, coefficient payload, coefficient QC,
AST payload/binding, exact MAP evaluation and packed eligibility bits. Decision
identities use the complete immutable evaluation-file digest with the exact
active Registry/register, allocation/manifest, shared core and boundary tuple.
CAL causes remain available through exact source links; MAP rejection never
rewrites PTC support. The required FITS/sidecar bundle preserves each array,
signed N/Q, unsupported NaNs, normalization/science support, hits, output rows
and the exact sparse local operator. Nominal calibration beams are not PSFs.
Original-footprint exposure, covariance and full upstream response are explicitly
unavailable; no NOI/FLT product was requested and no FRUIT feedback is executed.

The reader independently checks every required handoff and evaluates exact CAL
causes and retained PTC membership, checks AST half-open placement, then replays
support-ordered cleaned values through the serialized operator. Its numerator
comparison is exact, preserving operation order. WCS representation testing is
preregistered over every pixel center and all outer-boundary vertices with a
0.1 arcsec spherical separation bound. This tests representation agreement,
not independent absolute astrometric truth or source recovery.

Initial review found incomplete required handoff closure, insufficient schedule
identity checks, incomplete admission identity and incidence-only operator
verification. All were repaired with negative tests. A second review requested
full content-addressed admission identity and remaining source tuple fields;
those were added. Final independent exact-candidate review is still required.

The three-array engineering run used local networks 0, 7 and 12, all 124
segments each, observed-entry ALS rank 10, original factors 1/1/2, original
2-arcsec grid scale and coverage cut 0.5. It produced 47,007 / 47,128 / 48,485
science pixels in 156.77 s wall / 122.08 s CPU, peak RSS 4.95 GB. The run
predates the final admission identity repair and is debugging evidence only;
final scientific bundle/reload evidence must use the clean candidate.

Local dependency recovery was necessary because files in the old temporary
source snapshot disappeared. All surviving snapshot files compared identically
to existing local dependency-cache files; the cache was copied to a new owned
temporary snapshot and rebuilt offline. The evidence directory records the
comparison, configure arguments and initial failures. This remains supplemental
AppleClang/Homebrew Release evidence, not authoritative Spack/Unity qualification.
The current CTest inventory has no missing prior named tests (the prior log
contained repeated discovery numbering); report the actual current count.

Reproduce the route with the tracked helper (repeat `--source` for each verified
network, without relabeling missing local networks as included):

```text
/Users/gwilson/tolteca/bin/python tools/timestream_successor/prepare_forward_map.py \
  --source ABSOLUTE_PTC_PRODUCT_ROOT ABSOLUTE_ORIGINAL_INPUT_JSON \
  --owner-binding validation/map_first_forward_owner_binding_2026-09-22.json \
  --pixel-size-arcsec 2 --coverage-cut 0.5 \
  --request-dir ABSOLUTE_NEW_REQUEST_DIR --output ABSOLUTE_NEW_OUTPUT_DIR
build/bin/citlali ABSOLUTE_NEW_REQUEST_DIR/development.yaml
/Users/gwilson/tolteca/bin/python tools/timestream_successor/inspect_forward_map.py \
  ABSOLUTE_NEW_OUTPUT_DIR/a1100 --report ABSOLUTE_NEW_REPORT.json --plot ABSOLUTE_NEW_IMAGE.png
```

The MAP route currently executes serially in stable occurrence order; it rejects
`network_workers > 1` and PTC overrides rather than ignoring them. Coordinate,
verification and file work dominate this local first pass; measured accumulation
is subsecond. Shared compact per-segment geometry/causes and 1-bit admission avoid
bulky per-sample records. Coaddition, new JINC behavior, stronger response/covariance
roles, uncertainty/filter methods and production qualification remain outside
this increment. Full upstream source response and absolute astrometry are not
established by the constant/delta local-operator tests.


A bounded real-operator source probe is part of final product inspection: a
unit-peak circular Gaussian, with explicit test width equal to that array's
nominal calibration width, is inserted AFTER PTC at the AST tangent origin and
truncated at 8 sigma. The same persisted MAP membership/operator and science
support produce measured peak, integrated ratio, centroid and second-moment
width. A constant-input gain check accompanies it. These are MAP-only sampling
response measurements, not a measured beam, PTC attenuation, complete upstream
response or an authorization to restore astronomical signal. Full first-pass
RTC/CAL/PTC source response remains unavailable with the present saved carrier;
that explicitly limits source-recovery and final-chain 5% goal claims.


### First real ordinary base-MAP milestone complete — 2026-09-22

Native implementation/run: `7d16b8019a1f733e325bd48f419e6078e4f4c430` (tree
`811c7488b92b95f0cc2b69e6439f14b38f6dfed7`). Final reader:
`6bbb2b58bda3928bf044ef08cfe5dfa5ef30156f` (tree
`0a9c3c1969ee0c646d7b95b95653098462720001`). The two post-run repairs touch only reader/tests:
optional absent probe footprint stays advisory; native YAML scalar `unit: 1`
faithfully decodes as dimensionless one. Native source/build inputs are identical.
The final report must preserve these distinct immutable identities.

Durable results: `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-first-forward-map-20260922`. `products/` contains complete per-array bundles;
`evidence/` contains inspected PNGs, exact replay JSON and logs; `request/`
contains the exact invocation input. Copy verification compared every byte-hash
of **2,621 files / 2,754,612,660 bytes**. Original PTC/CAL parents
remain required and linked by exact paths/hashes; this is not a raw-data archive.
The machine-readable closeout is
`handoff/evidence/MAP_SUCCESSOR_FIRST_FORWARD_REAL_2026-09-22.json`.

Observation 152390/0/2 (NGC4449) was chosen because saved observed-entry products,
pointing/APT context and all three array representatives were actually available.
Each network contributes all 124 original processing segments with zero failed
fits in these particular local parents. This is not the later eleven-network
Unity campaign: locally absent networks are not claimed; that campaign's network-4
scan-121 failure/exclusion remains unresolved here and is not filled or bypassed.

| Array | Network | Detectors | Contributions | Science pixels | MAP-only probe peak | Probe integral ratio |
|---|---:|---:|---:|---:|---:|---:|
| a1100 | 0 | 630 | 74,946,311 | 47,007 | 0.930221 | 1.000344 |
| a1400 | 7 | 420 | 27,672,182 | 47,128 | 0.967527 | 0.997431 |
| a2000 | 12 | 414 | 27,157,906 | 48,485 | 0.982660 | 0.993499 |

Source-probe peaks are relative to unit input inserted AFTER PTC; nominal-width
Gaussian templates have explicit 8-sigma truncation. Their equivalent second-
moment widths are 5.1970/7.6420/10.0590 arcsec, with centroid offsets below
0.05 arcsec. Constant-response gain error is exactly zero. These measure the
realized MAP sampling operator only; they neither measure actual beams nor
establish RTC/CAL/PTC attenuation, source flux recovery or the final-chain 5%
goals. The complete upstream source-response carrier is unavailable.

Every real bundle passes exact cleaned-PTC numerator replay, exact retained
column/order/AST placement and coefficient/VAL closure, FITS checksum/reload,
N/Q, reference/support checks and all unsupported-sky NaNs. Finite WCS comparison
populations are 172,077 / 170,663 / 174,599 points; maximum spherical difference
is 1.145e-10 arcsec in each. This establishes stored WCS representation agreement,
not independent absolute astrometric truth.

All three signed-signal/normalization plots were visually inspected. RA increases
leftward and Dec upward. Array/network-specific coverage footprints and holes
remain visible; low-coverage pixels are unavailable, not zero. Signal is mainly
fine-scale signed structure with mixed positive/negative pixels. No robust galaxy
flux-recovery claim follows from these first-pass images. Normalization is never
called inverse variance, sensitivity, significance or exposure. Plot limits use
99.5-percentile absolute values for display only; FITS values are unchanged.

Runtime: **146.454 s wall, 122.425 s CPU, 4.965 GB peak RSS**. Input loading took
14.102 s. AST/handoff times were 30.672/17.280/13.516 s; projection/resolution
23.470/10.832/9.183 s; accumulation 0.182/0.066/0.065 s; serialization
1.502/0.620/0.602 s. End-of-run source nonmutation checks account for remaining
wall time. No upstream reduction was rerun; no response/covariance companions
were fabricated. Serial stable-order accumulation is a measured subsecond
component, not evidence for a new parallel optimization project.

Final gates: **1,283 runnable CTests pass / 1,284 registered**; one established
disabled MapFitterLifecycle case remains disabled. **207 baseline-tool tests,
172 timestream-tool tests (including 12 final reader fixtures), two real-parent
identity mutation cases and required full config preflight pass.** Final real
run has zero unexpected error-level messages. Initial compile/test/reader
failures and their passing repairs remain in evidence; no required-data skip is
claimed as pass. Reviewed tests cover the changed AST/live boundary as well as
the saved route. Earlier inflated numbering contained duplicate test names;
comparison finds no prior named test missing from the current inventory.

Independent Tier-2 final review: scientific/behavioral **pass with recorded
limitations**, architecture/ownership **pass**, repository/evidence **pass**
subject to the now completed copy/provenance closeout. Reviewer independently
reproduced focused tests, all 12 reader fixtures and the real a2000 replay, and
inspected the three images. All findings were repaired; there are no remaining
blocker/major/minor findings. Source and evidence review does not authorize
integration, production use, full package conformance or Unity/Spack qualification.

The current completed scope is one ordinary base observation MAP with explicit
uniform coefficients. Coadd, stronger response/covariance-qualified roles,
original-acquisition exposure, NOI/FLT numerical companions, general parallel MAP,
complete upstream response, absolute astrometric qualification and production
qualification remain unimplemented or unavailable as named above. FRUIT is
still deferred. No push, canonical integration, Unity access or upstream treatment
change occurred; the unrelated ALIGN checkout remains unchanged.

Prioritized next owner discussion (no choices made here):

1. **MAP evidence:** choose whether to extend the available-network observation
   map and obtain full upstream source-response/uncertainty carriers before
   quantitative flux/PSF or significance claims. These limit interpretation of
   current maps, not validity of the base product. Do not use gamma or hits as
   the missing uncertainty.
2. **FRUIT sky inference:** select the actual inference/admission rule and its
   required noise evidence, spatial/morphology and sign assumptions for deriving
   a replacement sky model from the reconstructed total. Current noisy first-
   pass maps do not supply a justified threshold or morphology rule by themselves.
3. **FRUIT numerical feedback and stopping:** specify model-to-feedback flux
   transformation, any floor/gain/damping, convergence and terminal-iteration
   choice. Per-iteration PTC relearning and replacement inference from total are
   already accepted and are not reopened. No feedback recurrence is activated.

## POINT 152389 follow-on — 2026-09-22 preflight

The owner selected POINT 152389 (1146+399) after noting that the NGC4449
pass was a poor first recognizable-source demonstration. The NGC products
remain engineering evidence; their completion does not complete this follow-on.
This continues the same Tier 2 spine increment from
`2049d9e4e5d91f8590d6a2f82fab972ac72aff0b`, tree
`d3b80ea87ebc6b466313d64114745adbd843366f`, on the same branch/worktree.
Initial dirt: only the existing untracked build locator. All three governance
documents were reread. No new WIP slot, integration, push, or Unity access.

Bounded adapter repairs may bind this selected observation's actual raw, APT,
telescope, native timing, processing support, constant POINT offset and PTC
publication identities. The existing AST/ALIGN/RTC/CAL/PTC numerical owners,
observed ALS rank 10, explicit array lowpass plans, MAP uniform coefficients,
J2000 frame and nominal 5/7.5/10 arcsec beams are preserved. Historical NGC
notches do not transfer to different detector occurrences. Older POINT maps
are comparison evidence only. Their legacy provenance lacks native cohort
lineage and cannot be relabeled as successor support.

Expected changes: private native ingress/processing/CAL/MAP adapters, bounded
input preparation, focused scope/lineage tests, POINT diagnostics and this
status/handoff. Scientific identity, units and missing-state semantics remain
with their existing owners. Memory is observation-shared metadata plus one
network's numerical state per worker; run costs must be measured. Gates are
focused adapter/CLI/MAP tests, broad CTest and Python/config gates, actual
three-array POINT products and independent fresh-context exact-SHA review.
The local AppleClang/Homebrew build remains supplemental; no representative
Spack/Unity qualification is claimed.

Reassessment: the existing no-spatial-mask directive is explicitly scoped to
152390; the older POINT config uses 20 arcsec protection. The owner has been
asked which treatment applies before POINT execution. No new treatment is
authorized merely by copying an old configuration. Continue independent
identity/interface preparation while that choice is pending. Stop for absent
required scientific authority or a necessary upstream algorithm change.

### Verified preparation; scientific selection still pending

Local raw/Tune digests match the exact selected APT for all 11 supplied
networks, comprising 5,518 detector channels and 7,698–7,700 native rows
per network. Absent networks are 6 and 10. The actual telescope is POINT
1146+399, Lissajous, Epoch 2000, CoordSys 0; SHA256
`2aec5f1f2be8c365127fcf275a02eb0c780eeda9494cd225e50e50c4386a9392`.
The new preparation tool selects the POINT input/constant offsets and retains
the reference successor 10-second forced processing intervals, 32-sample
context and array-specific lowpass artifacts. It creates a requested generation
bound to exact raw/Tune/APT/TEL/config identities; it does not forge a previous
realized cohort. Existing numerical owners construct and publish actual support.

Build and 1,283 runnable CTests pass (one established disabled lifecycle test),
207 baseline Python tests, 172 successor Python tests and full required config
preflight pass. Three real-input negative checks pass: absent source-protection
selection stops before Learn/Apply; foreign processing parents and invented
realization are rejected. The preliminary native preparation reached shared
APT/timing/AST/CAL preparation and then stopped at the expected source selection
error. Deliberate negative-test errors are expected, not successful reductions.
No POINT source amplitude, position, width, response or reduction cost has been
measured. Independent review of this preparation candidate remains required;
final observational review follows an actual authorized POINT run.

Evidence is indexed in
`handoff/evidence/MAP_POINT_152389_PREPARATION_2026-09-22.json`.
Prepared inputs: `/private/tmp/citlali-point-152389-bound-preparation-20260922`.
The preparation is deliberately non-runnable without an explicit selection.
After the owner's answer, bind only the selected RTC treatment, complete its
focused tests, run network 0 then all supplied networks, and build separate
J2000 array maps from their actual PTC products. Inspect and report source
centroid, peak and measured FWHM; do not confuse nominal calibration beams
with measured response or transfer the 152390 network-4 failure to 152389.

Reproduce input preparation from repository root (a new output path is required):

```sh
/Users/gwilson/tolteca/bin/python tools/timestream_successor/prepare_point_forward.py \
  --point /Users/gwilson/work_toltec/local_data/citlali-validation/v2/point \
  --reference-config /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-full-network-checkpoint-98380090e-20260919/unity-packet/assets/effective_config/citlali_o152390_0_2_c2.yaml \
  --output /private/tmp/citlali-point-152389-preparation-new
```

The outstanding question comes from the exact scope guard in
`tools/timestream_successor/rtc_multidetector_bindings.h`, not from a missing
file or a need to reconfirm J2000, beams, coefficients, or the observation.

Preparation candidate `7aa0d73e145816e1f5807263e7d5c3458710a414`, tree
`b64e310109c59dbb24c8b036d459e268b89be8f2`, received independent fresh-context
Tier 2 review: no blocker/major/minor findings. Scientific/behavioral and
repository/evidence dispositions pass with the recorded preparation-only and
local-environment limitations; architecture/ownership passes. The reviewer
independently ran 19 default CLI tests against the exact rebuilt candidate and
verified indexed evidence digests. Three actual preparation rejection checks
also pass on that exact candidate. Tracked worktree was clean; only the build
locator was untracked. The following evidence-only commit records this review
and does not change the reviewed runtime. Owner source-protection selection,
POINT execution and subsequent observational review remain pending.

### POINT source treatment selected; execution resumed

Owner response on 2026-09-22: "let's keep the current treatment as you recommend".
This explicitly selects no spatial source-protection mask for POINT 152389/0/2
for this first pass. Its separate binding is
`owner-152389-first-MAP-no-spatial-mask-2026-09-22`; the older 152390 authority
remains scoped and cannot substitute for this selection. This continues the
same approved Tier 2 MAP increment from preparation checkpoint
`4b7515f22a1b8555b2e44854bd17635fbf577996`. Governed numerical owners, settings,
frame, coefficients and beams are unchanged. Run one-network smoke, then all
11 supplied networks locally with bounded concurrent network ownership and one
internal numerical thread. Complete actual MAP/reload/source diagnostics and
exact-candidate independent review before reporting completion.

## POINT 152389 first forward products — executed 2026-09-22

Native runtime: `a42373f7576ea9b90f3a3b17393122f25a802735`. The selected
no-mask treatment, existing per-array lowpass/F1,F1,F2, ALS rank 10, J2000
frame, nominal beams and uniform MAP coefficients were used. The final MAP
uses 2 arcsec pixels and coverage cut 0.5, matching the previous MAP pass.
The preserved preliminary `map-run` used cut 0.1 and is superseded by
`map-pass/products`; no parameter search or upstream numerical alteration.

All 11 supplied networks were requested. Networks 2 and 11 stop in RTC because
admitted jump transitions have no existing processing-scan support. Their
learning/event/processing records are preserved. Admitted events occur in the
unused tail beyond the six 10-second processing intervals: for example n2
channel 246 seed 7448 (science ends 7289), and n11 several channels near 7366
(science ends 7288). These observations identify the interface problem, not
authority to waive admission or extrapolate processing support. No repair to
that numerical policy is included. Network 3 scan 5 reaches the unchanged ALS
100-iteration limit without convergence; its 420,413 eligible cells contribute
no MAP samples. Other segments remain usable under the accepted partial-output
contract. This is a failed/partial upstream observation attempt, not a successful
all-network reduction. Network 4 succeeds on this POINT; the older 152390 failure
does not transfer. Networks 6/10 are not supplied.

MAP completes successfully from networks 0,1,3,4,5 / 7,8,9 / 12 for
a1100/a1400/a2000 respectively, preserving all inherited causes. Contributions
are 13,453,121 / 6,719,154 / 1,402,249; science pixels 10,598 / 9,780 / 8,145.
All three complete bundles pass exact PTC-value/operator replay, parent/QC/VAL
closure, WCS representation, checksums, support and signed-NaN semantics.
The reader additionally rejects mismatched observation/array identities.
The actual source is visually clear in all three bands. A separate descriptive
unweighted elliptical Gaussian plus constant within 40 arcsec gives:

| Array | Central pixel peak (mJy/nominal beam) | East, north offset (arcsec) | Measured major × minor FWHM (arcsec) |
|---|---:|---:|---:|
| a1100 | 732.36 | +4.523, −5.401 | 5.802 × 5.073 |
| a1400 | 525.35 | +4.378, −5.405 | 7.165 × 6.282 |
| a2000 | 679.64 | +4.318, −5.048 | 9.488 × 8.459 |

Fitted amplitudes above the local constant are 830.26 / 554.15 / 731.64
mJy/nominal beam. These descriptive source widths are separate from nominal
calibration beams and do not establish full-chain PSF, flux recovery, uncertainty
or detection significance. Signed residual structure remains visible, particularly
in the lower-coverage a2000 map. No map values were altered for display or fit.

All-network upstream attempt: 82.25 s wall, 171.25 s process CPU, 3.736 GB
peak RSS, four workers with one internal thread each on 14 reported logical
CPUs. Serial network 0 smoke took 31.44 s native wall and matches all 48 PTC
numerical files bit-for-bit against the four-worker result (configuration paths
were rebound; numerical settings/inputs were identical). Successful final MAP:
57.51 s invocation wall, 34.40 s process CPU, 1.465 GB peak RSS.
Per-array AST/handoff, projection, accumulation and serialization timings are
in each bundle. Full response/covariance/original-footprint exposure remain
explicitly unavailable. MAP-only Gaussian probe remains a conditional MAP test.

Validation: 1,284 runnable CTests pass, one established disabled lifecycle case;
175 successor Python tests pass, including known offset/elliptical-source/holes
and observation-mismatch fixtures; two real-parent mutation checks pass; full
required config preflight passes. The prior preparation checkpoint's 207 baseline
Python tests remain applicable: no baseline-tool changes. The successful MAP
run has zero unexpected errors; the two RTC failures and one PTC failure above
remain failure evidence, not passing results. Initial smoke's missing descriptive
`positive_rationale` field was repaired in the input preparer before the full run.
Local AppleClang/Homebrew evidence is supplemental; no Unity/Spack certification.

Durable result root:
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-forward-map-20260922`.
Final science products: `map-pass/products/{a1100,a1400,a2000}`; source view
and fits: `diagnostics/final-source`; full-field/support diagnostics and exact
replay reports: `diagnostics/final-bundles`. Upstream numerical parents, native
failed-network records and digest-bound inputs are retained under the same root.
External original raw/APT/TEL sources remain required for a from-raw rerun.
Evidence index: `handoff/evidence/MAP_POINT_152389_FORWARD_2026-09-22.json`.
Final independent exact-candidate review is pending; FRUIT remains deferred.

Next MAP work: resolve out-of-scan jump handling and the nonconverged PTC segment
under their own scientific owners; complete upstream response/noise evidence
before stronger flux/PSF claims. FRUIT discussion should settle the inference
model for the reconstructed total and its acceptance/stopping rules. The
descriptive Gaussian here is not a selected feedback model; no recurrence,
threshold, damping or terminal-selection policy has been activated.

### Final POINT review and disposition

Independent fresh-context Tier 2 review passes with recorded limitations and
no blocker, major or minor findings at
`4c1aefac9ecacbdedeaaeffbe7a5038c2e564b88`, tree
`11cb71f46cb319ecb08ff83ad29265e1f6e3d7c1`. Its direct parent is the exact
executed native runtime `a42373f7576ea9b90f3a3b17393122f25a802735`; no native
source changed after the runs. The reviewer independently replayed all three
final MAP bundles and recomputed all three source fits with identical reports,
visually inspected the source, and ran 15 focused tests successfully. Broader
test logs were inspected rather than rerun by the reviewer.

Scientific/behavioral disposition: pass with the explicitly retained upstream
failures, missing optional stronger-role companions and descriptive-fit limits.
Architecture/ownership: pass; existing owners and numerical treatment remain
intact. Repository/evidence: pass with supplemental-local-environment limits;
tracked source clean, only the existing build locator untracked. Failed RTC
networks and the nonconverged PTC segment are not reclassified as success.

The owner-requested POINT central-source demonstration is complete with those
coverage exclusions. This does not establish all-network upstream conformance,
full-chain flux/PSF goals, package-wide MAP conformance, canonical integration,
activation or production authorization. The final evidence-only closure commit
records this review without changing the reviewed implementation or products.
Stop at this owner review point; FRUIT and upstream-policy repairs await the
next work order.
