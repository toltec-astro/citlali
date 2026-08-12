# SCI-CAL-001 successor-5 independent re-audit

Date: 2026-08-12

Auditor role: fresh, role-separated independent technical auditor

Candidate ref: `origin/codex/repair-sci-cal-001-successor-5`

Candidate commit: `5dfc414a13fe69e6b063608906d87e3b30491ec7`

Candidate parent: `693f1b107855e3ae9b36617323ca14aac868f304`

Candidate tree: `72e4df08bc3677290b03d1c39457ea049f8db813`

Parent-to-candidate binary patch SHA-256:
`1c9e634c574da60c40cf7e2808b1ec1ac25d1fa8f80cd4de7cb31230365cf7d8`

Candidate changed paths: 23

## Executive disposition

The candidate is **not a conforming completion of the bounded successor-5
repair**. F005, F007, F008, and local F009 remain open and nonconformant.
Independent production-source traces and bounded executable counterexamples
found defects that the candidate's green tests do not exercise:

- a real corrected observation reaches the new F005 carrier before that
  carrier is initialized, so initial setup rejects it; the later flux setup
  would erase a manually injected correction before admission;
- the F007 response identity still labels mutable effective configuration as
  immutable requested state, omits sample rate from coefficient-defining
  notch geometry, misses the TOD-only production writer, and registers coadd
  joins under selected-APT identity rather than the reduced observation;
- calibrated Beammap setup requires a finalized join before the only
  finalizer can run, and linked TOD metadata is mutated non-atomically after
  package publication; and
- the v4 auditor rejects valid legacy no-manifest production lineage while
  accepting production-impossible and forged lineage, and the changed active
  product contract still rejects an accepted calibrated v2 baseline.

The exact controlled axes therefore remain:

| Axis | State | Independent basis |
|---|---|---|
| Contract | `approved` | The frozen successor-5 owner authority remains the governing contract. |
| Implementation | `nonconformant` | Each repair finding retains at least one concrete production or executable-contract counterexample. |
| Validation | `in_progress` | All runnable deterministic candidate tests pass, but key fixtures manually bypass the failing production order or validate declarations without recomputing their payload identities. |
| Production | `fail_closed` | No production authorization is supplied; the local F009 consumer itself contains fail-open integrity checks, but the controlled production disposition remains fail-closed. |
| Verdict | `amend` | Reject complete successor-5 closure and return the bounded findings for owner-directed repair. |

F002, F003, F004, and F006 remain closed only within their previously
accepted bounds. F001 and F010 remain open and conditioned. A green test
matrix does not override the reproduced production-path contradictions.

## Entry checkpoint, frozen authority, and scope

Before substantive candidate source, diff, test-body exposure, artifact
editing, configuration, build, or test execution, the mandatory READY
checkpoint independently established and the coordinator accepted:

- exact local and live-origin candidate commit, single parent, tree, binary
  patch digest, and 23-path inventory;
- a clean detached worktree and index physically separated from the
  successor-5 repair, frozen coordination, and prior re-audit worktrees;
- absence of the proposed audit branch locally, in remote-tracking state, and
  at live origin;
- application classification for the candidate and documentation-only
  classification for this audit branch;
- the finding/exposure/test plan, all prohibitions, and the ceiling of exactly
  this report plus one local-evidence YAML; and
- existence, reachability, and exact hashes of the frozen coordination and
  source-audit artifacts.

The frozen coordination authority is
`7b341a5fb1c96080f4c3513f62a4f12441f2443d`. The independently verified
artifact SHA-256 values are:

| Frozen artifact | SHA-256 |
|---|---|
| Successor-4 owner acceptance | `d01d329d08e3622f7e2329e1f889a19f47d0b24858453b2e60ed85e6b8bbce2d` |
| Successor-5 bounded repair handoff | `28cb3732eaf4b43c4850dceefb836f099b0591b2feeb358a7bc5f1585c445a33` |
| Successor-5 finding ledger | `faf2f867a69062d040101bd4d2e969057e4fafb2e9ab3da8770f4fc1e821d45b` |
| Accepted successor-4 re-audit report | `47a7bc888a2c5b1981e96287b46c20392146f073f1cdb5cc092e842ddeeb9d9c` |
| Accepted successor-4 local evidence | `597f8fe2c9106e287cc1eda9e341ff8c50d0d90d151add8cf703904cbfae44a0` |

After approval, branch
`codex/reaudit-sci-cal-001-successor-5-20260812` was created directly at the
exact candidate. Candidate prose and tests were treated as claims until
independently traced or reproduced.

No candidate repair, application/configuration/test/build-system/validation
product/frozen-coordination edit, push, merge, Unity access or request,
astronomical reduction, source injection, external contact, downstream task,
or production authorization occurred.

## Complete changed-path assessment

The 23 modified paths consist of one status document, 11 core application
headers, two C++ test files, six baseline tool/fixture/test files, one raw
configuration census tool, and two validation registry/contract files. Every
path is inside the frozen 27-path repair ceiling; there are no unclassified
candidate files.

The RTC/PTC diffs add response-history capture, identity serialization, and
lifecycle/join checks around existing application calls. The audit found no
changed notch detection, selection, coefficient arithmetic, flagging, scan
cadence, mapmaking algorithm, new uncertainty/covariance product, or
cross-package architecture. The authorized F005 factor placement changes
calibration-product composition but does not alter the established map or
full-weight algorithms.

There are nevertheless material governance contradictions inside the named
scope:

- the new status entry claims observation-owned F005 state, complete
  production joins, production-shaped F009 integrity, and preserved v1--v3
  compatibility, while the traces below disprove those claims;
- active base product contracts were changed in place despite the registry's
  immutable-snapshot/successor-epoch policy; and
- three active and four preparing profile comparators were broadened from a
  root-only exclusion to `*/selected_calibration_apt.ecsv`, so the newly
  required package member is excluded from strict product comparison.

The raw execution census was not weakened: it grew to 87 records with no
review item or drift and an exact new digest recorded below.

## Independent findings

### F001 — open, conditioned, and unchanged

The candidate supplies no new authorized SCI-ALIGN, SCI-AST, exact-SHA Unity,
astronomical-standard, or observational evidence. F001 remains open and
conditioned without reinterpretation.

### F002 — narrow structural closure preserved

The fixed-DJF25 contract, node table, generated header, and production
operator are byte-identical to the parent. Their SHA-256 values remain:

| Artifact | SHA-256 |
|---|---|
| `data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_contract.json` | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| `data/calibration/sci_cal_001_fixed_djf25_full_domain_operator_nodes.csv` | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| `include/citlali/core/timestream/atmosphere_operator_nodes_generated.h` | `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262` |
| `include/citlali/core/timestream/atmosphere_operator.h` | `3fd4352d05e77e07c1e354b7e4124733505064667968676d8d4e94315017d584` |

The generator reconstructed 1,368 rows and 72 series, and all ten focused
C++ atmosphere tests passed. This preserves only the accepted structural
closure; it makes no atmospheric-model fidelity or uncertainty claim.

### F003 — accepted startup/admission closure preserved

The initial typed configuration validation and CLI execution boundary are
byte-identical to the parent (Git blobs
`93e2c576ae58b99dda461ee463de137ce99eaedd` and
`6a92025f54d91327fbe15e9324eaf37b354d01a8`). Unsupported calibrated units
still fail before output-root leasing or scientific-product work, while the
intentionally uncalibrated request remains supported. F003 stays closed only
at that approved boundary.

### F004 — accepted APT association closure preserved

`include/citlali/core/engine/calib.h`,
`src/citlali/core/engine/calib.cpp`, and
`tests/test_calib_apt_filtering.cpp` are byte-identical to the parent (Git
blobs `19e844df971efb939af260ee0cc5bc8bd03417eb`,
`09050f56752fc37dc72484479aa1eb5a2b65c6bc`, and
`54b310da831a5e823b1500c0cb2f462649272b09`). Nine focused APT lineage,
binding, and unit-policy tests passed. Unique optional-modern TolAPT
association, exact selected-source metadata, stable ordered-row association,
and valid legacy no-manifest lineage remain intact in the producer.

F004 stays closed within that accepted structural association claim. Local
F009 is nonconformant precisely because its new consumer rejects one of these
valid legacy producer states.

### F005 — nonconformant and open

#### The real production correction route cannot admit a corrected observation

`prepare_initial_observation_setup` loads the observation calibration and
immediately calls `apply_flxscale_correction`
(`initial_observation_setup.h:26-30`). A production `Calib` starts with an
empty `flux_conversion_factor`, while the new recorder requires its
cardinality to equal the selected APT and be nonzero
(`flxscale_correction.h:37-50`). A valid correction therefore returns false
and aborts initial setup.

The only carrier initialization is later in
`Calib::calc_flux_calibration`, which also clears the magic summary key and
resets the carrier to ones (`calib.cpp:1040-1050`). It is called from the
later reduction-observation input route
(`reduction_observation_inputs.h:72-88`), before admission reads that key and
carrier (`calibration_product_admission.h:319-335`). Thus artificial
pre-initialization would not repair production: the correction is erased
before admission.

The state is also not observation-owned. It is a magic entry in persistent
`Calib::mean_flux_conversion_factor`; the initial multi-input geometry pass
has no correction-state observation boundary, so a second corrected
observation can be rejected as an already-applied duplicate. The generic
fallback for engine-shaped scaffolds still mutates source APT `flxscale`
(`flxscale_correction.h:88-93`), contrary to the unqualified immutable-source
contract.

The candidate test at
`tests/test_science_map_fits_products.cpp:298-529` manually initializes the
carrier, directly constructs admission inputs, and then calls the downstream
helpers. It bypasses both the failing initial order and the later erasure.

#### Composed factors do not fail closed

Admission validates each finite positive operand separately
(`calibration_product.h:617-648`) and composes the signal multiplier without
validating the product (`:741-755`). The runtime calibrator then trusts product
validity/cardinality (`calibrate.h:146-168`). A bounded header-level harness
admitted both `DBL_MAX * 2 = inf` and
`denorm_min * 0.5 = 0` as valid complete products. This supplies the requested
nonfinite/underflow counterexamples.

#### Applied-factor provenance is not truthful enough to recover the state

The applied boolean/scalar exist in memory and enter a digest, but no map
PHDU, TOD NetCDF, Beammap ECSV metadata, or raw canonical-lineage output
serializes the actual scalar or applied boolean. Those outputs publish generic
factor prose, a non-recoverable factor hash, nuisance status, and multiplier
extrema. They can claim `valid_applied_value` and “explicit correction value
applied once” without disclosing that value.

#### Downstream arithmetic is conditionally consistent

If a moderate correction is manually injected into an admitted product, the
production branch leaves source APT `flxscale` and `sens` unchanged,
`calibrate_tod` applies `a` once through its established compatibility
carrier, approximate weight obtains the required inverse square, hybrid and
validated inherit that approximate baseline, and full weight remains the
inverse variance of already-calibrated samples. The naive map and normalized
nonzero realizations then reach `MapBuffer::noise_variance` and FITS
`noise_variance_I` consistently.

That lower-level arithmetic is not reachable through the required corrected
observation route and does not cure the lifecycle, fail-closed, or provenance
defects. F005 remains open.

### F006 — bounded `mJy/beam` closure preserved

The typed configuration boundary, runtime calibration setup, and complete
product admission still accept calibrated output only as top-of-atmosphere
point-source-peak `mJy/beam`; no additional calibrated unit was introduced.
F006 remains closed only for that bounded unit policy. It establishes neither
response fidelity nor total uncertainty.

### F007 — nonconformant and open

The candidate usefully records fixed, shared, and detector applications after
their actual RTC/PTC filter calls. Records retain duplicates and order and
carry phase, stage, scan, local PTC iteration, model-subtraction state, scope,
detector/ordinal, center/width geometry, and phase convention. Dormant stages
are labelled requested/effective rather than realized, and the isolated
homogeneous-join helper fails closed for missing or differing identities.

Four production defects remain:

1. `calibration_response_identity` hashes live
   `raw_time_chunk_config(engine)` and labels it `requested_state_sha256`
   (`calibration_product_admission.h:54-73`). Production mutates that live
   configuration when deriving a downsample factor
   (`downsample_config.h:74-85`), while the actual immutable request is
   `raw_timestream_plan.requested`
   (`raw_timestream_execution_plan.h:113-145`). The candidate test at
   `test_science_map_fits_products.cpp:1359-1375` positively expects live
   mutation to change the alleged requested identity.
2. Final CALID/PKGID variables are appended only by `add_tod_header`, whose
   production callers are RawObs map writers. With mapmaking disabled,
   `raw_observation_outputs.h:23-33` skips those writers. `create_tod_files`
   creates only the skeleton/layout. Thus supported TOD-only operation never
   receives the finalized join; the candidate test manually calls the
   metadata helper on a synthetic file.
3. Finalization registers the join under selected-APT observation identity,
   falling back to raw artifact identity
   (`calibration_product_admission.h:410-421`). Coadd membership and PHDU
   lookup use current reduced observation numbers
   (`rtcproc.h:552-584`). A science observation using a calibrator APT with a
   different obsnum therefore has no join. Reusing one APT for two science
   observations also makes the second observation conflict with the first
   join key. The candidate test hides this by making APT and reduced obsnum
   both `152390` and manually inserting `152391` with the first identity.
4. The realized notch record omits application sample rate even though notch
   coefficients depend on it (`rtc/filter.h:104-149`). A bounded harness used
   identical recorded center 10 Hz, width 2 Hz, and phase convention at 100
   and 200 Hz and obtained coefficient delta
   `0.65143128109757631`. Distinct actual response operators can therefore
   share the recorded geometry and response identity.

At every observation begin, history and per-scan PTC counters reset, but the
reduction-wide finalized join registry persists. Each outer fruit iteration
reprocesses observations while the recorded PTC counter restarts at zero and
contains no fruit-iteration identity. A changed later-iteration response for a
retained observation key conflicts with the old join; the candidate tests
synthesize PTC iteration values rather than exercise this production
lifecycle.

F007 remains open despite the conforming isolated recorders and heterogeneous
fail-closed helper.

### F008 — nonconformant and open

The local lifecycle primitives improve on successor-4: observation begin
clears live/final notch histories and per-scan PTC counters, recording after
consume rejects, repeated consume returns the immutable snapshot, and repeat
product finalization is idempotent for the same response or rejects a
conflict. The raw observation plan resets realized state and repeat completion
preserves or rejects changed counts/identities. State is Engine/RTCProc-owned;
no process static, singleton, global transaction, or rollback layer was added.

Production publication and lifecycle are nevertheless nonconformant:

1. `run_reduction_observation_pipeline` runs setup/TOD processing before the
   only calibration finalizer (`reduction_observation_pipeline.h:32-35`).
   `Beammap::setup` admits the product and then populates metadata
   (`beammap_setup_impl.h:14-24`), but the new metadata path immediately
   requires a finalized join
   (`beammap_setup_metadata_impl.h:81-84`). Every calibrated Beammap therefore
   throws before TOD execution, finalization, or canonical package
   publication. The candidate Beammap test manually finalizes its fixture and
   never executes this order.
2. TOD skeletons are atomically published early, but after canonical package
   publication `add_tod_header` opens each published file directly for
   in-place mutation (`tod_file_output_impl.h:24-27`). CALID and CALPKGID are
   added as separate operations (`tod_metadata_mapmaker_tau.h:140-145`). A
   throw or interruption between them leaves a partially linked artifact,
   violating per-artifact atomicity and the prohibition on unresolved linked
   products. The candidate failure test throws from a generic lambda without
   creating or partially writing its named dependent product.
3. The wrong-key, reduction-wide join registry described under F007 is not
   cleared at observation begin. It conflicts when a selected APT key is
   reused with an observation-bound identity and remains vulnerable across
   fruit iterations.

The candidate test named “unavailable” first calls history begin, which marks
history active; it does not exercise the true inactive/unavailable state.
Direct reused-scan and multiscan helper behavior is green, but the required
interrupted, unavailable, package-first, and dependent-writer production
boundaries are not established. F008 remains open.

### Local F009 — nonconformant and open

The production writer now targets the actual observation directory and exact
`{obs}/selected_calibration_apt.ecsv` sibling. It verifies source bytes,
stages/copies/rehashes the member, accepts only a matching existing member,
writes provenance atomically, and removes a newly created member if provenance
publication fails. The consumer derives the sibling from each provenance path
and hashes it. Calibrated membership is conditional, and an uncalibrated v4
without lineage/member remains supported. These are useful partial repairs.

Four independent contradictions keep the local implementation open:

1. Valid legacy selected-APT lineage may have no modern TolAPT manifest.
   Producer admission allows that state and emits an empty
   `tolapt_manifest_association_sha256` with `available: false`. The v4 auditor
   unconditionally requires every component, including that optional-modern
   component, to be canonical 64-hex
   (`audit_reduction_run.py:849-860`). The bounded legacy-shaped package was
   rejected only for that empty component. Conversely, the checked-in fixture
   sets `available: false` with a fabricated nonempty `111...` association,
   which production admission rejects, yet the auditor returns no error.
2. Component checks compare declared digests with other declared digests but
   do not recompute the raw-acquisition binding, ordered-row association,
   admitted-factor state, response basis, or CALID from their serialized
   payloads. Independently changing an ordered-row UID or factor target unit
   while retaining the declared hashes returned zero errors. Equivalent
   source-path, raw-artifact path/digest/identity, and response-provenance
   mutations also pass. PKGID is recomputed only from the already-claimed
   CALID, selected-APT digest, and acquisition digest.
3. The multi-observation test copies the same `000042` fixture unchanged into
   `000042` and `000043`. Both carry the same lineage observation and PKGID,
   yet `audit_provenance_sidecars(require_raw=True)` reports two valid covered
   observations. The fixture is not production-shaped and cannot prove
   per-calibrated-observation joins.
4. The parent contract already exposed a compatibility contradiction by
   requiring one root-level member from an accepted schema-v2 baseline that
   correctly has no v4 package. The frozen successor-5 handoff therefore
   required v1--v3 preservation and a stop if an exact contradiction was
   found. Instead of stopping or creating a successor epoch, the candidate
   changes the active entry to a per-observation conditional requirement. The
   active accepted point baseline has effective calibration true, so its valid
   v2 semantic audit still passes while the candidate contract still rejects
   it with:
   `selected-calibration-apt: pattern
   '152389/selected_calibration_apt.ecsv' matched 0; requires at least 1`.
   The predecessor error was the corresponding root pattern
   `selected_calibration_apt.ecsv`; successor-5 changed the pattern but did
   not resolve the incompatibility. Running the active profile against its
   own accepted baseline therefore fails audit/contract while config and
   byte-for-byte product comparison pass. Editing that active contract despite
   the discovered contradiction violates both the frozen stop rule and the
   registry's exact successor-epoch policy.

The six direct base contracts do express one conditional per-observation
entry, and the tests exercise several missing/tampered/stale/conflicting cases.
Those positives do not cure the valid-production false rejection,
declaration-only false acceptance, copied multi-observation identity, or
active-baseline break. The exact compatibility contradiction required a stop,
not an in-place policy change. Local F009 remains open.

### F010 — open, conditioned, and unchanged

No new authorized SCI-ALIGN, SCI-AST, exact-SHA Unity, astronomical, or
production evidence is supplied. F010 remains open and conditioned without
promotion.

## Deterministic validation on the exact candidate

A fresh Release build with tests enabled was configured outside the worktree
from the exact candidate. It used only already-present local dependency source
trees and installed libraries; no repair-worktree candidate binary and no
network dependency was used. Initial local dependency-discovery retries
created CMake cache/configuration state but no candidate binary or object file;
the final incremental configuration reused those explicit local dependency
policies and compilation succeeded. Only third-party deprecation warnings were
observed.

| Check | Independently observed result |
|---|---|
| Fresh build targets | `citlali_cli`, `citlali_test`, `citlali_safety_test`, `citlali_science_map_fits_products_test`, and `citlali_science_map_truth_test` built successfully |
| Fresh CLI smoke | exact candidate `citlali --help` exited 0 and displayed the expected command surface |
| Full CTest | 677/677 runnable passed, 0 failed; 678 enumerated with pre-existing disabled `MapFitterLifecycle.ExactProductSequence` |
| Grouped normal/safety/FITS/truth binaries | 620/620, 14/14, 43/43, and 31/31 passed; normal binary retains one disabled test |
| Focused atmosphere/APT/product/successor routes | 10/10, 9/9, 14/14, and 9/9 passed |
| Full baseline unittest discovery | 185/185 passed |
| Four changed baseline pytest suites | 136/136 passed |
| Targeted atmosphere/product-contract pytest | 30 tests plus 2 subtests passed |
| Full config preflight `--require-all` | 127/127 unit tests, four mode kits, 8/8 compact compatibility cases, 592 schema leaves, and 100% covered surface |
| Raw execution census | 87 records, zero review, zero drift, digest `6f850cbdf06458db3c408080a933149a87d97874e79d2a6161f4ba1c5757e488` |
| Atmosphere generator | exact: 1,368 rows and 72 series |
| Validation ledger | valid, 60 records |
| Science-change ledger | valid, 3 changes and 5 integration commits |
| Validation-profile registry | valid syntax, 4 active and 8 preparing profiles |
| Parent-to-candidate `git diff --check` | passed |
| Candidate patch and changed-path inventory | exact expected SHA-256 and 23 paths |
| Active point profile against its own accepted baseline | rejected: audit and contract fail; config and strict product comparison pass |
| F005 production/extreme-factor harness | fresh correction rejected; source APT/sens unchanged; admitted `inf` and zero composed multipliers reproduced |
| F007 join/request harness | reused APT key and output obs lookup rejected; immutable request stayed fixed while alleged requested response changed |
| F007 coefficient-geometry harness | same recorded geometry at 100/200 Hz produced coefficient delta `0.65143128109757631` |
| F009 integrity harness | impossible manifest pair and forged row/factor payloads accepted; valid legacy empty association rejected; copied two-observation package accepted |

The green candidate suite demonstrates buildability and ordinary regression
stability. It does not invalidate the concrete source-order, lifecycle,
collision, atomicity, and compatibility counterexamples. No astronomical
reduction or source injection was required or performed.

## Finding ledger

| Finding | Independent successor-5 disposition |
|---|---|
| F001 | open, conditioned, unchanged |
| F002 | retain narrow structural atmosphere closure, preserved |
| F003 | retain startup/admission closure, preserved |
| F004 | retain selected-APT lineage/association closure, preserved |
| F005 | open/nonconformant: corrected production setup is unreachable; carrier lifecycle, composed-factor fail-closed behavior, fallback immutability, and applied-scalar persistence remain defective |
| F006 | retain bounded `mJy/beam` closure, preserved |
| F007 | open/nonconformant: mutable request identity, TOD-only omission, wrong coadd key, incomplete operator geometry, and fruit-lifecycle ambiguity remain |
| F008 | open/nonconformant: calibrated Beammap finalization cycle, non-atomic linked TOD mutation, and retained join lifecycle remain |
| local F009 | open/nonconformant: valid legacy false rejection, forged/copy false acceptance, incomplete recomputation, and active v2 baseline break remain |
| F010 | open, conditioned, unchanged |

## Owner decision brief

Do not accept complete SCI-CAL-001 successor-5 closure. Preserve F002, F003,
F004, and F006 only within their existing bounds; keep F001 and F010
conditioned; and return F005, F007, F008, and local F009 for a new bounded
owner-directed technical repair. The v1--v3 active-contract contradiction and
the partial-link atomicity issue are authority-sensitive and must not be
silently resolved by broadening audit scope.

Retain contract `approved`, implementation `nonconformant`, validation
`in_progress`, production `fail_closed`, and verdict `amend`. Authorize no
production, Unity work, reduction, merge, push, external coordination, or
downstream activity from this re-audit.
