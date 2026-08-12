# SCI-CAL-001 successor-4 independent re-audit

Date: 2026-08-11

Auditor role: fresh, role-separated independent technical auditor

Candidate ref: `origin/codex/repair-sci-cal-001-successor-4`

Candidate commit: `693f1b107855e3ae9b36617323ca14aac868f304`

Candidate parent: `3af6faf996fa002b2647adca8f33991002d49ff1`

Candidate tree: `fb317a7862ff474c118d229bb45320adb560b3bc`

Parent-to-candidate binary patch SHA-256:
`59ba71493377630be5bff5164d779aa43d14ee1e788c002707f1f4fe62d5902d`

Candidate changed paths: 26

## Executive disposition

The candidate is **not a conforming completion of the bounded successor-4
repair**. F005, F007, F008, and local F009 all remain open and
nonconformant. Independent source traces and bounded executable
counterexamples found defects that the candidate's green tests do not cover:

- the actual per-observation `flxscale_correction` production route leaves
  approximate, hybrid, and validated weights unchanged when the calibrated
  samples acquire a factor `a`;
- materially different applied PTC/RTC notch states can share one response,
  calibration, and package identity, while repeated finalization changes
  already-finalized identities;
- required calibration/package joins are absent or ambiguous in supported
  TOD-only and multi-observation product routes; and
- the v4 audit and product-contract gates jointly accept a package with no
  digest-joined sibling APT while the contract rejects the path actually
  published by the application.

The exact scientific axes and verdict remain unchanged:

| Axis | State | Independent basis |
|---|---|---|
| Contract | `approved` | The accepted successor-4 scope remains the governing contract. |
| Implementation | `nonconformant` | Each bounded repair finding retains at least one material implementation or executable-contract counterexample. |
| Validation | `in_progress` | All runnable deterministic gates pass, but their fixtures do not cover the reproduced production paths and collision cases. |
| Production | `fail_closed` | No production or downstream authorization is supplied; the local F009 integrity surface itself has fail-open checks, but the controlled production disposition remains fail-closed. |
| Verdict | `amend` | Return only the bounded findings for owner-directed repair. |

F002, F003, F004, and F006 remain closed within their previously accepted
bounds. F001 and F010 remain open and conditioned. The audit does not close a
finding merely because tests pass.

## Entry checkpoint, frozen authority, and scope

Before substantive candidate source, diff, or test exposure, the mandatory
READY checkpoint independently established and the coordinator approved:

- live origin and local candidate identity, parent, tree, binary patch digest,
  and exact 26-path inventory;
- a clean worktree and index separated from repair and prior-audit worktrees;
- the absence of the proposed audit branch locally, in remote-tracking state,
  and at live origin;
- application classification for the candidate and documentation-only
  classification for this audit branch;
- the finding-to-path exposure map, deterministic test plan, and the ceiling
  of one dated report plus one local evidence YAML; and
- all frozen coordination objects and digests.

The frozen coordination authority is commit
`aaa578f12a1bacc9476d8067d9f1554029b67f89`. The independently verified
artifact SHA-256 values are:

| Frozen artifact | SHA-256 |
|---|---|
| Successor-3 re-audit report | `ee0f8c40e31300fd5c547b45d086a5e97f7be52e45d453016c0ade28c014e59a` |
| Successor-3 local evidence | `7245872f044fc15a7cdb631ea02b70a9746cba1351bc942c1dde8bffa2b25a6f` |
| Successor-3 owner acceptance | `73bd190349dd2ccd3405baa4cab294deb4e87c576367a54dbc44b3014a52a9e1` |
| Successor-4 bounded repair handoff | `036de8014ae959bdbc237d3ba3a79ee399479d56f66d8455f51b821455b003e7` |
| Successor-4 finding ledger | `0b00972f2c02d65a4f2137e37a0b5f19e1756bd3fe339b10764a27128ff67877` |

After approval, branch
`codex/reaudit-sci-cal-001-successor-4-20260811` was created directly at the
exact candidate. Every changed path, the surrounding production source, the
actual writer and validator routes, and the relevant tests were inspected.
Candidate prose and tests were treated as claims until independently traced or
reproduced.

No application, configuration, test, build-system, validation-product,
registry, mode-profile, baseline-policy, production, or frozen-coordination
file was edited. No repair, push, merge, Unity access or request, astronomical
reduction, source injection, external contact, downstream task, or production
authorization occurred.

## Complete changed-path assessment

The 26-path diff stays within the approved successor-4 application surface:

- F005 declarations and evidence: calibration-product metadata and CAL/map
  tests;
- F007/F008: calibration admission/finalization, map/TOD/Beammap metadata,
  raw execution state, and `rtcproc.h` applied-notch history;
- local F009: v4 raw provenance, package lifecycle, executable audit,
  product contracts, validation profiles, baseline policy/tests, and fixtures;
  and
- the status handback.

The `rtcproc.h` parent-to-candidate diff is metadata recording and
snapshot/consume support only. It does not change RTC detection, selection,
filtering arithmetic, flagging, scan enumeration, cadence, or other scientific
behavior. No weighting, variance, mapmaking, or calibration arithmetic file
was changed for F005. No unexpected science, cross-package, production
authority, or algorithmic broadening was found in the diff. The defects below
are failures to satisfy the already-approved contract, not authority for the
auditor to redesign those systems.

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

The generator check reconstructed 1,368 rows and 72 series, and all ten
atmosphere tests passed. This preserves only the accepted structural closure;
it makes no atmospheric-model fidelity or uncertainty claim.

### F003 — accepted startup/admission closure preserved

The configuration validation and CLI execution-boundary implementations are
byte-identical to the parent (Git blobs `93e2c576ae58b99dda461ee463de137ce99eaedd`
and `6a92025f54d91327fbe15e9324eaf37b354d01a8`). Unsupported calibrated units
still fail in initial typed validation before the output-root lease,
APT/observation/output work, or scientific-product mutation. The intentionally
uncalibrated request remains supported. F003 stays closed at its approved
boundary.

### F004 — accepted APT association closure preserved

The core Calib implementation, header, and focused lineage tests are
byte-identical to the parent. Unique optional-modern TolAPT association,
exact selected-source digest and metadata, stable ordered-row association,
legacy lineage, and validity distinctions remain intact. Changed calibration
admission preserves source digest, row-association consistency, and complete
modern manifest association. F004 stays closed only within the accepted
structural association claim; no producer authentication or scientific QC is
inferred.

### F005 — nonconformant and open

The candidate directly proves that the actual
`MapBuffer::calc_noise_products` to FITS `noise_variance_I` recipient scales
as `V' = a^2 V` for nonzero, already-normalized input realizations. The
independent harness additionally confirms that real naive mapping and
normalization deliver realizations scaled by `a`, and that full weighting
derives `W' = W/a^2` from already-calibrated sample variance.

The complete recipient claim is nevertheless false for the actual
per-observation correction route:

1. `initial_observation_setup.h` and
   `reduction_observation_calibration.h` configure the selected calibration
   and then invoke `apply_flxscale_correction` for their respective
   observation paths.
2. `flxscale_correction.h` multiplies only selected-APT `flxscale`; it does
   not update `sens`.
3. Beammap selected-APT construction had already multiplied `sens` by the
   original detector flxscale exactly once.
4. Admission consumes corrected `flxscale` and unchanged `sens`. RTC scales
   samples by corrected flxscale. In the extinction-disabled counterexample,
   compatibility FCF remains target-unit factor 1; in general it can also
   contain scan-mean extinction, but it never contains the observation
   flxscale correction.
5. Approximate weight uses compatibility FCF times unchanged selected-APT
   `sens`. Hybrid and validated start from the same approximate baseline, and
   their dimensionless modifiers do not restore the missing `a^-2`.

An exact-candidate executable counterexample used the production correction,
admission, RTC calibration, all four weight modes, naive mapping,
normalization, nonzero realizations, and `calc_noise_products`. For `a = 3`:

| Mode | Raw weight | Required corrected weight | Actual corrected weight | Recipient result |
|---|---:|---:|---:|---|
| approximate | 0.25 | 0.027777777777777776 | 0.25 | nonconformant |
| hybrid | 0.25 | 0.027777777777777776 | 0.25 | nonconformant |
| validated | 0.25 | 0.027777777777777776 | 0.25 | nonconformant |
| full | 0.49999999999999994 | 0.055555555555555552 | 0.055555555555555566 | conformant |

All four normalized map signals still changed from 2 to 6, realizations from
`[-2, 2]` to `[-6, 6]`, and variance from 4 to 36. That positive variance
result also confirms one application of `a`, not a doubled calibration pass;
it does not repair the approximate-derived formal weight, which is wrong by
`a^2 = 9`.

The counterexample source SHA-256 is
`7f8f5409551c2f526e601908c2453d8e360b7eee4a701016de772720a04af9eb`;
its compile/run script SHA-256 is
`6759922686e444b46984b8f5e06f9e2214fbf9f2f0a5e5bd0b99e897b80d00a0`.
The candidate test manually scales compatibility FCF, so it does not execute
this production mutation. The declared recipient provenance is also
untruthful for corrected observations: selected-APT `sens` contains the
original flxscale once, not the later observation correction.

Minimal bounded requirement: make the admitted observation correction
participate exactly once in approximate-baseline coefficient scaling and
therefore hybrid/validated, retain full-mode behavior, and add a production
correction-path test through nonzero map realizations and variance
publication. Choosing whether that factor belongs in `sens`, compatibility
FCF, or another approved factor is a scientific/arithmetic choice. This audit
does not make or implement that choice and returns it for owner review.

### F006 — bounded `mJy/beam` closure preserved

The `mJy/beam`-only calibrated configuration and typed admission boundaries
remain intact; no additional unit is admitted. F006 stays closed only for that
bounded target-unit policy. It does not establish response fidelity, total
uncertainty, or astronomical truth.

### F007 — nonconformant and open

The candidate adds useful applied-state identity material:

- the complete `TelElAct` sample sequence and every per-array LOS-tau vector
  are hashed;
- active standard FIR state includes low/high frequencies, terms, and
  `a_gibbs`;
- active standard fixed-notch state includes zero-phase, centers, and derived
  widths; and
- every recorded dynamic notch retains scan, scope, detector, ordinal,
  center, width, and zero-phase, including exact duplicates.

The identity is still not collision-resistant over the complete actually
applied response:

1. PTC fixed line-audit notches are applied by
   `ptc_line_audit_impl.h`, including the shipped model-protected path with
   pre-filtering disabled. `calibration_response_identity` reconstructs fixed
   line-audit notches only when pre-filtering is enabled, and the application
   point records no fixed-notch history. With one identical request, a
   non-model-subtracted fruit state can skip the PTC pass while a
   model-subtracted state applies fixed notches; both can receive the same
   response/CALID/PKGID when dynamic history is empty.
2. The same shared and detector dynamic-notch helpers are used by RTC and
   model-protected PTC. Their records say only `pre_filter` or `post_filter`
   plus scope and geometry. They omit RTC-versus-PTC phase, PTC iteration,
   and model-subtraction state, so materially different placement histories
   can collide.
3. `requested_state_sha256` hashes the mutable typed/effective raw config,
   not the immutable `raw_timestream_plan.requested`; derived downsample
   resolution mutates that object. There is no separate effective-state
   identity. Mapmaker class/grouping are unconditionally labelled realized,
   including supported mapmaking-disabled/TOD-only operation.

Required production joins are also incomplete:

- TOD files are created during setup, but CALID/PKGID are added only by
  `Engine::add_tod_header`, reached through RawObs map output. Supported
  mapmaking-disabled TOD-only operation skips that route, leaving NetCDF TOD
  products without canonical joins.
- Coadd FITS are written after all observations through generic PHDU metadata
  that reads the current, last observation's singular calibration product.
  A multi-observation coadd with different CALIDs is therefore labelled with
  one last-observation identity; the coadd provenance has no complete CAL
  membership representation.
- Candidate FITS, TOD, and Beammap tests manually finalize or call helpers;
  they do not run the complete production finalizer, writer, package
  publication, reopen, and unique-resolution chain.

Minimal bounded requirement: record fixed/shared/detector applications at the
actual application point with RTC/PTC phase, scan, PTC iteration, and
model-subtraction state; bind immutable requested and actual effective state
separately; label only executed stages realized; carry finalized joins through
the TOD-only route; and represent multi-observation coadd CAL membership
unambiguously or fail closed. A new coadd-membership schema would be an
architecture choice and requires owner authority rather than auditor design.

### F008 — nonconformant and open

The mutex-protected applied-notch snapshot and swap-based consume operations
are atomic, and the normal production pipeline has one finalizer call after
TOD processing and before output. The public lifecycle does not enforce that
once-only invariant:

- `finalize_calibration_product_identity` ignores
  `applied_identity_finalized`;
- the wrapper consumes response history on every call; and
- a second finalization consumes an empty history and silently rewrites both
  CALID and PKGID.

A header-only executable counterexample compiled against the exact candidate
finalized once with a response-identity string representing one dynamic notch
and then with a no-notch response identity. It returned:

```text
finalized=true
calibration_changed=true
package_changed=true
```

The counterexample source SHA-256 is
`c8581745ca433dcb0c1ebe27b5818186e41b5ecf1c997752f97489e026ef451e`.

Applied-notch history is engine-lifetime state keyed only by scan. It has no
observation owner or boundary reset. Because the `if_available` finalizer does
not consume history for invalid or unavailable products, interrupted,
uncalibrated, or failed observations can leak stale entries into a later
observation. Scan-number reuse and incomplete multiscans are therefore not
safely separated.

Products are written before canonical raw-provenance/package publication. A
later missing, stale, conflicting, copy, or YAML publication failure can leave
already-published FITS/TOD/ECSV products whose CAL/PKG links do not resolve.
The package writer's own single-output rollback is sound, but it is not a
transaction for the preceding linked product set.

The candidate retains truthful fail-closed statements for unavailable
empirical response fidelity, total uncertainty, nuisance covariance,
donor-target covariance, and scientific precision/accuracy. No stronger claim
is promoted.

Minimal bounded requirement: own/reset history per observation; make repeat
finalization reject or idempotently retain the immutable consumed snapshot and
identities; exercise interrupted/unavailable/multiscan lifecycles; and publish
the canonical package transactionally before linked products or roll those
products back. Transaction design beyond the existing output authority is an
architecture choice and is not made here.

### Local F009 — nonconformant and open

Several positive properties are preserved. The writer verifies finalized
lineage, source existence and digest, conflicting destinations, staged-copy
digest, rename, and final digest. Its YAML write is atomic, and it removes a
newly created member if YAML publication fails. v1-v3 executable compatibility
remains intact. All ten materialized product contracts now contain a required
selected-APT entry, all seven reduction-product baseline options account for
the basename, and every profile retains a product-contract gate.

The baseline change is an exclusion from ordinary product comparison, not a
digest-integrity check. It is safe only if the separate package contract and
executable audit consumer verify the member; they do not.

Those synchronized declarations do not match or verify the application
surface.

#### Actual publication path contradicts every contract

`publish_completed_raw_timestream_provenance` passes
`engine.output_paths.obsnum_dir_name` to the writer. That directory is
`<reduction>/<six-digit-observation>/`; the writer places
`selected_calibration_apt.ecsv` directly there. The authoritative actual path
is therefore `{obs}/selected_calibration_apt.ecsv`, one per calibrated
observation.

All six base entries, inherited into ten materialized contracts, declare
`scope: reduction` and root pattern `selected_calibration_apt.ecsv`. Reduction
scope has one empty context and defaults to maximum cardinality one. An exact
layout probe with only `000042/selected_calibration_apt.ecsv` returned:

```text
passed=false
matches=[]
unclassified=['000042/selected_calibration_apt.ecsv']
error="pattern 'selected_calibration_apt.ecsv' matched 0; requires at least 1"
```

Thus real application output is both missing-required-at-root and
unclassified at its actual nested path. A root recursive glob alone would not
repair multi-observation cardinality.

#### v4 audit does not verify canonical membership or identity

The v4 semantic consumer checks only superficial availability, 64-lowercase-
hex identity shape, canonical member label/copy string, equality of two
declared digest strings, and observation/realized joins. It receives the raw
document but not its path, so it cannot hash the sibling APT. It does not
require canonical lineage schema, required component/digest presence, source
digest agreement, or recompute calibration/package identity.

Every bounded mutation below returned `semantic_errors=[]`:

- remove both selected-APT digest fields;
- replace the lineage schema with `not-canonical`;
- replace package identity with arbitrary `d` repeated 64 times and update
  the declared joins; and
- replace calibration identity with arbitrary `e` repeated 64 times and
  update the declared joins.

The checked-in v4 fixture itself declares selected-APT digest `cccc...cccc`,
while the checked-in ECSV's actual SHA-256 is
`f5c609c91ecc0bc83f3d84c4dfd0daf7b1542d149f915d40495368be60c75bd5`.
The fixture is accepted.

A combined executable counterexample exercised the raw-provenance gate plus a
bounded selected-APT contract entry, not a complete profile audit or full
materialized product contract. It placed an accepted v4 raw sidecar and valid
timestream-output sidecar under observation `000042`, no sibling selected APT,
and a parseable ECSV only at reduction root. Results were:

```text
raw_valid=true
raw_schema_ok=true
raw_semantic_errors=[]
contract_passed=true
contract_errors=[]
sibling_exists=false
```

This proves that the newly added raw-provenance and selected-member integrity
surfaces jointly accept a package layout that production never publishes and
whose declared member is not present beside its lineage. It does not claim
that an otherwise incomplete reduction would pass every full profile gate.

#### Supported uncalibrated v4 is falsely rejected

The writer always emits v4 and truthfully emits
`calibration_lineage.available=false` when effective flux calibration is
disabled. F006 deliberately permits that state. The v4 auditor unconditionally
requires available lineage and returns
`v4 canonical calibration lineage is unavailable` for a writer-shaped
uncalibrated document. The unconditional required contract member has the same
conditionality problem wherever an uncalibrated profile is applicable.

The new authority surfaces therefore fail in both directions: real
per-observation output is falsely rejected, while a forged root member and
declaration-only identities are falsely accepted. Overall production remains
governance-fail-closed, but the local package-integrity evidence path is not
fail-safe.

Minimal bounded requirement: use one exact `{obs}/selected_calibration_apt.ecsv`
contract context per calibrated observation; condition lineage/member
requirements on effective calibration; make v4 audit path-aware; require and
hash the sibling member; validate canonical schema, complete components, and
source/package/component digest joins; recompute package identity; replace
fixtures with internally consistent production-shaped lineage; and test
actual single- and multi-observation layouts plus missing, tampered, stale,
conflicting, uncalibrated, and partial-failure cases. Moving publication to one
reduction-root member would require resolving multi-observation identity and
would be a package redesign, not an auditor-prescribed repair.

### F010 — open, conditioned, and unchanged

No new authorized SCI-ALIGN, SCI-AST, exact-SHA Unity, astronomical, or
production evidence is supplied. F010 remains open and conditioned without
promotion.

## Deterministic validation on the exact candidate

A fresh Release build was configured and compiled in the audit worktree with
tests enabled. The normal FetchContent path could not use the network under
the audit prohibition, so the final configuration used locally available
dependency source trees from the independently identity-matched candidate
build and compiled all four audited candidate targets from scratch; no
application binary from the repair worktree was reused. Only third-party
warnings were observed.

| Check | Independently observed result |
|---|---|
| Fresh build targets | `citlali_cli`, `citlali_test`, `citlali_safety_test`, and `citlali_science_map_fits_products_test` built successfully |
| Full CTest | 670/670 runnable passed, 0 failed; 671 enumerated with pre-existing disabled `MapFitterLifecycle.ExactProductSequence` |
| Grouped normal/safety/FITS binaries | 618/618, 14/14, and 38/38 passed |
| Full baseline unittest discovery | 177/177 passed |
| Four changed baseline suites | 128 tests plus 26 subtests passed |
| Targeted atmosphere/product-contract Python tests | 27/27 passed |
| Full config preflight `--require-all` | 127/127 unit tests, 4 mode kits, 8/8 compatibility checks, 592 schema leaves, 100% coverage |
| Raw execution census | 82 records, zero review, zero drift, digest `3c581622eb930ba1296d7a70bb14b63176e53270703cadb83d8f0484aad25918` |
| Atmosphere artifact generator | exact: 1,368 rows and 72 series |
| Validation ledger | valid, 60 records |
| Science-change ledger | valid, 3 changes and 5 integration commits |
| Validation-profile registry | valid, 4 active and 8 preparing profiles |
| Parent-to-candidate `git diff --check` | passed |
| Candidate patch and changed-path inventory | exact expected SHA-256 and 26 paths |
| F005 production correction harness | reproduced three nonconformant approximate-derived weights and conformant full weight |
| Repeated-finalizer harness | reproduced changed CALID and PKGID on second finalization |
| F009 declaration-integrity mutations | all four forged/missing cases accepted with zero semantic errors |
| F009 actual-layout contract probe | real nested member rejected and unclassified |
| F009 combined gate probe | raw audit and root contract both passed with no sibling package member |

The green validation matrix demonstrates buildability and regression
stability. It does not invalidate the concrete production-path, lifecycle,
collision, and contract counterexamples above. No astronomical reduction or
source injection was required or performed.

## Finding ledger

| Finding | Independent successor-4 disposition |
|---|---|
| F001 | open, conditioned, unchanged |
| F002 | retain narrow structural closure, preserved |
| F003 | retain startup/admission closure, preserved |
| F004 | retain accepted APT lineage/association closure, preserved |
| F005 | open/nonconformant: production observation correction violates inverse-weight transfer for approximate, hybrid, and validated |
| F006 | retain bounded `mJy/beam` closure, preserved |
| F007 | open/nonconformant: incomplete applied-state identity and incomplete/ambiguous product joins |
| F008 | open/nonconformant: repeated finalizer, observation-history leakage, and non-transactional linked publication |
| local F009 | open/nonconformant: actual path mismatch, declaration-only integrity, forged acceptance, and uncalibrated false rejection |
| F010 | open, conditioned, unchanged |

## Owner decision brief

Do not accept complete SCI-CAL-001 successor-4 closure. Preserve F002, F003,
F004, and F006 only within their existing bounds; keep F001 and F010
conditioned; and return F005, F007, F008, and local F009 for minimal bounded
repair. F005 factor placement and any new multi-observation CAL-membership or
cross-output transaction schema require owner scientific/architecture choice;
this audit stops without prescribing them.

Retain contract `approved`, implementation `nonconformant`, validation
`in_progress`, production `fail_closed`, and verdict `amend`. Authorize no
production, Unity, merge, push, reduction, or downstream activity from this
re-audit.
