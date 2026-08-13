# SCI-CAL-001 successor-7 independent re-audit

Date: 2026-08-13

Auditor role: fresh, role-separated independent technical auditor

Candidate ref: `origin/codex/repair-sci-cal-001-successor-7`

Candidate commit: `9037314fd84241fa535c486d4ffb28966bb0394d`

Candidate parent: `211e2f16f6354609de3ce6c6ee526d8aa4c6c59c`

Candidate tree: `9d2095159ac208a1096519a6fa710172275d3b73`

Parent-to-candidate binary patch SHA-256:
`e761c4c25070ea7b925f8e75c05c5dbec05c1849132312f279112482dcded4e0`

Candidate changed paths: 22 modified, zero added, deleted, or renamed

Frozen coordination commit:
`87af719885fe73c1f292c97efb92026c4405ba05`

## Executive disposition

The candidate is **not a conforming completion of successor-7**. Return it
for bounded repair. Two independent F008 production routes lose required
CAL-linked FITS outputs under the candidate's new global staging lifecycle,
and the F009 semantic consumer accepts two invalid v4 package variants with
fully recomputed self-consistent identities:

1. Pointing raw/filtered observation and raw/filtered coadd data and noise
   FITS objects are staged, written, and then cleared without
   `publish_atomically()`. Destruction removes the stage. No new final is
   created; any old final remains stale.
2. Science Wiener-filtered observation and coadd data and noise FITS objects
   are written during filtering and then cleared without publication. The
   later output stage is intentionally skipped as "already written," so the
   missing final is not recovered.
3. Requested YAML containing boolean `true` is accepted against a recomputed
   requested-state preimage containing integer `1`, because the consumer
   compares parsed Python objects with equality rather than type-exact YAML
   scalar semantics.
4. A two-row package-local selected APT is accepted when the declared
   detector join and factor basis cover only the first row. The consumer
   validates referenced rows but never requires complete table-row coverage,
   unlike the production selector.

These are non-waived required-gate failures. The mandatory stop rule was
therefore applied: no fresh full candidate build, candidate test suite, CTest,
config preflight, baseline/product/profile/ledger/header matrix, failure-safe
matrix, or readiness execution is claimed. The independent source trace and
focused counterexamples are sufficient to reject completion; the unrun gates
must be rerun only after an owner-authorized repair.

The exact scientific axes are:

| Axis | State | Independent basis |
|---|---|---|
| Scientific contract | `approved` | No new scientific choice was made. F002-F006 remain preserved, and the F007 identity changes follow the frozen contract. |
| Implementation | `nonconformant` | F008 loses required production FITS finals and F009 admits two invalid package forms. |
| Validation/readiness | `in_progress` | Focused counterexamples fail the candidate. The broad deterministic matrix was stopped and is not reported as passed. The historical fixture gate remains failed, owner-waived, and never passed. |
| Production | `fail_closed` | No integration, launch, reduction, promotion, or production authorization is supplied. |
| Verdict | `amend` | **RETURN FOR REPAIR** under coordinator/owner control. Preserve conforming work; repair only the bounded F008/F009 defects and rerun the complete matrix. |

F001 and F010 remain open and conditioned because no newly authorized
external or observational evidence was supplied.

## READY checkpoint, authority, and role separation

Before substantive candidate source/diff/test-body exposure, audit branch
creation, edits, build, or tests, the required READY checkpoint established:

- exact local repair ref, remote-tracking ref, live-origin ref, commit, sole
  parent, tree, standard binary-patch digest, count-only changed-path
  inventory, and clean audit worktree/index;
- a physically separate repair worktree at
  `/Users/gwilson/.codex/worktrees/4b1b/citlali-refactor` and audit worktree at
  `/Users/gwilson/.codex/worktrees/01f1/citlali-refactor`;
- frozen coordination identity and immutable artifact hashes;
- absence of `codex/reaudit-sci-cal-001-successor-7-20260813` locally,
  remote-tracking, and live at origin before creation;
- the exact two-document audit ceiling, finding-by-finding exposure and
  counterexample plan, deterministic gate plan, historical-waiver treatment,
  prohibitions, and stop conditions; and
- an initially unresolved 22-path candidate versus frozen 22-path handoff
  membership discrepancy.

The coordinator then supplied the bounded supplemental owner/coordinator
authority. The final repair ceiling is 24 paths: the frozen 22 plus exactly:

1. `tests/test_config_scaffold.cpp`, solely for stale synthetic calibration
   package fixtures: reduced-observation identity, exact requested-state
   preimage/digest, and nonempty stale-stage rollback injection.
2. `tools/config/audit_raw_timestream_execution_reads.py`, solely to record
   the legitimate F008 `calibration.product` access in
   `beammap_apt_table_output_impl.h`, changing occurrence count 7 to 8 while
   retaining 88 records, zero review-required records, no drift, and digest
   `39855eee65816e816edee44e9a6271e7940158c2cbc26c018efa8f7e09fcbdc8`.

The candidate changes 22 of those 24. The authorized but unchanged paths are
`include/citlali/core/pipeline/reduction_observation_pipeline.h` and
`tools/baseline/examples/sci_cal_001_selected_calibration_apt.ecsv`.
Leaving them unchanged is conforming. No further path, scientific,
architectural, audit-artifact, or gate expansion was authorized.

The frozen authority artifacts rehash exactly:

| Artifact | SHA-256 |
|---|---|
| Successor-6 owner acceptance | `84a0bdc4a867a079a13db83200475c13237a9befaefde1ecec75165b2d9f0092` |
| Successor-7 bounded repair handoff | `1403473764e8a2c4bf8ba48137fa6f2065e2e0f89390a8f1b9d835941581ed14` |
| Successor-7 repair finding ledger | `fab261fc4122838e21be3bd731f079c23514edff651e7a3335f35fc4a01b23cf` |
| Immutable successor-6 re-audit report | `9f9305d3324f67dbcb4bac6a510115dc9c1108be7d7cc2ddd376e724653cfbbd` |
| Immutable successor-6 evidence | `0953e3030a27c2b3d6bd7c623413d4f4290b5d1e7ca2c7ca9608c2a8e10ab9be` |

After the coordinator accepted READY and clarified scope, the audit branch was
created directly at the exact candidate. A final live `git ls-remote` check on
2026-08-13 again returned the exact candidate and coordination heads.

## Candidate scope

The canonical name-status inventory has SHA-256
`dc84e49166a5cf4347fb13ccfce6992753fd33def52ce7ca481d4142d097946c`.
The candidate contains 2,062 insertions and 102 deletions in these exact 22
modified paths:

- `doc/REFACTOR_STATUS.md`
- `include/citlali/core/engine/detail/beammap_apt_table_output_impl.h`
- `include/citlali/core/engine/detail/beammap_map_product_writers_impl.h`
- `include/citlali/core/engine/detail/lali_output_impl.h`
- `include/citlali/core/engine/detail/map_phdu_output_helpers.h`
- `include/citlali/core/pipeline/atomic_yaml_output.h`
- `include/citlali/core/pipeline/calibration_product_admission.h`
- `include/citlali/core/pipeline/raw_timestream_execution_plan.h`
- `include/citlali/core/pipeline/raw_timestream_observation_shadow.h`
- `include/citlali/core/pipeline/raw_timestream_provenance.h`
- `include/citlali/core/pipeline/raw_timestream_provenance_lifecycle.h`
- `include/citlali/core/utils/ecsv_io.h`
- `include/citlali/core/utils/fits_io.h`
- `tests/test_config_scaffold.cpp`
- `tests/test_science_map_fits_products.cpp`
- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/examples/sci_cal_001_raw_timestream_provenance_v4.yaml`
- `tools/baseline/test_audit_reduction_run.py`
- `tools/baseline/test_compare_reduction_products.py`
- `tools/baseline/test_validation_profiles.py`
- `tools/config/audit_raw_timestream_execution_reads.py`
- `validation/validation_profiles.json`

All candidate paths are within the final 24-path authority. `git diff
--check` passes. The scaffold diff is confined to its three authorized fixture
purposes and weakens no production admission or failure assertion. The census
tool changes only its expected record digest; no root, exclusion,
classification, count, or suppression changes. The independently regenerated
census reports 88 records, review-required 0, drift false, and classification
counts 33/7/41/3/4. `calibration.product` changes from seven to eight
occurrences solely through the Beammap APT writer access.

No candidate diff exists in RTC coefficient arithmetic, RTC application,
selected-APT production filtering, the accepted F005 numerical correction
path, or the accepted fixed-atmosphere operator. The observed F008 change is
a required-output publication regression, not an RTC numerical change.

## Finding-by-finding disposition

### F001 — open, conditioned, unchanged

No newly authorized SCI-ALIGN, SCI-AST, exact-SHA Unity, astronomical-standard,
or observational evidence is present. F001 remains conditioned.

### F002 — bounded closure preserved

The accepted fixed-DJF25 artifacts are parent-identical and retain these exact
SHA-256 values:

| Artifact | SHA-256 |
|---|---|
| Operator contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Operator node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| Generated node header | `d322bdc863ccb1292325c739865f772ef53f4e9f4101967752027ea0a2413262` |
| Production operator | `3fd4352d05e77e07c1e354b7e4124733505064667968676d8d4e94315017d584` |

This is preservation of the accepted bounded structural claim, not a new
atmospheric-fidelity or uncertainty claim.

### F003 — bounded closure preserved

The typed reduction configuration validation and startup boundaries are
parent-identical. Calibrated non-`mJy/beam` output remains rejected, while the
intentional uncalibrated mode remains supported.

### F004 — bounded closure preserved

`calib.h` and `tests/test_calib_apt_filtering.cpp` are parent-identical. The
selected-APT load/filter/lineage boundary and production rejection of unused
APT rows remain unchanged.

### F005 — bounded local technical closure preserved

The production correction and arithmetic/lifecycle sources are unchanged from
the accepted successor-6 candidate. No candidate change broadens correction
composition, recipients, selected-APT mutation, or numerical behavior.

### F006 — bounded closure preserved

The validated calibrated production target remains top-of-atmosphere
point-source-peak `mJy/beam`; no additional calibrated unit was introduced.

### F007 — source-conformant repair retained; runtime gate not rerun

No F007 source counterexample was found:

- active FIR identity now includes the exact application sample rate used by
  `make_filter(engine.telescope.fsmp)`; inactive FIR omits coefficient fields;
- active IIR-highpass identity includes the exact application sample rate used
  by `iir_highpass(..., telescope.fsmp)`; inactive IIR omits coefficient
  fields;
- actual-notch serialization includes sample rate and
  `reduced_observation_identity`;
- reduced-observation identity is established before applied-response history,
  and the sole recorder backfills it for all fixed/configured/shared/detector
  notch routes;
- requested state remains a SHA-256 of the immutable requested node, effective
  scheduling/parameters remain distinct, actual history remains separate, and
  dormant stages do not contribute active coefficients; and
- the response preimage remains a length-framed CALID component, so response
  distinctions propagate to CALID and package identity.

The RTC filter/application sources are parent-identical; the identity change
does not alter RTC numerical behavior. Candidate test bodies cover dormant
stages, FIR and IIR sample-rate separation, notch sample rate, and
cross-observation actual-notch separation. Those candidate tests were
inspected but not executed after the F008 stop. F007 is therefore retained as
a conforming bounded implementation by source trace, but this audit does not
claim its independent runtime gate passed.

### F008 — open and nonconformant

The candidate has useful partial F008 work:

- the unchanged canonical ordering publishes the raw package before
  dependents;
- YAML publication stages, synchronizes, closes, reopens, parses, recomputes
  semantic/cryptographic identities, and atomically replaces;
- explicit FITS publishers flush, synchronize, close, reopen, validate HDU
  count/order/name/shape/readability and CALID/CALPKGID, and atomically
  replace;
- Beammap ECSV stages, synchronizes, closes, reopens, validates schema/cells,
  metadata and joins, and atomically replaces; `NaN`, `Inf`, and `-Inf`
  parsing/round-trip support is present; and
- no global transaction was introduced or is required.

The candidate nevertheless changes **every** write-mode `fitsIO` constructor
to write only `<final>.fits.tmp`, and its destructor closes and removes every
unpublished stage. Publication now requires an explicit
`publish_atomically()` call. Candidate production calls exist only in the
Lali and Beammap finalizers. Two remaining production route families clear
the owners without publishing.

#### F008-A — Pointing outputs are deleted instead of published

Affected production routes, including both data and noise FITS vectors, are:

- Pointing raw observation;
- Pointing filtered observation;
- Pointing raw coadd; and
- Pointing filtered coadd.

`Pointing::write_pointing_map_fits_products` adds PHDU and image HDUs but has
no publication call. `Pointing::output` then computes would-be published
paths, clears the data and noise vectors, and records publication. Under the
new destructor behavior the observable lifecycle is:

1. constructor removes an old stage and creates `<final>.fits.tmp`;
2. PHDU and image HDUs are written to that stage;
3. `<final>.fits` is still absent, or an older final remains unchanged;
4. normal vector clearing invokes the destructor;
5. the destructor closes and deletes the stage because `published_` is false;
6. no new final exists, yet downstream publication bookkeeping proceeds.

The compiled focused counterexample exercises this exact owner-clear
lifecycle and reports:

```text
stage_before_clear=true
final_before_clear=false
stage_after_clear=false
final_after_clear=false
```

It exits zero only when the stage existed before clear and both stage and final
are absent afterward. This proves loss on the normal Pointing lifecycle, not
merely an injected failure path. Existing finals are preserved, but only as
stale old products; that does not satisfy required publication.

#### F008-B — science Wiener-filtered outputs are deleted and never retried

Affected production routes, again including both data and noise FITS vectors,
are:

- science filtered observation; and
- science filtered coadd.

Science map filtering writes HDUs during the Wiener loop. Intermediate handle
destruction may close a library handle, but it never calls the atomic
publisher. `finalize_map_filter_fits_outputs` then only clears the two vectors.
The same destructor removes their stages. The later filtered-observation and
filtered-coadd output stages explicitly skip output because science products
are classified as already written during filtering. The observable result is
therefore an absent new final, or a stale pre-existing final, followed by
publication bookkeeping against paths that were not published.

Candidate FITS tests manually invoke `publish_atomically()` and therefore do
not exercise either missing production call site. The Pointing scaffold uses
fake counters rather than real staged FITS owners. No candidate test verifies
final-path existence and CAL joins through the actual Pointing or science
Wiener-filter lifecycle.

The required native close-error behavior is also not independently
established: the tested `after_close` checkpoint is an interruption injected
after `pfits.reset()`, not a demonstrated underlying library-close error.
This is an additional unclosed validation obligation, although F008-A and
F008-B already require rejection.

### Local F009 — open and nonconformant

Several F009 mechanisms are materially present: owning-directory and
package-observation joins, full observation/realized calibration field joins,
component/CALID/PKGID recomputation, current exact-v4 admission, generic v1-v4
historical recognition, and effectively uncalibrated v4 handling with
unavailable lineage and no selected-APT sibling.

They do not reject the following two independent invalid packages.

#### F009-A — requested YAML scalar type collision is accepted

The counterexample starts with a production-shaped v4 package whose requested
configuration contains:

```yaml
flux_calibration:
  enabled: true
```

Its serialized requested-state preimage instead contains integer `1`. The
counterexample recomputes the preimage digest, response provenance,
calibration identity, package identity, and observation/realized joins. The
consumer returns `errors: []`.

The decisive code parses the preimage with `yaml.safe_load` and compares the
parsed mapping to requested state using Python `!=`. Python considers
`True == 1`, so a type-different requested preimage is accepted. Digest and
identity recomputation only authenticates the forged preimage itself; it does
not repair the non-type-exact comparison. Candidate tests cover added keys,
text/digest mutation, and declared digest mismatch, but not YAML scalar type
collisions.

#### F009-B — an unbound extra selected-APT row is accepted

The counterexample creates an exact two-row package-local ECSV and recomputes
its artifact digest, row-association digest, raw-acquisition binding,
factor-state digest, CALID, PKGID, and observation/realized joins. The declared
ordered detector rows and detector `flxscale` vector cover only row zero. The
consumer again returns `errors: []`.

`validate_selected_apt_factor_binding` requires the number of declared joined
rows to equal the factor vector and checks that each referenced source index is
unique, in range, and value-consistent. It never requires the table row count
to equal the declared row count or requires the referenced index set to cover
every package-local row. Production selection explicitly rejects any unused
APT row. A self-consistent partial package can therefore pass consumer
validation even though production could not have emitted it.

Exactly seven historical profiles are restored from the overbroad basename
exclusion to the predecessor-scoped
`*/selected_calibration_apt.ecsv` pattern: the three phase-4 profiles, two
phase-5 profiles, and two SCI-MAP-001 profiles. No other profile uses that
restoration. The SCI-CAL v4 epoch and all four current profiles remain
`preparing`, have no accepted baseline, and do not exclude their package-local
selected APT. No production promotion occurred.

### F010 — open, conditioned, unchanged

No newly authorized external or observational evidence changes F010. It
remains conditioned.

## Historical truth and immutable evidence

The historical fixture gate remains exactly
`failed_owner_waived_never_passed`; it is **not passed**. Its only accepted
failure is six Pointing plus twelve science historical product errors caused
by forbidden `sig2noise_pixel_I`. That field remains prohibited for all
new/current products. The audit did not rerun this gate after mandatory stop
and does not convert the recorded waiver into a pass.

The governing objects are candidate-unchanged and retain these SHA-256 values:

| Object | SHA-256 |
|---|---|
| `validation/accepted_runs.json` | `4a134dcdd14e0444d96875547f628a3353574cc66574dd9a559bcf59dafb94bb` |
| `validation/phase5_validation_readiness.json` | `b9daf6ab3973d2d35968ab27d2b7c75eca8534d2baeb6af9bb43725261f04755` |
| `tools/baseline/phase5_readiness.py` | `3a27fa5279c75432aa0939cbcc2add2db4d30df92379f6bf511ff281202b2af7` |

The `sci-cal-001-production-candidate-2026-08-12` epoch and
`sci-cal-001-current-{point,oof,science,beammap}-v1` profiles remain
`preparing`. This audit supplies no promotion or production authority.

## Executed checks and mandatory skips

Executed evidence is limited to read-only identity/source checks, one
standalone C++ lifecycle counterexample, one standalone Python semantic
counterexample script, and the bounded raw-execution census:

- candidate commit/parent/tree/ref/path count/patch digest and `diff --check`:
  passed;
- live candidate and coordination branch heads: passed on final retry with
  network access; the initial sandboxed lookup failed only because DNS was
  unavailable;
- frozen authority and immutable evidence hashes: passed;
- final-authority path membership and both supplemental-path purpose checks:
  passed;
- raw execution census: passed, 88 records, digest `39855eee...`, review 0,
  drift false;
- F008 compiled owner-clear lifecycle counterexample: reproduced missing
  final with exit zero;
- F009 requested-type and extra-row semantic counterexamples: both reproduced
  `errors: []`; and
- parent-identity/source traces for F002-F007 and candidate source/test-body
  inspection: completed.

The following were deliberately **not run** after the first mandatory stop
condition and are not claimed passed:

- fresh full candidate configuration/build or `citlali_cli` build;
- focused candidate C++ tests or full candidate test binaries;
- full CTest;
- full public-header isolation/linkage matrix;
- full config preflight with `--require-all`;
- baseline Python discovery and focused baseline/product comparisons;
- validation product, profile, validation-ledger, and science-change-ledger
  gates;
- full failure-safe YAML/FITS/ECSV/TOD publication suite;
- atmosphere generator check, session-exit census, and other complete census
  matrix beyond the one supplemental raw-execution census; and
- ordinary Phase-5 readiness and `phase5_readiness.py --verify-fixtures`.

Candidate or prior-run claims about those gates are not independent evidence
for this audit.

## Recommended owner disposition

**RETURN FOR REPAIR / amend.** Do not integrate, merge, push, launch a
downstream task, run a reduction, promote the SCI-CAL v4 epoch, or authorize
production from this audit.

An owner-authorized successor should be bounded to:

1. making every Pointing and science Wiener-filtered data/noise FITS production
   route explicitly complete the per-artifact stage/synchronize/close/reopen/
   validate/atomic-replace lifecycle before publication bookkeeping;
2. enforcing type-exact requested-config preimage comparison; and
3. enforcing complete selected-APT table-row coverage by the package-local
   detector join/factor basis.

The successor should preserve the conforming F002-F007 work, historical
waiver truth, immutable evidence, exact profile statuses, and numerical
behavior. After repair, a fresh independent audit must rerun the complete
non-waived deterministic gate matrix. This document does not authorize that
repair or another re-audit.
