# SCI-NOI-002 Cycle 2 independent re-audit — 2026-08-06

Status: **re-audit complete; verdict `amend`; exact repair commit is not ready
for application integration**.

No P0 defect was found. Seven P1 findings and three P2 findings prevent the
Cycle 2 repair assertions from closing. The most immediate P1 is an entry-path
lifecycle error: the normal processor session begins noise publication before
the reduction directory exists, so a fresh session throws before its reduction
iterations. The same bookkeeping state is also scoped to a whole session while
FRUIT output and mapmaking provenance are iteration-scoped, which makes an
enabled multi-iteration reduction accumulate stale members and counters and
then fail completion. The active validators additionally accept semantically
impossible disabled packages, do not implement exact product joins, and cannot
exercise the production response-invalid branch of filtered-scatter validity.

This is a documentation-only independent audit of exact pushed repair commit
`d1d19145df574571a894772fdc9410c86cba1041`. It is not a continuation or
repair of that commit. It changes no application, test, configuration,
canonical ledger, or registered handoff authority; performs no integration,
push, Unity access, astronomical reduction, production action, or finding
repair; and stops for owner/coordinator review.

## Exact target and provenance

- Worktree:
  `/Users/gwilson/.codex/worktrees/2069/citlali-refactor`.
- Dedicated audit branch: `codex/reaudit-sci-noi-002-cycle2`.
- Exact target and entry `HEAD`:
  `d1d19145df574571a894772fdc9410c86cba1041`.
- Exact target parent:
  `0bc4d95d6bb2117442d0ccdb79c57e42e0b79989`.
- Target tree: `826f601e2c3447765e5d1b25285ac365ee3fd120`.
- Parent tree: `25f39d8b1ba2527c2a154a69a527b0d8835a412a`.
- `git rev-list --count parent..target` returned one and the target/parent
  merge base is the parent.
- After a fresh fetch, `origin/codex/repair-sci-noi-002` resolved exactly to
  the target. An audit/coordination branch was never used as an application
  base.
- Entry worktree and the final pre-documentation worktree were clean.
- Target subject: `repair: harden SCI-NOI-002 package lifecycle`.

Frozen Cycle 2 authority was read from exact Git object
`287840715c8e4ae778ce57b7166de86e7b7dfa9c`, tree
`d56ddbd4583a7cf9dd7ee5a27095db469b632cd4`, not
from the target worktree or a task narrative. The following identities were
recomputed over the exact Git-object bytes before any artifact was relied on:

| Frozen authority/evidence | SHA-256 |
| --- | --- |
| `doc/audits/prompts/SCI_NOI_002_CYCLE2_REPAIR_PROMPT.md` | `42ba64a459f8496d4f167cc9fd32a0634023b37252249becd850b64b626e889d` |
| `doc/audits/handoffs/SCI-NOI-002/SCI-NOI-002_CYCLE2_REPAIR_AUTHORITY_MANIFEST_2026-08-06.yaml` | `36602cd0ba779a3fe9d9419c3b0ed53d86e26eb4c887ec7b5d19c0442ad86202` |
| `doc/audits/packages/SCI-NOI-002_CYCLE2_OWNER_DECISION_AND_REPAIR_HANDOFF_2026-08-06.md` | `3b8587564418632bcf9cddd417ebb30d08374127fd75420179c7b6050187559d` |
| `doc/audits/packages/SCI-NOI-002_CYCLE2_REPAIR_DISPATCH_READINESS_2026-08-06.md` | `1ef66a8436d14d95dfb0fcf369706e743f7a3bf3248a72b4849cebed7e623dfa` |
| prior `SCI-NOI-002_INDEPENDENT_REAUDIT_2026-08-06.md` | `a66ef3f17976a7149ef04d2fec08e2c1faa2947c4b883fddd651a3eb57e44517` |
| prior `SCI-NOI-002_REAUDIT_LEDGER_UPDATE_PROPOSAL_2026-08-06.yaml` | `6f9410fc2015bccba49001e337b063660387edd39f1879c0d5de7100dd1c970e` |
| first repair prompt | `45fc19d6ccf0e55aa1c2a1189d97f72ffb9b51027c659580d7cbcc9415c4bc71` |
| first repair authority manifest | `6f4b84995c8cb118bdb182b9189c4b95464c4fe7e414b762debb8786e825ce79` |
| first repair dispatch readiness | `246c751397b2d138939372c7943e8187ba36658bbb7a5b8f96bb01edca0f4804` |
| owner decision brief | `3520172cfc11e8e34f280f9ebdf147ea414c7a3a4ca6109bad55354a5ff3cf71` |
| original exact-application final audit | `2874ffe950aed769f73277ed8f60ecab8860692d24e7c541f05a47a041a8a40d` |
| frozen independent mathematical core | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` |
| original noncanonical ledger proposal | `5574d8e34fcfba8f4709d5848e79732ff3557a9817be25604375b9f3d4ec278d` |
| SCI-NOI-001 R3 bounded ensemble evidence | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` |
| SCI-NOI-002 inbound XAUD-001 | `dfcd59e9d59395ba84f7dfed1656690daae694872c2a1a40bf4f5c79f6abed3` |
| SCI-NOI-002 inbound XAUD-002 | `9eb6c778409d344ce73387f44ac4a5429a89d43b8e768c19bf3da1ed6967c1e5` |
| SCI-NOI-002 inbox manifest | `1de9dbc5f9aca42f5a9f9b1f05b7b14c092ee9b443946d0c52eeb27c8da117b0` |

The manifest-listed outgoing proposals were also verified, without treating
them as returned evidence: FLT-001 XAUD-003
`47b5533ba88e3dfac19c5beda2e92ff84d149bf3d50a41ec3740fa1b9615d9d7`,
FLT-002 XAUD-001
`655054b9253bfb0023f3598c2606ab15ea4d29d404a3d5ef4345c26c078c1249`,
SRC-001 XAUD-002
`258325b407757dcc716ec4d53db300e56bb896a93f993dc500f1080050399544`,
MODE-001 XAUD-003
`6b12c60277f9dd408a21f1b31e3470aaebc84ed746a39b105ceb9669224e11b8`,
FRUIT-001 XAUD-002
`000583fe2ab82adb926b9c8c9d9f78829eb88c1d980a1208bdc3e3c45a9a7fcd`,
and BEAM-001 XAUD-003
`ebaf770d7269474f0709b20c6faceb357a67ae0e4668002679ad47d7292cb312`.

## Independently derived contract

The frozen objects establish this bounded contract independently of the repair
narrative:

1. The estimator remains the empirically centered second moment of the exact
   completed `source_imprinted_current` realization stack,
   `V_p = (1/R) sum_r (Y[r,p] - mean_p)^2`. It is descriptive stack scatter,
   not iid-unbiased sample variance, physical-noise variance/covariance,
   inverse variance, precision, calibrated significance, aperture uncertainty,
   count adequacy, or a production calibration.
2. **RA-B001:** package membership comes only from the explicit current-run
   list of successfully published members. Paths are normalized, unique,
   inside the reduction root, regular, and non-symlink. Every member has a
   full-file SHA-256; the ordered inventory has an unambiguous aggregate
   digest; FITS, ECSV, and NetCDF joins are exact; and only an atomic final
   publication may assert completion. Recursive directory discovery is
   forbidden.
3. **RA-B002:** a disabled result retains the request, resolves effective
   enablement/count to false/zero, reports generation not executed, makes all
   realized counters available zero, marks outputs completed, and uses
   `effective_disabled_zero_work`. An enabled request with zero realizations
   remains invalid.
4. **RA-B003:** plan-derived expectations and observed counters are separate.
   Observed values advance only at successful existing publication/finalization
   boundaries, and incomplete or partially published work may not appear
   complete. No per-sample ID, sign stream, or persistent realization ledger
   is authorized.
5. **RA-B004:** filtered-scatter validity is derived from the actual calculated
   scatter, the actual response, and the actual support. `R_lt_2`, unavailable
   or nonfinite scatter, invalid response, and invalid support fail closed with
   distinct truthful reasons and NaN values in canonical identity and every
   alias.
6. **RA-R001:** no canonical/legacy duplicate physical HDUs. One compatible
   legacy `EXTNAME` plane may remain only when its join carries the truthful
   canonical identity, version, digest, validity, and restriction metadata.
7. **RA-R002:** Mapdiag stored names and calculations remain compatible and
   unchanged; descriptions and identity/restriction metadata are truthful; no
   duplicate canonical variables are added.
8. F001, F002, and F008 were accepted closed by the first repair. F003, F004,
   and F007 were open for Cycle 2. F005 remains `open` with disposition
   `open_conditioned` and filter-parity status exactly
   `scope_blocked_not_applicable_pending_FLT` regardless of local metadata
   work. F006 remains `open`, manifest state `held_external`, and wholly
   SCI-FRUIT-001-owned; FRUIT behavior and its configuration dependency must
   not change.
9. No FLT/SRC/FRUIT/MODE mathematics, dense covariance, per-sample IDs,
   R/I/Q/phase behavior, primary-channel substitution, or duplicated full
   package metadata is admitted. Auxiliary measured channels remain future
   evidence only and cannot replace primary-x noise semantics.

## Methods and complete diff accounting

The parent-relative diff was inspected in full, including every call site and
its surrounding lifecycle. It changes 21 existing files, with 2,380
insertions and 547 deletions; no path was added, deleted, or renamed.
`git diff --check parent target` passed.

The review traced:

- request/effective/expected/realized noise-plan state;
- the processor-session, reduction-iteration, FRUIT, observation, coadd,
  filter, Beammap, Mapdiag, source-table, and final sidecar lifecycles;
- all six writer hooks in `beammap_map_product_writers_impl.h`,
  `lali_output_impl.h`, `map_filter_execution_impl.h`,
  `mapdiag_output_impl.h`, `pointing_output_impl.h`, and
  `source_table_output_impl.h`;
- path normalization, symlink handling, duplicate admission, join parsing,
  member hashing, aggregate hashing, revalidation, pending-file cleanup, and
  atomic rename;
- product identity and physical-HDU multiplicity;
- the C++ finalizer, Python active auditor, JSON contract, and exact tests;
- all numerical code changes in `map.cpp` and all Mapdiag value bindings.

Two negative semantic probes were constructed directly against the active
auditor. A disabled/no-work document with a three-member FITS/ECSV/NetCDF
inventory was accepted with `semantic_errors=[]` and `package_errors=[]`.
A requested-enabled/count-zero document with mapmaking disabled was also
accepted with `semantic_errors=[]`. These are evidence of findings below, not
successful closure gates.

## New findings

### P1-001 — publication starts before a fresh reduction directory exists

`run_reduction_processor_session` calls
`begin_noise_product_publication(engine.output_paths.redu_dir_name, ...)` at
`include/citlali/core/cli/reduction_execution.h:161` immediately after leasing
the configured output root. `OutputPathState::redu_dir_name` defaults empty at
`include/citlali/core/pipeline/output_path_state.h:9`; it is first reset and
assigned by `TimeOrderedDataProc::create_output_dir` at
`include/citlali/core/engine/detail/todproc_raw_input_impl.h:44,49-59`, which
is reached later from per-iteration setup. The begin function requires the
path already be an existing directory and throws otherwise at
`include/citlali/core/pipeline/noise_provenance.h:155-158`.

Consequences:

- a fresh ordinary processor session throws before its first reduction
  iteration;
- a reused processor can point at the preceding reduction directory and delete
  its complete/pending noise authority before the new iteration changes the
  path.

The lower-level `prepare_and_run_cli_reduction_pipeline` and
`run_cli_reduction_pipeline` templates also contain no begin step, although no
repository call site establishes them as independently supported session entry
points. There is no test that traverses the actual processor-session path after
the new call. This is an application blocker and stale-run authority defect
under RA-B001/RA-B003.

### P1-002 — session-scoped counters and members corrupt FRUIT iteration semantics

This finding is latent behind P1-001: it becomes observable once publication
begin is moved after the first output-directory creation. The noise begin step
then clears expected state, realized state, and member state once per session
(`noise_execution_plan.h:223`). Mapmaking state is reset per iteration
(`mapmaking_execution_plan.h:110-118`), output roots may be reused or replaced
per the iteration policy (`iteration_output_layout.h:13-44`), and enabled FRUIT
requires at least two iterations (`fruit_loop_activation_validation.h:49`).
The new writer hooks add observed counters at every output
(`noise_execution_plan.h:321`) and append every member, while final expected
counts are derived only from the completed final-iteration mapmaking plan
(`noise_execution_plan.h:435`).

With `save_all_iters=false`, subsequent iterations reuse output names and
produce duplicate admissions; with `save_all_iters=true`, earlier iteration
members are outside the final reduction root. In either case counters are
accumulated across iterations while expectations are not, so final completion
throws. Final noise generation is intentionally skipped after a FRUIT feedback
map exists (`mapmaking_dispatch.h:49`), but the retained positive-size buffers
make later hooks add their full cardinality again
(`noise_execution_plan.h:355-415`). Session-level
`generation_executed=true` remains truthful for an ordinary run because its
first iteration generated noise; it would become untruthful current-run
evidence only for a restarted invocation that begins with a loaded feedback
map and generates no new stack.

This violates RA-B001 and RA-B003 and changes the NOI/FRUIT interface behavior
by turning an otherwise admitted multi-iteration path into a failing
publication lifecycle. No FRUIT mathematics, configuration dependency,
default, threshold, iteration rule, add-back, or stopping law was edited, but
the assertion that F006 execution behavior remained unchanged is false.

### P1-003 — completion semantics are not reconciled to package identities

The finalizer correctly sums joined realization HDUs and compares that number
to `realization_image_write_count` (`noise_provenance.h:554-570`). That is its
only member-to-realized-semantic cross-check. It does not enforce that
disabled/no-work state has no stack-derived empirical member, that
enabled/product-enabled state has the expected empirical identities, or that
the member inventory agrees with `empirical_product_map_count`. Mapdiag and
source-table hooks also append NetCDF/ECSV members unconditionally after their
writers return (`mapdiag_output_impl.h:101-115` and
`source_table_output_impl.h:31-40`).

The added C++ happy-path fixture itself initializes a default disabled plan,
then admits a physical FITS `conditional_finite_stack_scatter`, finite joined
NetCDF diagnostics, and a joined ECSV table and publishes the package complete
(`tests/test_config_scaffold.cpp:3185-3187,3200-3226,3288-3304`). The
stack-derived FITS/NetCDF products decisively contradict disabled zero work;
the ECSV member is not by itself realization-derived. The active Python auditor
accepts the equivalent impossible package. Conversely, an enabled package can
omit expected empirical products and still pass if its realization count
happens to agree.

This breaks the disabled zero-work meaning in RA-B002 and permits incomplete
or contradictory authority under RA-B001/RA-B003.

### P1-004 — member safety and FITS/ECSV/NetCDF joins are not exact

The writer emits ten FITS join fields, including digest-kind and missingness,
but both C++ and Python validators inspect only eight and omit `NOIDGKND` and
`NOIMISS`. Scope, validity, and restriction are accepted merely when nonempty,
not when equal to the declared product identity. FITS identities are stored in
a set, so duplicate non-realization canonical identities in multiple physical
HDUs collapse instead of failing; repeated realization identity is separately
counted and is expected.

The C++ ECSV validator uses whole-header substring searches rather than a
structured exact mapping and omits digest-kind/column binding, while the Python
validator parses a mapping. Both NetCDF validators substring-match comments
rather than parsing an exact structured join; the defined comment join also
has no semantic-digest/digest-kind fields. C++ and Python therefore do not
enforce one exact join language. Separately, the active semantic auditor does
not validate several authoritative sidecar sections; compact members need not
duplicate those sections, but package authority must still validate them.

The C++ producer rejects a leaf symlink, canonicalizes an in-root intermediate-
symlink alias to its regular target, and deduplicates/stores that canonical
relative path. It does not establish a symlink-free path chain. The Python
auditor accepts intermediate symlink components and deduplicates the lexical
relative spelling before resolution, rather than the resolved file identity.
There are also residual mutation windows between revalidation and final rename;
the marker rename itself is atomic, and eliminating the window would require a
stronger handle/snapshot contract.

The repair is materially stronger than recursive discovery—it uses explicit
members, canonical in-root resolution, full SHA-256, revalidation, and an
atomic pending-to-final rename—but the frozen requirement was exact joins and
non-symlink membership. RA-B001 and RA-R001 are not closed.

### P1-005 — production filtered validity does not use actual response/support

`filtered_scatter_validity` exposes the four requested reason labels and its
unit test injects them directly. Production publication, however, passes the
raw-parent `science_policy_support` mask and a hard-coded
`response_normalization = 1.0` (`map_image_output_helpers.h:699-715`). Project
authority says `science_valid` is the only authoritative raw-validity mask and
that downstream raw validity remains separate from local support and response
(`doc/SCIENTIFIC_CONVENTIONS.md:265-266,296-299,504-511`); the map implementation
shows `science_policy_support` and `science_valid` can differ
(`src/citlali/core/mapmaking/map.cpp:541-562`). The production path therefore
can admit a raw-invalid pixel and represents no operator-local filtered support.

No actual calculated response is derived or passed; `RESPNORM=1.0` is itself
described as “Legacy identity response; no aperture/template calibration”
(`fits_image_metadata_keys.h:104-106`). Consequently `response_invalid` is
unreachable through production output. The helper also conflates reason
classes in a mixed case: if scatter is finite somewhere off support but every
supported scatter value is nonfinite, it returns `support_invalid` instead of
`scatter_unavailable_or_nonfinite` (`map_image_output_helpers.h:570-593`). The
fixture only injects all-finite/all-nonfinite scatter and all-one/all-zero masks
(`tests/test_science_map_fits_products.cpp:401-438`), so neither production
authority nor the mixed case is tested.

Fail-closed NaN handling is present once this helper returns unavailable, and
aliases use the same copied arrays/status. That local mechanism is correct but
its production inputs do not satisfy RA-B004. If unity identity response was
intended to satisfy “actual calculated response,” that is a material frozen-
authority ambiguity; use of raw policy support remains inconsistent with the
repository's explicit raw-validity authority. The repair cannot silently
resolve either issue in its own favor.

### P1-006 — split Beammap bookkeeping counts unselected maps as published

Split Beammap writes only detector indices selected by configured flag values
(`beammap_map_product_writers_impl.h:145-201`). It records the split files,
then records output-stage counts from the full `MapBuffer`
(`beammap_map_product_writers_impl.h:266-281` and
`noise_execution_plan.h:401-415`). When configured flags select a strict
subset and realization images are enabled, the admitted FITS realization-HDU
count is smaller than the claimed full-buffer write count and finalization
throws (`noise_provenance.h:554-570`). With the same strict subset and no
realization images, the package can instead claim empirical/scientific map
counts for unselected maps because those counts are never reconciled to
identities.

This is a selection-accounting/cardinality mismatch caused by an ostensibly
bookkeeping-only hook and violates RA-B001/RA-B003. The existing selection is
unchanged. No strict-subset split-noise lifecycle fixture exists.

### P1-007 — literal enabled-zero authority is not implemented

`NoiseExecutionPlan::reset_from_request` rejects zero only when mapmaking is
already enabled (`noise_execution_plan.h:176`). The Python auditor likewise
requires positivity only for the effective enabled state. Thus a retained
request with `enabled=true`, `n_noise_maps=0`, and mapmaking disabled resolves
to disabled zero-work and is accepted; the direct probe returned no error.

The frozen owner decision states without qualification: “Enabled-zero remains
invalid.” The prior re-audit also states that enabled requests with count
`<=0` are rejected. Under that literal authority this is a P1 contract breach.
If the coordinator intended only *effectively enabled* zero to be invalid, the
frozen authority is ambiguous and must be clarified rather than waived by this
audit.

### P2-001 — two compatible legacy HDUs still duplicate one physical plane

The repair removes the candidate-added canonical HDUs, but it retains both
`sig2noise_*` and `sig2noise_pixel_*`, writing the same
`coefficient_standardized_signal` matrix twice with the same canonical product
identity (`map_image_output_helpers.h:652`). The Cycle 2 owner decision permits
“one physical plane” under a compatible legacy name and requires tests to
reject duplicate planes. The registry/tests instead require both. This retains
one extra `8N`-byte payload per coefficient-standardized plane/map, and those
bytes participate in each full-file hash pass.

This duplication predates Cycle 2's new canonical HDU, but the singular owner
policy is explicit. RA-R001 is only partially satisfied.

### P2-002 — aggregate digest encoding and auditor order are ambiguous

The aggregate preimage concatenates raw relative path, file digest, and decimal
size separated by newlines (`noise_provenance.h:594-596`). A valid POSIX filename
may contain a newline, so the encoding is not injective. The digest kind calls
this representation canonical without a length prefix or escaping. The C++
writer sorts members, but the Python auditor recomputes in supplied YAML order
(`audit_reduction_run.py:1573`) and never requires lexical order, so a reordered
package with a recomputed digest remains valid to the active auditor.

Full member hashes remain sound; this is a deterministic aggregate-identity
and cross-validator defect under RA-B001.

### P2-003 — RA-R002 tests do not prove unchanged values or exact metadata

The inspected Mapdiag implementation retains the three stored variable names
and their existing bound value vectors, adds no duplicate variable, and changes
only comment construction. It therefore passes RA-R002 at the implementation
axis. The added fixture, however, supplies one identical vector for every
field, captures calls in a `std::map`, and checks only selected names plus one
identity substring. It cannot detect swapped bindings or duplicate calls and
does not assert exact comments/restrictions, absence of old wording, or exact
unchanged values. RA-R002 validation remains incomplete; this is not evidence
of a Mapdiag numerical change.

## Decision-by-decision disposition

| Decision | Disposition | Independent basis |
| --- | --- | --- |
| RA-B001 | **not closed** | Explicit nonrecursive membership, full file hashes, canonical in-root checks, revalidation, and atomic final rename are present. P1-001/003/004/006 and P2-002 leave lifecycle, exact-join, non-symlink, completeness, and aggregate-identity requirements open. |
| RA-B002 | **not closed** | Serialization and the active auditor support the intended available-zero representation. P1-003 contradicts disabled zero work, and P1-007 conflicts with the literal enabled-zero rule. |
| RA-B003 | **not closed** | Expected and realized fields are structurally separate and counters advance after writer finalization on simple paths. P1-001/002/003/006 show stale scope, partial publication, and selection/count mismatches. |
| RA-B004 | **not closed** | Helper reason classes and NaN masking exist, but production does not supply actual response or filtered support (P1-005). |
| RA-R001 | **not closed** | Candidate-added canonical planes were removed, but duplicate compatible legacy planes remain and duplicate canonical identities are not rejected (P2-001/P1-004). |
| RA-R002 | **implementation pass; validation incomplete** | Stored names, calculations, and value bindings are unchanged; comments are truthful and no duplicate variable was added. P2-003 records the missing exact regression proof. |

## Original finding disposition

| Finding | Cycle 2 re-audit disposition |
| --- | --- |
| F001 | **closed remains supported** within the finite-stack descriptive estimator and deterministic limits. |
| F002 | **closed remains supported** for the global existing-use-only nonprecision scale identity and restrictions. |
| F003 | **open**. Package membership is no longer recursive, but lifecycle, semantic reconciliation, path safety, exact joins, and aggregate identity remain nonconformant. |
| F004 | **open/conditioned**. Canonical identities and much terminology are corrected, but RA-R001 remains open and exact join/alias validation is incomplete. |
| F005 | **open; disposition `open_conditioned`; parity status `scope_blocked_not_applicable_pending_FLT`**. Local fail-closed code cannot close filter parity, and this audit does not waive or broaden the boundary. |
| F006 | **open; manifest state `held_external`; SCI-FRUIT-001-owned**. No FRUIT mathematics or configuration dependency was edited, but Cycle 2 bookkeeping changes multi-iteration execution behavior at the NOI/FRUIT interface (P1-002). The existing FRUIT handoff remains the applicable route. |
| F007 | **open/conditioned**. Separate fields and simple-path counters exist, but current-run, iteration, actual-generation, member, and split-selection truth are not established. |
| F008 | **closed remains supported** within the approved conditioned ensemble identity and prohibited interpretations. |

## Six writer hooks and scope-creep assessment

On a successful, single-iteration, nonsplit path, each of the six approved
writer changes is mechanically bookkeeping after an existing close/atomic
write boundary: it captures existing filenames, appends membership, or adds
aggregate counts. Their direct diffs do not modify a numerical array,
filename constructor, product ordering, filter selection, threshold, default,
or algorithm.

That local statement is not sufficient for the global claim. P1-002 and
P1-006 show that the hooks change completion behavior and misaccount existing
selection for FRUIT and split Beammap because the bookkeeping authority is
scoped and counted incorrectly. P1-001 also puts lifecycle initialization on
the wrong side of directory creation. The hooks are therefore syntactically
bookkeeping-only but not behavior-preserving across all approved modes.

The only intended scientific numerical-output effect is fail-closed NaN for
invalid filtered scatter/response/support. It is implemented both in
`src/citlali/core/mapmaking/map.cpp` and by copy-and-mask publication in
`map_image_output_helpers.h`; no other numerical algorithm change was found.
Mapdiag calculations and bindings are unchanged. The authorized removal of
candidate-added canonical HDUs is a storage/product-layout change, not a
numerical or selection change. No other estimator, selection threshold,
configured count/default, FLT/SRC/FRUIT/MODE/JINC/RTC/PTC mathematics, dense
covariance, per-sample identity/sign stream, R/I/Q/phase behavior, or auxiliary
measured-channel substitution was found. Full package semantics remain
centralized in the sidecar; per-product joins are compact rather than copies of
that package.

No unauthorized product class, persistent realization ledger, generic
publication/filter framework, or new file was added. Scope is therefore
scientifically bounded, but the authorized lifecycle work contains correctness
defects and a forbidden practical F006 behavior regression.

## Deterministic local verification

All commands used the repository's local toolchain and
`/Users/gwilson/tolteca/bin/python`. No Unity host was contacted.

| Gate | Result |
| --- | --- |
| clean entry, exact target/parent/remote/tree, one-parent accounting | pass |
| target `git diff --check` | pass |
| Homebrew Release configure with tests | pass after the authorized dependency download completed |
| `cmake --build build --target citlali_cli citlali_test citlali_science_map_fits_products_test citlali_safety_test -j 8` | pass |
| `citlali_cli --version` | `v4.0.0-3635-gd1d19145d` |
| focused core noise/lifecycle tests | 32/32 pass |
| focused FITS noise-product tests | 8/8 pass |
| full `citlali_test` | 563/563 pass; one separately disabled test |
| full science-map FITS binary | 29/29 pass |
| full safety binary | 14/14 pass; its deliberate exception-reporting fixture emitted the expected critical message |
| full CTest | 606/606 executed tests pass out of 607 registered; `MapFitterLifecycle.ExactProductSequence` is intentionally disabled |
| active baseline auditor self-tests | 71/71 pass |
| contract/reduction/science-ledger/profile validator tests | 49/49 pass |
| `tools/config/run_config_preflight.py --require-all` | 127 tests and all mode/kit gates pass; reported 100% coverage |
| `validation/product_contracts.json` JSON parse | pass |
| validation-ledger validator | pass; 60 records |
| science-change-ledger validator | pass; 3 changes and 5 integration commits |
| disabled-with-members negative probe | **unexpectedly accepted**, supports P1-003 |
| requested-enabled-zero/mapmaking-disabled negative probe | **unexpectedly accepted**, supports P1-007/authority ambiguity |

The green registered tests do not traverse a real processor session after the
new begin call, a direct pipeline entry without it, enabled FRUIT with the new
counter/member lifecycle, split Beammap with partial flag selection, a
production invalid filter response/support, intermediate symlink paths,
newline-bearing names, or duplicate canonical product identities. Test success
therefore does not outweigh the static lifecycle contradictions and direct
negative probes.

## Limitations

- No local astronomical data reduction or Unity build/reduction was permitted
  or performed. Nothing here establishes observational or production validity.
- The fresh-session and FRUIT findings were derived from exact control-flow and
  state ownership rather than an astronomical run. The precondition and
  counter contradictions are deterministic and occur before scientific
  validation would be relevant.
- TOCTOU exposure cannot be eliminated by post-hoc user-space validation alone;
  the audit records the remaining windows but does not prescribe a general
  publication framework.
- F005 remains bounded by external FLT authority. Auxiliary measured channels
  were not treated as primary-x evidence.
- No new outgoing handoff is proposed. The existing FLT and FRUIT proposals
  already preserve the relevant ownership boundaries; duplicating them would
  create competing authority.

## Recommendation

Record the exact target as **`amend` / not integration-ready** and keep
production status `existing_use_only`. Preserve F005 as open/conditioned with
parity status exactly `scope_blocked_not_applicable_pending_FLT`, and preserve
F006 as open/held-external and SCI-FRUIT-001-owned.
Return the application work to a bounded successor repair based only on exact
target `d1d19145df574571a894772fdc9410c86cba1041`, followed by a fresh
independent re-audit. At minimum the successor must correct publication
initialization and iteration ownership, reconcile member identities with
realized semantics, make joins/path identity exact, pass actual filtered
response/support, account for split Beammap selection, resolve the literal
enabled-zero authority, and enforce the one-physical-plane policy.

Stop here for owner/coordinator review. Do not integrate, push, contact Unity,
launch another repair from this audit task, or mark production/astronomical
validation complete.
