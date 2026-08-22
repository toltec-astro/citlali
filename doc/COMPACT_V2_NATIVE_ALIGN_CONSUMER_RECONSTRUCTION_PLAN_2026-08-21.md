# Compact-v2 Native ALIGN Consumer Reconstruction Plan — 2026-08-21

## Decision

The intended native-cohort consumer behavior in historical commits
`fd3627fc70060a78e65b47b3f798825fd3238514` and
`9d9d55a54fb16cd3964af79522d0d37de253dce2` is salvageable, but neither
commit is an integration unit. The implementation must be reconstructed in
bounded stages on the current compact-v2 and JINC convergence line. No stage
may copy the historical canonical APT v1 authority or silently fall back to a
legacy rectangular/common-time identity.

This document is a design and validation plan only. It adds no application
consumer code, changes no mode routing, and makes no ALIGN, APT, JINC, or
production-readiness claim. Implementation is blocked until an independent
review accepts the contracts and stage boundaries below against an exact plan
commit.

The first independent review of exact plan commit `82b086856f891873167760534b64a0811840f3cb`
returned `revise` on 2026-08-22. This revision resolves its four blocking
findings by distinguishing both existing Beammap calibration lanes, naming
the baseline-governed consumer-selection `flag`, freezing packet-counter
continuity, and assigning observation, scan/chunk, and output transaction
owners. It also adopts the review's nonblocking recommendation to freeze gap
slot-association tolerance and presence-mask parity. A new exact-SHA
independent review remains required.

## Frozen inputs

| Role | Identity | Use |
| --- | --- | --- |
| Reconstruction base | `e0270dffbc1c7d927d1c5b09202da309c0c5bbcd` | Current compact-v2/JINC convergence authority, including accepted JINC ownership hardening |
| Native-cohort foundation | `c87d5693dbcf185b2e76d15b41ac55ff3d71f1ef` | Already integrated foundation represented by current `timestream_native_sample.h`, `timestream_coincidence_cohort.h`, and their tests |
| Historical consumer reference | `fd3627fc70060a78e65b47b3f798825fd3238514`, tree `e45286cc15b3d448fb19dcfffa0d2a90bdb23edb` | Behavioral evidence only; never cherry-pick |
| Beammap lineage correction | `9d9d55a54fb16cd3964af79522d0d37de253dce2`, tree `9e87f2f733adefe6fca6d07ce0791e3ab7e430ed` | Mandatory mode-boundary behavior; never cherry-pick |
| Historical consumer binary patch | SHA-256 `7623f3ebd792b980e4f85e55b4a2a009d6c4e86c57a256e5331544bb1d816450` | Frozen audit identity |
| Historical Beammap correction binary patch | SHA-256 `a1fc0d2b4afe161b2b4f9ab642ddb62de18abdbbcd7f0b164e7a336bc616bb40` | Frozen audit identity |
| Compact APT authority | `citlali::pipeline::canonical_apt_v2`, `canonical_apt_bundle_v2.h`, and ADR 0012 | Only admitted APT product/identity authority for new consumer code |
| Targeted JINC science authority | Unity-tested `e77460cffad49387795009539d6abc7e370e8b58` plus fail-closed child `e0270dffb` | Preserve working-support arithmetic and targeted `redu04` result |

The historical consumer commit changes 48 paths and adds 20,689 lines. It
combines a duplicate APT-v1 implementation with native timing, pointing,
KIDs ingress, RTC/PTC processing, mapmaking, and product provenance. That
coupling is precisely what this reconstruction must remove.

## Governing identities and invariants

### 1. Detector identity comes from the verified compact-v2 bundle

An ordinary Science or Pointing consumer may receive a typed detector
relation only from the same `canonical_apt_v2::VerifiedBundle` transaction
that passed filesystem, receipt, component, target, source, and raw-byte
verification in `Calib::get_apt`.

The adapter must publish an immutable detector-column relation containing, at
minimum:

- the verified matched-bundle `ComponentIdentity`;
- the verified relation-component identity or an exact immutable binding to
  the verified `RelationTable`;
- legacy detector column, defined only as the index in the existing
  presentation-rank-ordered numeric `Calib::apt` view;
- output artifact-local UID;
- target `ScopedRowReference`;
- raw network and channel;
- relation disposition and, when present, selected seed reference; and
- the exact baseline-governed consumer-selection `flag` value and its typed
  missing policy.

The consumer-selection flag is the compact-v2 baseline-governed field named
exactly `flag`. It is never inferred from target KMP `kids_flag`, native sample
flag bits, or Beammap's later `flag2`. A matched relation row requires exact
signed-int64 `flag`. An unmatched or ambiguous row may carry typed missing
only when its verified compact-v2 field rule authorizes that state; the typed
relation retains it as absent and never fills or interprets it. Authorized
typed missing is not a nonfinite-value failure.

`presentation_rank`, `application_rank`, `source_rank`, UID, network, and
channel remain distinct artifact-local facts. None becomes a persistent
detector identity. The adapter must join `VerifiedBundle::apt.rows` to
`VerifiedBundle::relation->rows` by `output_uid`; it must not reconstruct a
typed relation from the lossy numeric `Calib::apt` map, table position, a
floating-point UID, or file order.

Admission is all-or-nothing. Duplicate/missing output UIDs, rank errors,
network/channel collisions, incomplete raw coverage, identity disagreement,
out-of-range detector columns, or unrepresentable legacy columns reject the
candidate before changing live `Calib` state. `Calib::apt` remains the
one-way legacy value adapter; the typed relation is a separate immutable
authority, not a bidirectionally synchronized copy.

### 2. Delivered native samples retain time authority

For each raw network, native identity is
`(network_id, delivered_native_row, reconstructed_time_unix_sec)`. The
delivered row and reconstructed timestamp are immutable. A common-time slot
is only a relational coincidence coordinate and provenance label; it is
never substituted for a measured sample timestamp.

Packet-counter ingress is fail-closed and exact: a nonfinite, fractional, or
out-of-range value rejects the alignment candidate before publication. For
admitted signed integer counters `before` and `after`, two delivered rows may
share a contiguous run only when `before` is not the maximum representable
counter and `after == before + 1`. Repeated, decreasing, jumping, and
maximum-to-minimum rollover transitions close the run; there is no inferred
wrap policy. A scan boundary also closes support even when adjacent counters
would otherwise be contiguous.

Gap slot association preserves the established compatibility construction:
the common grid uses its realized `dt`, association tolerance is exactly
`dt / 2`, each native timestamp is rounded to its one candidate grid slot,
and admission requires both an injective native-row/slot mapping and exact
presence/absence parity with the established legacy mask. Alternative
within-tolerance row selection is not admitted by this lane.

No operation may synthesize a detector sample for an absent cell, interpolate
a detector value across a gap, reuse a native row in two cohort cells, or
bridge two runs for filtering, downsampling, PCA, variance, pointing, or
mapmaking.

### 3. Missing, invalid, and measured are different states

The existing foundation's `mapped_valid`, `mapped_invalid`, and `absent`
states remain exhaustive. A finite placeholder may exist only inside a PCA
working buffer for an excluded rectangular cell. It is not a sample, is never
scattered, and cannot contribute to a map or product-support count.

Every scatter is transactional against the immutable native identity and
expected revision. A stale revision, duplicate destination, changed
timestamp, nonfinite replacement, or partial result rejects the entire
operation without observable mutation and permits a clean retry.

### 4. Pointing follows the measured detector sample

Telescope interpolation and detector pointing must be evaluated at each
network's exact reconstructed native timestamps. A detector column obtains
its network through the compact-v2 relation. Common-grid telescope values may
remain a compatibility view for legacy-inactive execution, but are not an
authority for a native-required sample.

Source masking, kernel/source models, detector variance windows, naive
projection, and JINC projection must use the same admitted native pointing
pair. An absent cohort cell creates neither detector data nor pointing.

### 5. Mature numerical bodies do not change implicitly

The reconstruction may add typed admission, gather/scatter adapters, and
explicit native dispatch around existing numerical bodies. It must not alter
RTC filters, PTC cleaners, PCA mathematics, naive accumulation, JINC kernel
weights, JINC formal/working support, map grouping, or floating-point
accumulation order unless a separately named, measured, reviewed scientific
change is opened.

On complete identical-time fixtures, native dispatch must be exactly equal to
the established rectangular result. Changes to mapmaker files require a
reviewed worker-suffix or equivalent arithmetic checksum demonstrating that
only pointing/sample selection changed.

### 6. Mode lineage is explicit and asymmetric

| Mode | Matched-v2 detector relation | Native timing/pointing | Native consumer/product lineage |
| --- | --- | --- | --- |
| Science | Required when native consumer execution is activated | Required | Candidate activation path |
| Pointing | Required when native consumer execution is activated | Required | Candidate activation path |
| Beammap, detector/automatic grouping | Forbidden as an input authority; this lane builds the APT from raw detector inventory | May be used without consumer lineage; a producer-native carrier requires separate review | Must remain disabled until a Beammap-specific producer lineage exists |
| Beammap, existing non-detector grouping | The established calibration-table load remains unchanged but is never a matched-v2 native-consumer authority | May be used without consumer lineage; a producer-native carrier requires separate review | Must remain disabled until a Beammap-specific producer lineage exists |
| OOF or any other mode | No inferred activation | No inferred activation | Fail closed pending an explicit mode decision |

The historical `9d9d55a54` correction is therefore a first-class acceptance
contract, not a cleanup to apply later. A Beammap must continue to build its
APT from raw detector inventory for detector/automatic grouping. Existing
non-detector grouping may continue through its current calibration-table load;
this plan neither removes nor relabels that legacy lane. Neither Beammap lane
may require, publish, or inherit an observation-matched native-consumer
relation merely because native timing or pointing objects exist.

### 7. Lifecycle and transaction ownership is explicit

The verified compact-v2 relation plus immutable alignment and pointing handles
belong to one observation lifetime. Each handle is published only after its
own complete admission; the native-ready observation binding is published
atomically only after the exact set is present. All are reset/destroyed at the
observation boundary.

The measured detector mapping, mutable native-sample ledger, and monotonic
operation sequence belong to exactly one scan/chunk transaction. They are
created fresh after scan/chunk admission and destroyed on commit, rollback,
or boundary exit; no mutable revision or `last_operation` state may cross a
scan/chunk or observation boundary. The existing science matrix remains the
value owner where applicable.

Required product staging, commit/rollback, and deterministic index replacement
belong to the existing output/publication owner. Numerical processors may
return immutable provenance facts but do not own output publication. None of
these owners is implemented as process-lifetime state or new cross-cutting
public `Engine` state.

## Reconstruction architecture

The current native-cohort foundation remains the bottom layer. New code must
be divided by authority rather than recreated as the historical 2,138-line
bridge header.

| Layer | Sole responsibility | Forbidden responsibility |
| --- | --- | --- |
| Compact-v2 detector relation | Immutable verified bundle-to-detector-column join | Raw I/O, timing, PCA, pointing, mapmaking |
| Native network alignment | Delivered times, packet counters, run boundaries, relational slot associations | Detector values, APT matching, telescope interpolation |
| Native pointing plan | Telescope/offset values sampled at exact native times | Detector cleaning or map accumulation |
| Measured detector scan | One-scan/chunk exact raw channel-to-typed-column gather, ledger, revision sequence, and immutable cell lookup | Filtering, synthesized gap filling, cross-boundary state, publication |
| RTC run adapter | Dispatch one contiguous run at a time and record exact support/ORed flags | Cross-run windows or detector identity decisions |
| PTC cohort adapter | Gather finite rectangular working groups and transactionally scatter results | Treat placeholders as samples or change PCA math |
| Science projection adapter | Supply exact measured cells and matching native pointing to existing mapmakers | Change naive/JINC numerical kernels |
| Product provenance | Bind observation, relation, alignment, pointing, operation, revisions, support, and outputs for the existing output owner | Own scientific values or mutate processing state |

Headers must remain independently compilable. Large application orchestration
belongs in private implementation fragments or small cohesive interfaces; it
must not become new cross-cutting public `Engine` state. The observation owner
retains only observation-scoped immutable handles; the scan/chunk owner retains
the measured mapping, ledger, and operation sequence; the existing output
owner retains publication state. Each clears its state at the boundary named
above.

## Staged implementation and stop gates

Every stage is a coherent reviewable commit. A failed stage is corrected in
place; later stages do not begin to conceal an earlier contract failure.

### Stage 0 — independent plan review

An independent reviewer must inspect this exact plan commit and record:

1. whether compact-v2 bundle, relation, and row identities are sufficient for
   the proposed detector-column adapter;
2. whether the Science/Pointing/Beammap/other mode matrix is correct;
3. whether common-slot nonauthority and run-boundary rules preserve the
   accepted SCI-ALIGN foundation;
4. whether the proposed transaction and lifecycle owners are explicit;
5. whether numerical and product claims are appropriately bounded; and
6. an exact verdict of `accept`, `revise`, or `reject` tied to the plan commit.

Implementation remains blocked on any verdict other than `accept`.

### Stage 1 — compact-v2 typed detector relation

Implement only the immutable v2 relation and an atomic `Calib::get_apt`
publication transaction. Retain `VerifiedBundle::relation` facts without
introducing a second parser or a v1 fallback.

Focused tests must cover matched-bundle admission; target/output joins;
presentation permutations; exact int64 preservation; complete detector-column
and raw network/channel coverage; matched, unmatched, and ambiguous relation
states; exact baseline `flag`; authorized typed missing only for verified
unmatched/ambiguous rows; rejection of `kids_flag`, sample flags, or `flag2` as
substitutes; bundle/row/relation identity binding; and atomic rejection of
tamper, stale scope, duplicates, omissions, wrong channel/network, foreign
rows, unauthorized nulls, and invalid governed values. A Beammap baseline must
not be admitted through this consumer API.

Stop after the relation tests and the complete existing compact-v2 suite.
There is no runtime native-consumer activation in this stage.

### Stage 2 — native alignment and pointing carriers

Reconstruct per-network delivered times, packet counters, discontinuities,
contiguous runs, relational slot associations, and telescope interpolation at
native times. Store immutable observation-owned handles without changing the
current common-time compatibility products. Use exact signed-counter `+1`
continuity without rollover and the established `dt / 2`, single-rounded-slot,
injective, legacy-presence-parity association rule.

Focused tests must demonstrate subcadence-drop preservation without
synthesis; counter repeat, decrease, jump, maximum-counter rollover, and scan
boundary partitioning; no cross-run association; exact `dt / 2` tolerance and
legacy presence-mask parity; collision rejection; input-order and network
permutation invariance; exact equality for identical native times; telescope
and detector-offset evaluation at native time; and atomic rejection of
duplicate, absent, nonfinite, fractional/out-of-range counter, stale, or
cross-scope candidates.

Stop before reading detector values or dispatching RTC/PTC/mapmaking.

### Stage 3 — measured detector ingress and scan admission

Join each raw KIDs matrix channel to exactly one compact-v2 detector column,
gather only complete native run/cohort cells, preserve original flag bits, and
publish an immutable measured scan mapping. Create the mutable ledger and
operation sequence fresh for this scan/chunk; destroy them on commit,
rollback, or boundary exit. The existing science matrix may remain the value
owner; the mapping must not retain an unnecessary second O(rows x detectors)
value copy.

Focused tests must cover complete and partial cohorts, exact row/channel
selection, input network permutation, typed-relation presentation
permutation, original flags, zero/large exact UIDs, noncontiguous network
membership, and atomic rejection before any scan or lifecycle mutation.

Stop with the native-required processing mode still unable to enter RTC.

### Stage 4 — RTC contiguous-run dispatch

Dispatch the established RTC numerical body separately for each admitted
contiguous native run. Downsampling anchors reset per run. Every output row
records its exact native support, selected anchor, detector partition, and
bitwise-OR input flag support. Operations that require a cross-run window fail
closed.

Focused tests must prove exact legacy equivalence on complete identical-time
fixtures, run-local downsample anchoring, ordered and complete support,
ORed flags, no gap bridging, deterministic repeated results, and no mutation
on a rejected run.

Stop before PTC or map projection. RTC product writing remains disabled until
Stage 7 provenance is ready.

### Stage 5 — PTC/PCA cohort gather and transactional scatter

Build detector-level PCA groups from the typed relation and explicit cohort
cells. Preserve the existing grouping algorithms (`nw`, `array`, and other
already supported ordinary groups) only where exact typed membership exists.
Use checked finite placeholders for excluded cells, preserve invalid native
values, and scatter replacements transactionally with monotonically
increasing revisions inside the current scan/chunk owner only.

Focused tests must cover ordinary groups ignoring private placeholders;
exact noncontiguous memberships; pass-through groups; optional modes failing
closed before a cleaner; placeholder-value invariance; stale/duplicate/
nonfinite scatter rejection and retry; and exact rectangular equivalence when
all networks share identical times.

Stop before mapmaking. Unsupported second-pass/windowed operations fail closed
rather than reverting to common-grid science.

### Stage 6 — native science pointing and map projection

Route each admitted detector/sample cell and its exact native pointing pair to
the existing naive and JINC population paths. Missing or invalid cells do not
project. Map indices remain governed by the current map-grouping authority and
the JINC ownership preflight remains in force.

Focused tests must prove source-mask/kernel/variance/map pointing consistency;
naive and JINC projection of measured cells only; equality with the existing
rectangular result at identical times; typed-relation permutation invariance;
and rejection of stale, foreign, incomplete, duplicate, unequal, nonfinite,
or synthetic candidates before map mutation. Existing naive and JINC focused
suites must pass unchanged, and valid-path accumulation arithmetic must have a
reviewed equality/checksum record.

Stop before claiming output lineage or enabling ordinary mode routing.

### Stage 7 — observation/scan/product lineage and explicit activation

Add compact provenance that binds the matched-v2 bundle/relation identity,
raw manifest, alignment plan, native pointing plan, scan operation, revision
transitions, RTC support, PTC grouping, and map/product occurrence. Required
product publication and index replacement remain atomic and deterministic.

Only now may reviewed Science and Pointing routes activate the native-required
consumer. Detector/automatic Beammap must take the compile-time or typed
runtime raw-producer route that cannot request matched-consumer lineage. The
existing non-detector Beammap calibration-table lane remains unchanged and
also cannot request that lineage. Other modes remain inactive.

Focused tests must cover lifecycle begin/commit/rollback/retry; stale,
missing, foreign, and partial lineage rejection before mutation; deterministic
relation/raw-manifest digests; complete prepared and committed snapshots;
required-product failure propagation; deterministic index replacement; and
the explicit detector/automatic Beammap producer-without-input-relation case
plus an unchanged non-detector Beammap calibration-lane case.

## Validation matrix

Each application stage must pass its focused tests plus:

- all existing SCI-ALIGN foundation tests;
- all compact-v2 unit, protocol, relocation, determinism, tamper, guardian,
  and product-contract tests;
- all affected RTC/PTC, pointing, naive, JINC, lifecycle, output-schema, and
  public-header tests;
- a `citlali_cli` build;
- complete CTest with no new skip or disabled test;
- the complete baseline-tool suite;
- validation and intended-science-change ledger validation;
- required four-mode config preflight with zero required-data skips and all
  authority audits; and
- `git diff --check` and an unexpected error-level log audit.

At Stages 4 through 7, run focused equality cases at OpenMP thread counts 1,
2, 4, and 8 where the existing implementation can dispatch in parallel.
Thread-count equality is exact where the current numerical contract is exact;
no new tolerance may be introduced to hide identity or ordering drift.

### Unity routing

No Unity run is required for the plan or Stages 1–3 because they do not
activate a numerical consumer. Before Stage 4 begins, freeze one small
owner-reproducible native-gap fixture locally. Before Stage 7 can be accepted,
the owner must run at least:

1. a Science or Pointing observation with a verified matched-v2 bundle and
   retained native-gap diagnostics;
2. a matched identical-time or no-gap comparison against the legacy-inactive
   path;
3. naive and JINC projections from the same admitted native scan where both
   are configured;
4. a detector/automatic Beammap producer regression proving no matched-v2
   consumer authority is required and comparing against the accepted relevant
   Beammap baseline; and
5. an existing non-detector Beammap calibration-lane regression proving that
   it remains unchanged and does not acquire matched-consumer lineage.

The exact binary, source commit/tree, requested and merged configurations,
input bundle identity, raw input digests, logs, product index, and retained
products must be recorded. Missing required products, unexpected error-level
messages, unexplained detector/flag changes, or an unreviewed numerical delta
fail the gate.

## Explicit exclusions

This lane does not admit:

- canonical APT v1 issuance, admission, scope types, or fallback;
- the historical old-base timing repair ending at `c77105b9b`;
- split-direction Beammap mode or support-cropping diagnostics;
- the separate PTC metadata repairs `7fc59344c`/`5c6309125`;
- changes to RTC/PTC/JINC/naive numerical algorithms or defaults;
- Beammap matched-consumer lineage;
- OOF activation without a separate mode contract;
- R integration, TolTECA build integration, or production authorization; or
- a broad `Engine` singleton/state expansion.

## Completion criterion

The reconstruction is complete only when every admitted stage has its own
exact commit, independent disposition, focused evidence, complete affected
gates, and—where routed above—owner-run Unity evidence. Until then, the
current native-cohort foundation remains a validated contract surface but not
a production consumer, and the historical `fd3627fc7`/`9d9d55a54` commits
remain behavioral evidence only.
