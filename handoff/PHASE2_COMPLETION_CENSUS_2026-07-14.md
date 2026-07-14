# Phase 2 Completion Census — 2026-07-14

## Purpose And Scope

This is a read-only completion census for Phase 2, config authority and
provenance. It records the conclusions of the 2026-07-14 review of:

- `doc/REFACTOR_STATUS.md`;
- `handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md`;
- `doc/STRUCTURAL_REFACTOR_PLAN_2026-06-29.md`;
- `doc/CONFIG_AUTHORITY_AND_PROVENANCE_INVENTORY_2026-07-10.md` and
  `tools/config/config_authority_inventory.json`;
- the raw, processed, mapmaking, coadd, noise-products, post-processing, and
  Beammap authority/transition documents; and
- `handoff/HANDOFF_2026-07-14.md` and the accepted-run ledger.

This report does not change the roadmap or authorize implementation. The living
status document continues to govern sequencing. The machine-readable authority
inventory defines the 13 domains counted here.

## Executive Conclusion

Phase 2 is not complete.

The current domain census is:

| Classification | Count |
| --- | ---: |
| Complete | 8 |
| Awaiting validation | 1 |
| Implementation-ready | 2 |
| Scientifically blocked | 2 |
| Intentionally deferred as a whole domain | 0 |

The eight complete domains have credible domain-level closure evidence. That
does not by itself satisfy the global Phase 2 exit gate. Section F.1 of the
adopted external review also requires atomic observation configuration,
leaf-level state classification, complete source provenance, two authoritative
validation gates, and current mode-appropriate evidence.

No whole inventoried domain can simply be omitted while still claiming the
current F.1 definition of done. Some implementations and deeper cleanups can be
deferred, as listed below.

## Domain Census

### Complete

#### `runtime`

Requested, effective, and realized state are separate. Execution consumes the
effective plan. Required atomic `runtime_provenance.yaml` is versioned and Unity
point `redu27` accepted the sidecar and unchanged products.

#### `timestream-core-output`

RTC/PTC output shape, chunking, selection, scan-to-row realization, and output
registration use typed authority. Legacy output/chunking mirrors were removed.
Per-observation `timestream_output_provenance.yaml` and unchanged complete point
products were accepted at `redu28`.

#### `raw-timestream`

All 169 raw paths have typed readers, serializers, and a one-way typed-to-RTC
execution adapter. The legacy parser, ten reverse mirrors, and parity oracle
were retired. Point `redu42`, Beammap `redu18`, science `redu33`, and parser-
retirement point `redu43` close the relevant gates. The two adjacent
polarimetry paths remain a separate compatibility boundary. Explicitly
unavailable flagged-sample and dynamic-notch counters do not invalidate this
domain because they are labeled unavailable rather than inferred.

#### `processed-timestream`

All 171 frozen paths have direct typed readers and serializers. Requested,
effective, and realized processed provenance is accepted for point, Beammap,
and science. Production legacy parsing and PTC-to-typed mirrors were removed.
`PTCProc` remains only a one-way numerical execution target.

#### `mapmaking`

All 22 frozen leaves enter through the typed boundary. Requested and effective
grouping are separate; JINC, maximum-likelihood, map buffers, and WCS targets
are populated one way. Version-2 provenance records observation identity,
map/write cardinality, coadd state, and completion. Point, Beammap, and science
gates at `e8e42945` are accepted.

#### `noise-products`

The six-path request/effective/realized plan, deterministic RNG identity, map
cardinality, empirical products, realization writes, and required atomic
provenance are complete. Disabled point `redu47`, bounded full-output point
`redu49`, and science `redu15` close the required cases.

#### `coadd`

`coadd.enabled` has one typed owner. Effective activation depends on mapmaking
without mutating the request. Coadd provenance is cross-checked with mapmaking.
Disabled point `redu46` and enabled science `redu11` close the domain.

#### `pointing`

The five-key request, effective fit policy, one-way PTC source-center adapter,
per-observation fit cardinality, and atomic provenance are complete. The policy
defect found by `redu50` was corrected; exact point `redu51` closes the gate.

### Awaiting Validation

#### `post-processing`

All 35 supported leaves under `post_processing.*` and `wiener_filter.*` now
have typed requested/effective authority. Map filtering, source finding, and
source fitting consume that authority through one-way numerical adapters. The
remaining activation shadow and duplicate histogram reader are retired. Point
`redu58` accepts the cleanup with exact products.

The domain remains incomplete only because two runtime gates are outstanding:

1. science coadd-filter/source routing; and
2. Beammap iterative detector-fit cardinality.

The current comprehensive science and Beammap runs may be accepted if their
artifacts satisfy the recorded gates. They need not be restarted solely to use
the later reduced-cost repository overlays.

### Implementation-Ready

#### `beammap`

The 74-leaf `beammap.*` surface is frozen. There are no known typed-request
gaps, configuration literals are confined to the declared boundary, and the
static audit protects the characterized state. The remaining bounded work is
clear:

- create a separate effective Beammap plan;
- record phase, prior, split-flag, and mode/iteration normalization without
  changing the request;
- replace the shared fitting-radius synchronization with a typed effective
  input at the fitting boundary;
- record attempted/completed iterations, detector-fit cardinality, required
  output cardinality, and completion without duplicating post-processing fit
  state; and
- publish and audit stable requested/effective/realized Beammap provenance.

Gaussian fitting, priors, quality flagging, detector maps, and other numerical
algorithms are outside this authority migration.

#### `kids-external`

KIDs is correctly treated as an external schema boundary rather than a target
for a Citlali-owned reimplementation. The minimal unfinished Phase 2 work is to
declare the external schema/version and persist the requested and effective
KIDs configuration identity, including dependency/tool identity. Deep KIDs
typing inside Citlali can be deferred.

### Scientifically Blocked

#### `polarimetry`

The domain has partial provenance and mixed cross-domain policy. In particular,
`calibration.ignore_hwpr` and typed polarimetry policy do not yet express one
approved meaning. There is no enabled polarimetry reference gate. Disabled,
ordinary point/Beammap/science evidence does not validate enabled HWPR or
polarized execution.

The shortest closure path is not to invent the science contract. Either:

- approve the policy and validate an enabled reference dataset; or
- mechanically reject enabled polarimetry at capability/preflight and record
  it as intentionally unsupported until a later project.

Documentation-only deferral is insufficient. The narrow compatibility behavior
that initializes Stokes I for unpolarized runs may remain if it cannot enable
polarized execution or populate raw typed state backward.

#### `astrometry-photometry`

The handoff describes this mixed domain as deferred, but the adopted F.1 exit
criteria explicitly require scientifically complete atomic observation config.
The domain therefore remains a Phase 2 closure blocker, not a safe whole-domain
deferral.

Open issues include:

- coordinate frame, epoch, range/wrapping, sign, and extrapolation rules;
- finite astrometry vector elements and MJD ordering/coverage;
- separation of requested catalog/source settings from realized fits;
- calibration/source provenance; and
- the policy for a missing Beammap per-array source flux.

The current photometry loader resets the typed source object but populates the
persistent legacy source-flux map without first establishing replacement
semantics. A later observation must never inherit incidental flux state from an
earlier one. The current loader also retains a direct `std::exit` failure path;
removal of all library exits is Phase 3 work, but this observation-config
failure must at least join the normal propagated configuration diagnostic
boundary for the supported Beammap path.

## Roadmap And Ledger Inconsistencies To Resolve

These are interpretation or bookkeeping inconsistencies, not a recommendation
to reopen completed domain implementations.

1. **Domain completion versus global Phase 2 completion.** The inventory marks
   eight domains complete using domain-specific gates. F.1 additionally
   requires a complete leaf census, atomic observation config, ordered source
   provenance, and two validation gates. A completed domain must not be read as
   proof that Phase 2 as a whole is complete.

2. **Subsystem inventory versus leaf-level definition of done.** The authority
   inventory explicitly began as a subsystem census. F.1 requires every
   executable low-level leaf to have an owner, unit, allowed domain, mode
   applicability, and requested/effective/observation-resolved/realized
   classification. The existing frozen manifests provide much of the path
   coverage, but there is not yet one complete machine-readable leaf contract.

3. **Compact overlay wording.** `REFACTOR_STATUS.md` says Phase 2 requires
   reviewed overlay fixtures for each supported mode, while external-review
   F.1 says compact-config deployment is not required and the full hermetic
   numbered-overlay semantics gate is mandatory only before rollout. The
   shortest consistent interpretation is:

   - retain and review matched low-level mode overlays and their exact bytes,
     roles, ordering, and hashes for Phase 2 evidence; and
   - explicitly defer compact-config production rollout and its full hermetic
     expert/list/null/alias/unknown-key/multiple-step suite as an I8 rollout
     blocker.

   If the owner intends the living status to require the full compact suite for
   Phase 2 anyway, that must be stated explicitly.

4. **Deferred mixed domains versus F.1.** The recent handoff preserves
   polarimetry and astrometry/photometry as unresolved mixed domains. That is
   accurate as a current inventory state. It is not sufficient for Phase 2
   closure: polarimetry needs a mechanical capability disposition, and
   astrometry/photometry needs atomic supported-mode behavior.

5. **Typed label versus unresolved polarimetry authority.** The machine
   inventory records typed execution authority for polarimetry while also
   labeling the migration `mixed-adapter`, provenance partial, and the
   calibration-owned policy unresolved. The mixed/partial classification is
   the operative completion status until the cross-domain HWPR decision is
   resolved.

6. **Per-domain sidecars versus source provenance.** Domain sidecars preserve
   typed requested/effective/realized state well, but accepted-run records note
   that original ordered TolTECA overlay sources were often not retained.
   F.1 requires collision-safe source bytes plus path, role, precedence, hashes,
   canonical merged YAML, calibration sources, and schema/tool versions. This
   remains a global provenance gap even where domain sidecars are complete.

## Recommended Shortest Defensible Sequence

The sequence should preserve the roadmap rule that operational authority moves
one domain at a time.

1. **Close post-processing validation.** Accept the already-started science and
   Beammap runs only if they show correct coadd filtering and iterative fit
   cardinality, valid sidecars, exact matched config, no missing/extra/skipped
   required records, and zero unallowlisted errors. Mark the domain complete in
   the inventory, status, handoff, and ledger only after both pass.

2. **Obtain the minimum owner decisions before implementation.** Decide HWPR
   semantics/support, astrometry coordinate/time rules, and missing Beammap
   flux behavior. Do not broaden these discussions into algorithm redesign.

3. **Complete the bounded Beammap authority domain.** Implement its effective
   plan, direct effective fitting input, realized cardinality, required atomic
   provenance, semantic audit, and focused reset/parity tests. Close it with a
   matched Beammap Unity gate before moving to the next operational domain.

4. **Complete atomic astrometry/photometry observation configuration.** Build
   and validate one complete observation value before mutating processors;
   replace rather than merge source-flux state; record calibration/source
   identity; and prove that a second observation cannot inherit the first.
   OOF should exercise multi-observation astrometry/time state and Beammap
   should exercise source identity/flux behavior.

5. **Close the minimal KIDs external boundary and global source manifest.** Do
   not reimplement KIDs. Persist its requested/effective external identity and
   add the durable ordered config-source/provenance envelope required by F.1.

6. **Disposition polarimetry.** The recommended shortest path is a fatal
   enabled-polarimetry capability/preflight rejection until an approved
   contract and reference dataset exist. Record the domain as intentionally
   unsupported/deferred only after the rejection is mechanically enforced.

7. **Run the final Phase 2 snapshot matrix.** Establish current matched point,
   OOF, Beammap, and science baselines on the final Phase 2 tree so Phase 3
   begins from current pre-change characterization. Require strict product and
   provenance auditing under the criteria below.

## Exact Phase 2 Completion Criteria

Phase 2 is complete only when every item below is true.

### Authority And State

1. Every supported executable low-level leaf has one machine-readable owner,
   unit, allowed domain, mode applicability, and state classification.
2. Accepted requested config is immutable.
3. Context-free normalization produces a separate effective plan.
4. Observation/calibration input produces a separate atomic observation plan.
5. Realized execution and product state does not overwrite requested or
   effective state.
6. Compatibility aliases exist only at loading boundaries.
7. No bidirectional typed/legacy synchronization exists.
8. One-way numerical adapters may remain, but processors may not repopulate or
   override typed policy.

### Validation

9. The startup gate rejects parser failures, unknown enums, non-finite required
   scalars and container elements, invalid ranges/domains, missing required
   keys, inconsistent duplicate facts, and unknown/unconsumed Citlali-owned
   keys.
10. An observation-scoped gate runs after calibration/observation resolution
    and before scientific execution.
11. Astrometry, photometry, source flux, calibration-dependent choices,
    coordinate/frame/range rules, finite vector elements, and MJD
    ordering/coverage are validated as one observation value.
12. Repeated-observation tests prove replacement/reset and no stale inheritance.
13. Phase-specific adapter parity tests cover context-free and observation-
    resolved state; later learned/diagnostic values are treated as realized
    metadata, not parity failures.

### YAML And External Boundaries

14. RTC, PTC, mapmakers, and mode code contain no processor-owned YAML parsing
    for Citlali-owned executable leaves.
15. The KIDs YAML reader is explicitly declared as an external subsystem
    boundary with a schema/version and requested/effective identity.
16. Polarimetry is either enabled and validated or mechanically rejected.

### Provenance

17. Supported domains have stable, semantically audited provenance with no
    unexplained `partial` or `missing` state.
18. A durable run manifest preserves collision-safe exact source bytes,
    ordered source path/role/precedence/hash records, canonical merged low-level
    YAML, canonical typed request, effective plans, realized decisions,
    calibration sources, KIDs external identity, and schema/tool versions.
19. Required provenance publication is atomic and failure propagates to the
    CLI.
20. Unavailable realized fields are labeled unavailable and are never guessed.

### Runtime Evidence

21. Post-processing science and Beammap gates pass.
22. Beammap authority/provenance passes a matched Beammap gate covering all
    detector identities, flags, fit/output cardinality, and required products.
23. Atomic astrometry/photometry passes multi-observation OOF and Beammap
    evidence appropriate to the behavior touched.
24. The final Phase 2 tree has current matched point, OOF, Beammap, and science
    characterization suitable for the next shared structural phase.
25. Accepted runs contain zero unallowlisted error-level records.
26. Strict comparison reports zero missing, extra, or skipped required records;
    point/timestream changes include complete TOD and metadata; numerical
    differences remain inside the accepted per-mode profiles.
27. Required domain and global provenance passes semantic and cross-sidecar
    consistency audits.

### Governance

28. `doc/REFACTOR_STATUS.md`, the machine authority inventory, domain
    documents, dated handoff, and accepted-run ledger agree on every domain's
    state and final candidate evidence.
29. Compact-config rollout is either explicitly deferred with I8 recorded as a
    rollout blocker, or its full TolTECA overlay acceptance suite passes.
30. No further analysis-control migration or Phase 3 shared-boundary work
    begins before these gates close.

## Deferrals That Do Not Weaken The Architecture

The following may be deferred beyond Phase 2 when recorded with their retained
boundary and re-entry condition:

- compact-config production rollout and TolTECA catalog/template replacement;
- the full hermetic compact numbered-overlay semantics suite, if compact
  rollout is explicitly deferred, while retaining reviewed matched low-level
  overlays and actual-run source provenance;
- enabled polarimetry implementation after fatal capability rejection is in
  place;
- R/quadrature-channel execution until its measured-channel contract exists;
- broad RTC/PTC, cleaner, JINC, Wiener, Beammap-fit, prior, or flagging
  algorithm redesign;
- removal of mature one-way numerical adapters solely for stylistic purity;
- deep KIDs schema reimplementation inside Citlali after the external identity
  boundary is complete;
- raw flagged-sample and dynamic-notch realized counters while they remain
  explicitly unavailable;
- concurrent reduction support;
- ABI/plugin or dependency-injection machinery;
- Phase 3 header/compiled-boundary and broad library-exit work;
- Phase 4 reproducible CI, controlled performance/RSS certification, and
  same-candidate release validation; and
- install/export work unless external library consumption is accepted as a
  project goal.

The following cannot be deferred without weakening or contradicting the
current Phase 2 architecture:

- post-processing's science and Beammap gates;
- Beammap's effective plan and provenance;
- atomic astrometry/photometry and stale-flux prevention;
- the minimal KIDs schema/version and configuration-identity boundary;
- the global ordered config-source provenance record;
- a mechanical polarimetry support/rejection decision; or
- final current mode characterization before Phase 3 shared structural work.

## Questions For The Project Owner

Only the questions needed to unblock the shortest path should be answered now.

1. **Polarimetry capability:** Is enabled polarimetry a supported Phase 2/next-
   phase capability? If not, may the implementation add a fatal capability
   rejection now?
2. **HWPR policy:** What does `ignore_hwpr: false` mean: require HWPR data, use
   it when available, or merely do not force-ignore it? What should happen when
   required data are absent?
3. **Astrometry contract:** What are the canonical coordinate frame, epoch,
   range/wrap, tangent-plane sign, extrapolation, and MJD ordering/coverage
   rules?
4. **Beammap source flux:** Is a missing per-array flux fatal, may it come from
   a named catalog, or may the observation proceed explicitly uncalibrated?
   Incidental inheritance from a previous observation is excluded in every
   case.
5. **KIDs support surface:** Which KIDs TOD types are operationally supported in
   each mode, and what units/calibration validity must the external boundary
   record? Unsupported types must fail rather than silently fall back.
6. **Overlay gate interpretation:** For Phase 2, does "reviewed overlay fixtures"
   mean retained matched low-level mode overlays while compact rollout remains
   deferred, or does the owner intend to require the full hermetic compact
   numbered-overlay semantics suite before Phase 2 closes?
7. **Source provenance coordination:** Can TolTECA provide Citlali/the durable
   run manifest the exact ordered overlay paths, roles, precedence, bytes, and
   hashes? If not, which component is assigned ownership of that F.1 record?

## Handoff To The Next Task

The next task should first inspect whether the in-progress post-processing
science and Beammap artifacts have been synchronized and accepted. If both
pass, update the post-processing domain and ledger coherently, then begin only
the bounded Beammap authority plan described above. If either fails, diagnose
that mode gate before starting another authority migration.

Do not treat this census as permission to modify scientific algorithms, expand
polarimetry, wire compact config, start Phase 3, or broaden `Engine` state.
