# SCI-MAP-001 second-cycle independent repair re-audit — 2026-08-05

## Audit identity and boundary

This is the fresh second-cycle, contract-first re-audit of the bounded
SCI-MAP-001 repair. It is an audit and disposition record, not a repair.

- Required repair branch: `codex/repair-sci-map-001`.
- Verified repair-branch tip and starting HEAD:
  `f84b9fd7d7364f9d35317fc6c15b55d2a30e89f7`.
- Verified starting worktree: clean, including untracked files.
- Candidate parent: `02b9eb303037eb3f3a7bb90838b478bb5262e346`.
- Audit-only branch: `codex/second-cycle-reaudit-sci-map-001`, created only
  after the exact clean-entry check.
- Historical independent re-audit authority:
  `851035e67f63bdb2bacc122b17566877a9e6db97`.
- Project-owner amendment authority:
  `6409a36d324072c9b29145c620d01a0686275870`.
- Amendment artifact SHA-256, both in the amendment commit and at the repair
  candidate:
  `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`.

No application code, test, configuration, original audit/amendment artifact,
external corpus, canonical ledger, or coordination snapshot was changed. No
Unity access, campaign, repair, task launch, delegation, integration, or push
was performed. The seven-case corpus at
`/Users/gwilson/work_toltec/local_data/2026-ENG-citlali-MAP` was not copied,
rerun, or inspected; its accepted bounded disposition comes from the immutable
owner amendment and historical re-audit record.

## Contract fixed before implementation inspection

The following amended acceptance surface was recorded before inspecting
`f84b9fd7`:

- **F004:** when observation realization products and coaddition are enabled,
  required observation-owned realizations persist in addition to coadd
  realizations, using the same admitted operator, identity, component/shape,
  support/validity, provenance, and exact cardinality; required output fails
  closed and propagates.
- **F005:** finite floating and signed-`int64` aggregate overflow, and finite
  projected coordinates outside the representable integer/index range, reject
  before any live map, product, realization, diagnostic, or coadd mutation.
- **F007:** binary64 typed/sidecar WCS remains the lossless authority; physical
  FITS WCS is within `0.1 arcsec`, with exact sign, handedness, orientation,
  shape, and centered integer observation/coadd reference-pixel relations.
  Binary64 sidecars remain the exact threshold authority; FITS cards are
  finite, correctly identified and unit-bearing, policy/alias exact, and agree
  with the sidecar at `rtol=1e-12`.
- **F010:** the eight facts, two threshold rules, aliases, availability rules,
  raw-validity carriage, and exact sidecar authority remain unchanged; the
  amended FITS-card rule and aggregate safety must hold without changing
  threshold or coadd arithmetic.
- **F011:** exact-candidate local truth, production FITS, TSan concurrency,
  realization gamma, complete coadd/WCS atomicity, aggregate/index failures,
  output/provenance, complete CTest, baseline, config, and package gates must
  cover the repaired surface without a required-data skip or unexpected
  serious record.

F012 is owner-accepted only for exact-`ed28dafb` external execution,
successful completion, returned product/inventory, visible observation/coadd,
and sequential/OpenMP claims. Missing raw/sample ledgers, per-scan
pre-normalization traces, wrapper/Slurm/environment/retrieval records, and
same-case S-X observation-realization files remain limitations, not rerun
requests. F013 remains conditioned on SCI-ALIGN-001, SCI-CAL-001,
SCI-AST-001, SCI-PTC-001, and SCI-VAL-001; production remains
`existing_use_only`.

## Independent implementation and test assessment

### F005 aggregate and index safety conforms

The ordinary Stokes-I primitive now forms checked sparse deltas before taking
the live-map mutex. Duplicate finite contributions use the established
left-to-right `lhs + rhs` operation with a non-finite-result rejection. Under
the mutex, every live floating and signed count target is preflighted before
the first commit; the full realization tensor is also checked before any
bundle member mutates. Finite projected coordinates are range-checked before
`llround` and integer conversion.

The focused tests exercise staged and live floating overflow, realization
overflow, signed count overflow, and finite out-of-index projection, and hash
or compare the whole bundle before and after rejection. The generic helpers
are used for signal, coefficient, kernel, coverage, both exposure planes, both
count planes, and realizations. F005 is therefore conformant in its amended
scope.

### F007 persistence conforms to the amended authority

The production `Engine::write_maps` path is exercised with non-binary32-exact
typed WCS and thresholds for both observation and coadd products. The test
computes all-pixel typed-to-FITS sky separation and passes `<=0.1 arcsec`; it
also checks RA/Dec axis types and units, exact signs, zero supported
orientation, exact shapes, and the integer coadd CRPIX offsets in typed and
physical FITS state.

The exact sidecar decimal/hex threshold round trip is independently retained.
Every required FITS threshold card is finite, has `BUNIT=1`, has the correct
`ESTTYPE`, agrees with its sidecar authority at `rtol=1e-12`, and the policy
and alias values are exactly equal with the correct `ALIASOF` identity. No WCS,
threshold-selection, normalization, or coadd algorithm change was needed.

### F004 required persistence is repaired, but its realized cardinality is not

The repair correctly creates and writes raw observation realization files
when coaddition is enabled. Production-path tests verify observation and
coadd files, component/shape/unit identity, common response and companion
identity, and required-noise-inventory rejection before the first primary HDU
or legacy-WCS mutation. The ordinary primitive and coadd path retain the same
admitted realization operator and support boundary.

However, the realized noise provenance is not stage-aware when coaddition and
map filtering are both enabled:

1. `should_create_observation_per_obs_outputs()` is false with coaddition, so
   `create_obs_map_files()` creates raw observation noise files but no filtered
   observation noise files.
2. Observation output writes raw observation products and then accumulates the
   coadd; it does not run the filtered-observation output stage.
3. Coadd output writes both raw and filtered coadd realizations.
4. `record_noise_run_completed()` nevertheless multiplies the sum of
   observation and coadd realization counts by the global two-stage filtering
   count. It applies the same overcount to empirical product-map provenance.

For one three-map observation, one three-map coadd, and two realizations, the
actual routed realization writes are `6 raw observation + 6 raw coadd + 6
filtered coadd = 18`; provenance records `(6 + 6) * 2 = 24`. Empirical product
maps analogously route `3 + 3 + 3 = 9` while provenance records `12`.

The new coadd cardinality test calls `record_noise_run_completed(..., false)`;
the existing filtering-enabled test has no coadd. Thus no test covers the
failing combination. This is a direct exact-cardinality/provenance
nonconformance under F004 and the smallest remaining F011 coverage gap.

### F010 conforms after the F005 and F007 re-audit

The eight distinct F010 facts, normalization and science-policy masks,
bitwise aliases, exact binary64 sidecar thresholds, availability/absence
rules, required-companion conjunction, and raw-validity carriage remain in the
accepted implementation and pass the focused contract, FITS, and provenance
tests. Aggregate safety now conditions all accepted finite inputs correctly,
and the amended threshold-card boundary passes. No normal finite-domain
threshold or coadd arithmetic changed. F010 may close after this re-audit.

### F011 remains open

Every executed local gate passes on the states it covers, including the newly
registered production FITS, aggregate/index, concurrent realization, gamma,
and full legacy-WCS digest assertions. F011 nevertheless remains open because
the required coadd-plus-filter observation/coadd realization provenance state
is both untested and nonconformant as described above. A green aggregate suite
cannot substitute for the missing exact state.

## No-broadening and bounded performance assessment

The application change is confined to the ordinary non-polarized Stokes-I
primitive, FITS/output preflight, observation-noise file creation, and noise
cardinality bookkeeping. JINC, polarization, normalization, coadd arithmetic,
threshold selection, and established mature algorithms are unchanged.
`all_valid_coadd_preserves_historical_arithmetic_order_bitwise` and
`ordinary_profiles_match_reference_and_requested_parallel_exactly` pass.

A small before/after measurement reused the existing ordinary-profile fixture
without introducing a benchmark or instrumentation framework. The retained
pre-`f84` binary has the 29-test source surface that is byte-identical between
`ed28dafb`, `02b9eb3`, and the `f84` parent; the current binary has the repaired
31-test surface. After warm-up, three alternating pairs of 500 repetitions of
the unchanged ordinary-profile test produced:

| Binary | Wall times (s) | Median (s) | User CPU (s each) |
| --- | --- | ---: | ---: |
| pre-`f84` | `0.35, 0.35, 0.35` | 0.35 | 0.13 |
| `f84b9fd7` | `0.36, 0.35, 0.36` | 0.36 | 0.13 |

The approximately 3% wall-clock difference on this tiny fixture is not a
dramatic regression and is below the resolution needed for a stronger
performance claim. Static inspection shows a mixed lock effect: sparse
`setFromTriplets` aggregation moved outside the mutex, while live sparse
preflight and one additional full noise-tensor read pass occur inside it. The
complexity class is unchanged. Production-scale contention was not measured
because no existing fixture exposes lock-only timing; that remains a bounded
performance limitation, not a blocker or an invitation to open-ended
optimization.

## Gates

All application-bearing local gates used the supported macOS bootstrap in a
fresh Release build directory and the exact CLI identity
`v4.0.0-3631-gf84b9fd7d`.

| Gate | Result |
| --- | --- |
| clean entry and exact repair SHA/branch tip | pass |
| amendment current/commit/registered SHA-256 equality | pass |
| `citlali_cli` Release build | pass |
| focused science-map truth | 31/31 pass |
| focused TSan | 9/9 pass, no sanitizer report |
| production FITS products | 22/22 pass |
| complete CTest | 593 registered; 592/592 enabled pass; zero failures; one pre-existing unrelated disabled lifecycle test |
| baseline-tool unit tests | 147/147 pass |
| full config preflight | 127/127 unit tests; four mode kits; eight compatibility cases; 100% declared compact coverage; all boundary audits pass |
| validation ledger and profile registry | pass; 60 records, four active and eight preparing profiles |
| product/provenance surface | 12/12 focused provenance tests plus production FITS and complete CTest pass |
| repair diff and worktree checks | pass |

The first aggregate `check` invocation exposed only its existing excluded
target orchestration: the safety placeholder was not built. Building the
explicit safety, focused truth, TSan, production FITS, and CLI targets and then
running complete CTest produced the counts above; there was no application
test failure or required-data skip.

The amended package has 21/21 checksum members intact. As designed for F012,
the campaign self-check rejects the `f84` product registry against its exact
`ed28dafb` source pin. Running the same verifier against an ephemeral read-only
export of the pinned `ed28dafb` source passes all checks and the campaign-driver
self-check. The negative current-source result is therefore a successful pin
check, not evidence that the old campaign ran at `f84`; no external product was
read or rerun.

No unexpected error- or critical-level application record occurred in an
accepted gate.

## Finding dispositions

These are proposals for coordinator review. “Closed” means only the finding's
registered MAP scope; it authorizes no downstream consumer or dependency.

| Finding | Second-cycle disposition | Basis |
| --- | --- | --- |
| F001 | **closed; not reopened** | Ordinary sequential/requested-parallel and concurrent realization behavior remains under the same checked mutex path; TSan is 9/9. |
| F002 | **closed; not reopened** | Explicit invalidity, finite-positive contribution support, distinct masks, and required-companion validity behavior are unchanged. |
| F003 | **closed; not reopened** | Full binary64 identity and two-phase centered admission remain atomic; legacy WCS is now included in the rejection digest. |
| F004 | **remain open** | Required raw observation realization persistence is repaired, but coadd-plus-filter realized product/write cardinality and provenance are overstated. |
| F005 | **close after re-audit** | Floating and signed-count staged/live overflow and finite out-of-index projection reject before any bundle mutation. |
| F006 | **closed; not reopened** | Nonprecision coefficient semantics and the absence of an unauthorized covariance/significance claim are unchanged. |
| F007 | **close after re-audit** | Production typed/sidecar-to-FITS WCS, exact orientation/centering, and amended threshold-card relations pass. |
| F008 | **closed; not reopened** | Lossless one-way realized identity, threshold, product, membership, and raw-parent provenance remain intact and tamper-rejected. The F004 count defect is assigned to the newly repaired realization-persistence surface, not used to reopen the earlier sidecar-authority finding. |
| F009 | **closed; not reopened** | Centered integer embedding, `L=I`, strict bundle admission, and normalized nonprecision coadd arithmetic are unchanged. |
| F010 | **close after re-audit** | Eight facts, masks, aliases, sidecar authority, aggregate safety, and amended FITS cards conform without arithmetic broadening. |
| F011 | **remain open** | The coadd-plus-filter cardinality/provenance state is missing from tests and fails by source-path accounting despite every executed gate being green. |
| F012 | **close in bounded owner-accepted scope** | Accept only exact-`ed28dafb` execution/product/inventory/visible-coadd/SEQ-OMP claims; every absent raw/trace/operational/S-X lane remains an explicit limitation. |
| F013 | **remain open** | ALIGN, CAL, AST, PTC, and VAL dependencies remain open; production remains `existing_use_only`. |

## Decision and minimum remaining work

No scientific or operational owner choice is required. The second-cycle MAP
repair is **not complete**. The smallest remaining application/test defect is
to make noise completion provenance use the already-established separate
observation and coadd output-stage counts, then cover the coadd-plus-filter
case with exact empirical-product and realization-write cardinality. No
algorithm, WCS rule, campaign, or new framework is needed to describe that
defect.

Package-level proposal:

- contract status: `approved`;
- implementation status: `nonconformant`;
- validation status: `in_progress`;
- production status: `existing_use_only`;
- verdict: `amend`;
- re-audit status: `complete` for exact candidate `f84b9fd7`;
- F012 evidence: `accepted_bounded_with_limitations`;
- integration and production expansion: not authorized.

The companion decision and machine-readable ledger update are proposals only.
Only the coordinator may integrate them into canonical state.
