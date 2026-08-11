# SCI-RTC-001 learned-sampling Stage A successor-2 independent re-audit

Date: 2026-08-11

Role: fresh role-separated independent auditor

Audit state: complete, stopped for coordinator/owner review

Application candidate: `cbb2fd767e0676906d1413ae84022270bee1a667`

Disposition: **RETURN FOR REPAIR — Stage A successor-2 is not accepted**

The exact candidate does not close SRA-001 through SRA-009. The audit records
eleven P1 findings and four P2 findings. The failures are technical departures
from already frozen decisions; no new scientific decision or owner
interpretation was needed.

Detailed evidence is in
[RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_EVIDENCE_2026-08-11.md](./RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_EVIDENCE_2026-08-11.md).
The machine-readable finding register is
[RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv](./RTC_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_2_REAUDIT_FINDINGS_2026-08-11.csv).

## 1. Exact audited identity and role separation

- Live remote application branch:
  `origin/codex/repair-rtc-learned-sampling-stage-a-successor-2`.
- Live remote and local application candidate:
  `cbb2fd767e0676906d1413ae84022270bee1a667`.
- Parent: `66c96757164af2c83ee1449d00fea30d131a7e3f`.
- Candidate tree: `4727864c7ca4f078649fcf6473a7225d5d3aa9f8`.
- Binary patch SHA-256, computed from `git diff --binary <parent>
  <candidate>`:
  `d1521fbc0a5afdfcfa61b41c57ba483b1d69969a45115829d2f8d973a51c9c39`.
- The candidate changes 23 application/configuration/test/validation/documentation
  paths and is therefore an application-code candidate, not a documentation-only
  coordination object.
- Auditor worktree:
  `/Users/gwilson/.codex/worktrees/fce6/citlali-refactor`.
- Audit branch:
  `codex/reaudit-rtc-learned-sampling-stage-a-successor-2-20260811`.
- The auditor is independent of repair task
  `019ff09e-e42f-7630-9535-0bc048afb773` and its distinct worktree
  `/Users/gwilson/.codex/worktrees/24fb/citlali-refactor`.
- The audit branch was created only after the coordinator accepted the mandatory
  READY checkpoint. No repair report was treated as proof of closure.

## 2. Frozen authority reconfirmed

- Coordination commit:
  `3fe0aa30eaa0d8848dbb39eb720457326c0b43ba`.
- Frozen repair handoff:
  `doc/audits/packages/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REPAIR_HANDOFF_2026-08-11.md`,
  SHA-256
  `60c3237752ba6195b80223e05f3b41e536a6e13263ed2c5a138c97575a1572e5`.
- Frozen finding ledger:
  `doc/audits/proposals/SCI-RTC-001_LEARNED_SAMPLING_STAGE_A_SUCCESSOR_REPAIR_FINDING_LEDGER_2026-08-11.yaml`,
  SHA-256
  `7721535ef8f6af34a6f587add155f5de6f22ce322c8f981a36aa2a32abbd7a50`.
- Prior independent re-audit commit:
  `923eca42a8f892a18774df8f483dd78404807d59`.
- Prior report, evidence, and finding artifact SHA-256 values respectively:
  `a4f5a662be4b4a7d9569152dd6220444b45eb490a523a74cf5367e639c987214`,
  `d368c9f3f699c60c1f27a33d53db1f64b2b932aaf03319969646abb6ffe6133f`,
  and
  `ef42bb65c1763f8453e9a26c0452cfcc3665234767f1b5176f8f1a8788459b9b`.

This re-audit preserves the predecessor authority except for the narrow frozen
SRA-001 through SRA-009 overlay. It does not infer new sampling, HWPR,
polarimetry, ranking, acceptance, or production authority.

## 3. Disposition axes

| Axis | Disposition | Basis |
| --- | --- | --- |
| Scientific scope and preserved numerical authorities | **Partially conforming; not sufficient for acceptance** | Fixed FWHM values, isolated motion thresholds/v95, factor-1 inclusion, existing-filter counterfactual intent, phase-zero conventions, and the no-ranking boundary are present. The production binding to those authorities is defective. |
| Total-intensity/HWPR boundary | **Fail, P1** | Enabled polarimetry is rejected, but the carrier used by supported total-intensity diagnostics is produced through alignment that still branches on stale `calib.run_hwpr`; explicit HWPR-dependent status is also not dominant for every invalid-state permutation. |
| Observation lifecycle and exact validity context | **Fail, P1** | The source carrier is captured after interpolation; consumption checks only `obsnum`; final context is computed from already RTC-downsampled PTC data, candidate factors are applied again, scan guards are omitted, and required exclusion fractions are absent. |
| Cadence/filter provenance | **Fail, P1/P2** | Normal epoch-valued 488 Hz grids are falsely rejected as irregular; requested/effective consistency can be hidden by invalid realized cadence; frequency-derived requests are misclassified. |
| Numerical/resource safety | **Fail, P1/P2** | The work guard undercounts coherent response work and omits interval lookup/category work and auxiliary storage. Exact finite `M_max` is lost above `INT_MAX`. |
| Successor product contract and identity | **Fail, P1** | The executable validator accepts complete malformed conditional schemas, omits required exact bindings, and products embed a nonexistent generic contract ID plus the point profile for every mode. |
| Build/source identity | **Fail, P2** | A 40-hex value is written, but configure-time-only capture permits stale clean identity after a source/index/ref change followed by an ordinary build. |
| Atomic publication and provenance join | **Fail, P1** | NetCDF staging mechanics are mostly sound, but the fixed-name sidecar is replaced before rtcdiag finalization and is not a canonical raw-input manifest. A failed rtcdiag publication can invalidate the prior-good product's manifest join. |
| Observe-only/non-interference evidence | **Fail, P1 evidence closure** | Static inspection found no direct science mutation, but the required production A/B, B/A, failure, repeat, and permitted-parallel permutations are absent; the existing test compares unrelated local vectors. |
| Scope control | **Conforming** | No Stage A ranking, recommendation, candidate selection, factor application, Stage B authority, Unity/reduction action, or `ptcdiag` change was found. |

## 4. SRA-001 through SRA-009 outcome

| Finding | Outcome | Severity | Independent determination |
| --- | --- | --- | --- |
| SRA-001 | **Not closed** | P1 | The carrier labeled pre-interpolation is captured after detector-grid alignment; production consumption does not bind observation index and telescope path; lifecycle permutations are absent. |
| SRA-002 | **Not closed** | P1 | Total-intensity source preparation indirectly depends on stale/uninitialized `calib.run_hwpr`; the explicit unsupported-HWPR seam is not deterministic for all prerequisite states. |
| SRA-003 | **Not closed** | P1 | Exact validity is evaluated on an already-decimated/coarsened PTC domain, then decimated again; scan-boundary guards and required category fractions are missing. |
| SRA-004 | **Not closed** | P1/P2 | Actual work and storage are underbounded; a concrete admitted case exceeds the configured work limit; exact large finite `M_max` is discarded. |
| SRA-005 | **Not closed** | P1 | Conditional cardinality, full shapes, exact identities, FIR/manifest digest binding, cadence/filter provenance, and mode-specific contract/profile identity are not enforced. |
| SRA-006 | **Not closed** | P2 | The committed build identity can become stale while still appearing clean and exact 40-hex. |
| SRA-007 | **Not closed** | P1 | Numerical expectations reuse candidate helpers and non-interference tests do not cross the production boundary or lifecycle permutations. |
| SRA-008 | **Not closed** | P1/P2 | Ordinary Unix-epoch native grids can be rejected; requested/effective and effective/realized consistency are not independently truthful. |
| SRA-009 | **Not closed** | P1 | The externally joined fixed sidecar is published first; later rtcdiag failure can preserve a product whose referenced prior manifest has been overwritten. |

## 5. Principal findings

The exact records, severities, evidence locations, impacts, and bounded closure
conditions are in the CSV. The principal technical blockers are:

1. The carrier claims source telescope rows before interpolation but is captured
   after `load_and_point_telescope_data`, whose alignment replaces telescope
   columns and `TelTime` on the detector grid.
2. The final exact-context writer consumes RTC-processed PTC data and applies
   every hypothetical factor again, so factor 1 is already coarsened and higher
   factors are double-decimated. OR-coarsened flags cannot reconstruct the
   native counterfactual.
3. The resource preflight omits one complete unaliased response evaluation for
   every `M > 1`. For `Q=257`, `M_max=2`, `L=100001`, `D=N=1`, it admits
   estimated work `360,703,607`, while the exact invoked path requires
   `515,005,150`, above the configured `500,000,000` limit. Additional interval
   lookup, category allocation, source arrays, and raw-manifest storage are also
   uncharged.
4. The executable successor contract accepts an available product with every
   production three-dimensional candidate field collapsed to one dimension and
   bogus identity bindings, and an unavailable product declaring count 7 with
   no candidate dimension. Both return no validation errors.
5. The value called a canonical raw-input manifest contains configuration and
   observation-resolution provenance but no raw input membership, roles, paths,
   or hashes. Because its fixed filename is replaced before rtcdiag finalization,
   a later failure can break the prior generation's provenance join.
6. An ideal 488 Hz grid near Unix epoch `1.7e9 s` has representable step spread
   `2.384185791015625e-7 s`; the implementation tolerance is `1e-9 s`, so the
   valid grid is marked `irregular_realized_cadence`.

## 6. Positive controls retained

- Enabled polarimetry is rejected before reduction execution, and production
  rtcdiag explicitly requests total-intensity analysis.
- Fixed diffraction FWHM values are exactly 4.66, 5.94, and 8.48 arcsec for
  a1100, a1400, and a2000.
- The isolated motion helper uses linear empirical eligible-interval `v95`, an
  inclusive 1 arcsec/s low-velocity exclusion boundary, and invalidity strictly
  above 3600 arcsec/s. It explicitly reports insufficient source rows, no valid
  intervals, no guarded support for a short scan, and no complete context for an
  unusable applied scan. Its production carrier is nevertheless the wrong grid,
  so these semantics are not bound to the approved source domain.
- Factor 1 is present; ordinary finite candidate enumeration does not silently
  truncate; existing RTC coefficients or identity `[1]` are used without
  candidate-specific FIR synthesis or suitability claims.
- Phase-zero alias enumeration and factor-1 zero-alias/
  `not_applicable_no_decimation` behavior are numerically consistent in isolated
  spot checks. No Stage A acceptance threshold or ranking is introduced.
- Guard masks and residual science flags have separate in-memory categories;
  detector output accounting and temporal “at least one valid detector”
  semantics are internally consistent on the supplied domain. The supplied
  domain and boundary mapping are wrong.
- Schema and algorithm identities use `rtcdiag-v3` and
  `rtc-learned-sampling-stage-a-v3`; writer, finalizer, and contract enumerate
  the same 81 candidate arrays; four preparing mode profiles exist.
- A finite status/reason enum and serialized vocabulary exist, but prerequisite
  ordering, nonfinite-time classification, large-`M_max` mapping, and the weak
  executable contract permit semantically wrong reasons. Vocabulary presence
  is not equivalent to deterministic truthful status.
- The existing `rtc_diagnostics` contract check and all inspected `ptcdiag`
  source blobs are invariant.
- Unique adjacent `O_EXCL` staging, sync/close before rename, no destructive
  pre-delete, task-created temporary cleanup, and post-publication append refusal
  are present. They do not make the external fixed-sidecar generation atomic.
- Static source review found no direct Stage A write to science samples, flags,
  timestamps, cadence, FIR, factor, RTC/PTC/map inputs, weights, maps, or
  non-rtcdiag products. Required production non-interference evidence remains
  absent.

## 7. Deterministic validation summary

- Fresh Release configure: pass.
- `citlali_cli`, `citlali_test`, `citlali_safety_test`, science-map truth, and
  science-map FITS-product targets: pass.
- Focused learned-sampling/config tests: 34/34 pass.
- Ordered-writer tests: 7/7 pass.
- Full CTest: 653 executed, 653 passed, 0 failed, 1 disabled of 654 enumerated.
- Product-contract/profile focused Python tests: 38/38 pass.
- Complete baseline Python discovery: 177/177 pass.
- Complete config preflight: pass, including 127/127 config tests, four mode
  kits, 8/8 compact-compatibility cases, 100% compact-surface coverage, config
  authority, and raw-execution census.
- Validation ledger: 60 records valid.
- Intended-science-change ledger: 3 changes and 5 integration commits valid.
- Validation profile registry: valid, 4 active and 12 preparing profiles.
- Phase 5 readiness command: pass and truthfully reports `preparing`, promotion
  not ready.
- Session-exit census: 710 dependencies, 0 library exits, 0 CLI exits, 0 growth.
- Candidate `git diff --check`: pass.
- Independent resource, cadence, and malformed-contract adversaries reproduce
  the failures described above despite all ordinary gates passing.

Exact commands and outputs are preserved in the evidence artifact.

## 8. Stop statement

This independent audit made no application, configuration, test, build-system,
validation-product, canonical coordination, or production-code edit. Only the
three approved documentation audit artifacts were created and committed on the
audit branch.

No repair was performed. No push, merge, integration, Unity access/request,
science reduction, Stage B activity, downstream launch, external contact,
production authorization, task-created repair, or task-created re-audit
occurred. The work stops here for coordinator/owner review.
