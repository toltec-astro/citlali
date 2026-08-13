# SCI-CAL-001 successor-8 bounded technical repair handoff

Date: 2026-08-13

Handoff ID: `SCI-CAL-001-SUCCESSOR-8-REPAIR-DISPATCH-001`

Status: frozen owner-authorized scope; role-separated repair not launched

## Exact authority and proposed base

- Owner acceptance:
  [`SCI-CAL-001_SUCCESSOR_7_OWNER_ACCEPTANCE_2026-08-13.md`](SCI-CAL-001_SUCCESSOR_7_OWNER_ACCEPTANCE_2026-08-13.md),
  SHA-256
  `53631061cf1faa10a21ffb810830fc923d7072f3046f8d2e932acb1aa15c7f39`.
- Immutable re-audit report:
  [`../SCI-CAL-001_SUCCESSOR_7_REAUDIT_2026-08-13.md`](../SCI-CAL-001_SUCCESSOR_7_REAUDIT_2026-08-13.md),
  SHA-256
  `93e57a60aba11ed3f2996b08f0b35b639d7d2abc86466adedff9428e74ee33bc`.
- Immutable local evidence:
  [`../evidence/SCI-CAL-001_SUCCESSOR_7_LOCAL_EVIDENCE_2026-08-13.yaml`](../evidence/SCI-CAL-001_SUCCESSOR_7_LOCAL_EVIDENCE_2026-08-13.yaml),
  SHA-256
  `974bc36e9a28647eda5b4932a1f0b85e79f8a4c9d3f97fbf6dcc92b6c2544029`.
- Machine-readable finding ledger:
  [`../proposals/SCI-CAL-001_SUCCESSOR_8_REPAIR_FINDING_LEDGER_2026-08-13.yaml`](../proposals/SCI-CAL-001_SUCCESSOR_8_REPAIR_FINDING_LEDGER_2026-08-13.yaml).

The exact pushed application base is
`9037314fd84241fa535c486d4ffb28966bb0394d` on
`origin/codex/repair-sci-cal-001-successor-7`:

- parent: `211e2f16f6354609de3ce6c6ee526d8aa4c6c59c`;
- tree: `9d2095159ac208a1096519a6fa710172275d3b73`; and
- parent-to-candidate binary-patch SHA-256:
  `e761c4c25070ea7b925f8e75c05c5dbec05c1849132312f279112482dcded4e0`.

The proposed repair branch is
`codex/repair-sci-cal-001-successor-8`. This handoff does not create the
branch, worktree, or repair task.

## Frozen scope and retained states

Successor-8 is limited to F008-A, F008-B, F009-A, and F009-B. F002 through
F007 retain their accepted bounded dispositions. F001 and F010 remain open,
conditioned, and outside repair. The CAL axes remain `approved`,
`nonconformant`, `in_progress`, and `fail_closed`, with verdict `amend`, until
a later independent re-audit and owner disposition.

No new CAL science, calibration arithmetic, RTC/PTC/mapmaking numerical
behavior, uncertainty/covariance product, empirical response-fidelity claim,
cross-package architecture, global transaction/rollback layer, accepted-run
rewrite, validation weakening, or production promotion is authorized.

## Mandatory READY checkpoint before edits

A later role-separated repair task must start from the exact pushed base in a
fresh clean worktree and return before editing with:

1. exact local and live-origin base/ref, parent, tree, patch digest, proposed
   branch, and clean worktree/index state;
2. confirmation that the successor-8 branch was absent before creation;
3. finding-to-change and finding-to-test traceability within the exact
   nine-path ceiling below;
4. the focused F008-A/F008-B/F009-A/F009-B counterexample plan before any
   broad-test execution;
5. preservation evidence for F002--F007 and confirmation that F001/F010
   remain conditioned and outside scope;
6. confirmation that historical readiness and profile/run authorities remain
   immutable, `sig2noise_pixel_I` stays prohibited for new/current products,
   and the v4 epoch/profiles remain `preparing`; and
7. confirmation that no science/arithmetic redesign, global transaction,
   unrelated writer refactor, validation weakening, or production promotion
   is proposed.

Any path or behavior outside this handoff requires a stop and separate
coordinator/owner review. Silence prohibits a capability.

## Exact changed-path ceiling — nine paths

F008-A Pointing publication lifecycle:

- `include/citlali/core/engine/detail/pointing_output_impl.h`
- `tests/test_config_scaffold.cpp`
- `tests/test_science_map_fits_products.cpp`

F008-B Science Wiener-filtered publication lifecycle:

- `include/citlali/core/pipeline/map_filtering_setup_outputs.h`
- `include/citlali/core/pipeline/filtered_observation_outputs.h`
- `include/citlali/core/pipeline/filtered_coadd_outputs.h`

The two C++ test paths may serve both F008 subfindings and count only once.

F009-A/F009-B semantic admission and executable counterexamples:

- `tools/baseline/audit_reduction_run.py`
- `tools/baseline/test_audit_reduction_run.py`

Coherent repair handback only:

- `doc/REFACTOR_STATUS.md`

These groups contain exactly nine unique paths, all present at the exact
candidate. No new file or build-system path is authorized. A demonstrated
need outside the ceiling stops the task before editing that path.

## F008-A — Pointing required FITS publication

Before clearing any owning vector, atomically publish every required Pointing
RawObs, FilteredObs, RawCoadd, and FilteredCoadd data and noise FITS artifact.
Publication uses the existing per-artifact stage, synchronize, close, reopen,
validate, and atomic-replace mechanism. Record output publication only after
the required artifacts have successfully published.

Focused executable tests must exercise all four production map types and both
data/noise families through the actual owner lifecycle. They must verify final
existence, complete reopen/readback, CALID/PKGID joins where required, stage
cleanup, and preservation of an existing valid final when a replacement
write, close, reopen, validation, or replace step fails. A missing new final
or stale old final must never be recorded as the completed new product.

## F008-B — Science Wiener-filtered required FITS publication

Atomically publish all required Science Wiener-filtered observation and coadd
data and noise FITS artifacts. Publication must occur before their owners are
cleared. The post-filter output lifecycle must no longer skip solely on a
false assumption that filtering already wrote complete final products; any
skip must follow successful publication under the explicit output policy.

Focused executable tests must cover observation/coadd and data/noise routes,
actual Wiener-filter write/finalize/output sequencing, final reopen and join
validation, stage cleanup, retry/skip truth, and preservation of existing
finals under injected failure. This authorizes no broad writer redesign and
no global cross-output transaction or rollback architecture.

## F009-A — exact YAML scalar semantics

Requested-config preimage validation must compare exact YAML scalar type and
value recursively. In particular, boolean `true` must not compare equal to
integer `1`; analogous YAML scalar-type collisions must fail closed with a
cause-specific semantic error. Canonical requested-response identity and
digest recomputation remain unchanged.

Focused Python tests must include the accepted boolean/integer counterexample,
valid identical typed scalars, nested mappings/sequences, null/string/number
distinctions, and recomputed otherwise-self-consistent package identities.

## F009-B — exact selected-APT membership coverage

For an effectively calibrated v4 package, the package-local selected APT must
be covered exactly by the declared ordered detector-row association and
factor basis. Reject duplicate, missing, unused, or extra rows even when all
referenced rows, digests, CALID, and PKGID are otherwise self-consistent. This
matches the existing production admission rule and does not create a new APT
lineage system.

Focused Python tests must include the accepted extra-row counterexample,
missing/duplicate/out-of-range associations, row reorder and value tampering,
valid exact coverage, valid legacy optional-modern lineage, and supported
effectively uncalibrated v4 behavior.

## Fresh broad validation matrix

Only after every focused counterexample passes, run the complete fresh matrix
on the exact successor-8 candidate. It includes configuration/build,
`citlali_cli`, affected test binaries, full CTest, public-header
isolation/linkage, `tools/config/run_config_preflight.py --require-all`, full
baseline Python discovery, focused baseline/product comparison, validation
product/profile/ledger and science-change gates, raw execution census,
atmosphere-generator check, session-exit growth audit, complete YAML/FITS/
ECSV/TOD failure-safe publication coverage, and ordinary Phase-5 readiness.
Record every command, result, skip, and unavailable prerequisite.

The historical write-producing `phase5_readiness --verify-fixtures` result
remains `failed_owner_waived_never_passed`, solely for six point plus twelve
science immutable historical `sig2noise_pixel_I` errors. It must reproduce
that bounded truth and is never relabeled pass. Prior candidate results are
not fresh successor-8 evidence.

## Preservation and handback

Preserve F002--F007 behavior and accepted claims. Preserve F001/F010 as
conditioned external dependencies. Preserve historical products, verifier,
contracts, profiles, accepted runs, and declared outcomes byte-for-byte.
`sig2noise_pixel_I` remains prohibited in every new/current product, and the
v4 current-production epoch/contracts/profiles remain `preparing`.

Return the exact repair commit, parent, tree, standard binary-patch digest,
changed paths, commands/results/skips, focused and broad evidence, finding
traceability, and unresolved dependencies. Stop for coordinator review. Do
not start a re-audit, merge, push, access or request Unity, run a reduction,
contact external parties, or launch downstream work.

## Non-authorization

This handoff does not launch repair. It authorizes no application edit by this
coordination task, science/arithmetic redesign, RTC/PTC/mapmaking numerical
change, global transaction, unrelated writer refactor, validation weakening,
production promotion, accepted-evidence rewrite, Unity access or request,
reduction, external contact, downstream launch, re-audit, merge, or push.
