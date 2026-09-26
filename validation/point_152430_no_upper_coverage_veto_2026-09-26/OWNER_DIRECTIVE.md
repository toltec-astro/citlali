# Codex directive: POINT 152430 — remove the high-coverage veto, discard the stopped attempt, and rerun

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** Implement, check, delete the superseded attempt’s products, execute the replacement comparison, assess, and stop. Do not return another proposal in place of execution.  
**Scope:** POINT 152430/0/2, a2000 only, with its own associated tune 152430/0/1 and the previously selected networks 11/12. Ordinary MAP; BASE-POINT, CORR, and DIAG; five branches per policy through T10.

> Greater coverage alone is not grounds for rejecting a valid map pixel. The maximum coverage in the source-masked background describes the training sample; it is not a physical boundary of usable data. Remove that automatic veto, retain the existing reasonable noise prediction, and complete the intended experiment. Delete the superseded stopped attempt’s products rather than preserving them for historical completeness. Protect its input data.

## 1. Authority and unchanged experiment

This order supersedes two requirements for this diagnostic route:

1. Coverage above the background training maximum must no longer, by itself, invalidate admission-noise transfer, exclude a pixel from eligible support, or trigger a source-contact/boundary failure.
2. The previous stopped 152430 attempt does not need a retained report, compact archive, reconstruction package, or historical failure record. Its derived products are authorized for deletion as specified below.

The complete comparison remains the one defined in `CODEX_INDEPENDENT_POINT_A2000_FROZEN_METHOD_VALIDATION.md`, with the scientific-method change above. This is explicit authorization to rerun it, not just to test the initial candidate. Read applicable repository instructions and recover the current implementation. Do not repeat completed workspace inventory/purge work, reopen observation selection, or run another historical 152389 campaign.

The source/branch reported for the earlier execution—science revision `54adaeebd2068cf827fe39f0aae8c3fbddcb03e0`, branch `codex/map-successor-first-forward-001`—are discovery anchors, not instructions to reset the worktree. Preserve unrelated changes and the working source/environment. No merge, push, production-default promotion, Git-history rewrite, or Unity modification is authorized.

## 2. Protect inputs and erase the superseded attempt

First recover the exact input associations into a small **new-run input manifest**. Preserve the observation’s original detector files, telescope/auxiliary streams, associated tune files/products, APT, matching and calibration inputs, and metadata needed to identify and reduce them. Do not substitute adjacent observations, tunes, or calibration products. Preserve reusable pipeline source, tests, required method definitions/thresholds, and the build environment; these are not failed-run products.

Then delete the stopped attempt’s derived products, including its maps, masks, learned noise products, reports, figures, compressed provenance, reviews, seals, cleanup archives, and any remaining temporary RTC/PTC arrays. Do not relocate them to an archive or quarantine, make a backup, or recreate their report before deletion.

Resolve and verify these exact targets against the actual workspace:

```text
~/work_toltec/local_data/citlali-validation/development-runs/
  successor-point-152430-independent-a2000-20260926/

<working repository>/validation/
  point_152430_independent_a2000_2026-09-26/

<working repository>/handoff/
  POINT_152430_INDEPENDENT_A2000_2026-09-26.md
```

Remove other demonstrably attempt-owned scratch/report copies by individually resolved paths, including copies in other local worktrees if present and inactive. Update current status/handoff references so they no longer advertise the erased attempt as a maintained result. Do not delete shared source/build trees, unrelated experiments, the retained 152389 comparison, or anything on Unity.

Before deletion, verify the targets are inactive, do not traverse symlinks, and establish that protected inputs are outside them. If a required input exists only inside a target, move it to the established input area, verify its identity, and update the new input manifest before deleting that target. Recheck protected inputs afterward. No requirement to demonstrate regeneration of the discarded *results* applies.

The owner has authorized this scoped deletion; do not ask again. A concise summary of removed roots/bytes and input preservation is sufficient. Do not construct another permanent per-file failure archive or chain of cleanup reviews. Use ordinary tracked-file deletion for repository copies. Existing commits may still contain old bytes: do not rewrite Git history to purge them, and do not claim that ordinary deletion does so.

## 3. Change the admission rule, not the noise model

Separate **coverage extrapolation metadata** from **numerical/data usability**.

For an otherwise eligible map pixel with valid required inputs and a finite, positive noise prediction, `coverage > background_training_maximum` is informational only. It must not become a hard failure indirectly through BASE eligibility, neighborhood full-footprint support, covariance-transfer flags, serialization/reload, or the source-component contact guard. Apply this semantics consistently to BASE-POINT, CORR, and DIAG without replacing their distinct numerical noise operands with one another.

Retain the existing computational continuation of the noise-amplitude function. Inspect its formula and check its values over the actual coverage range encountered by the replacement M0. This is a small implementation sanity check, not a new scientific study. A finite positive continuation with no singularity, abrupt discontinuity, or pathological collapse over that range is sufficient for this bounded run. Do not demand direct blank-sky validation at every target coverage.

Do **not** introduce a percentage allowance, a new upper-coverage cutoff, special exceptions for 458/475, an obsnum-specific bypass, a new inverse-square-root law, or an automatic boundary-value clamp. Do not shrink the source mask to make the training range encompass the source. Do not retune noise amplitudes, admission thresholds, or covariance to produce acceptance. A genuinely broken numerical prediction remains an error requiring a specific diagnosis; high coverage is not itself that diagnosis.

Preserve checks for missing/non-finite data, nonpositive or unusable variance, invalid required covariance, invalid projection, real support gaps, and the actual geometric feedback boundary. Low-coverage behavior is unchanged by this order. Remove only the high-coverage cause of invalidity; do not globally overwrite validity flags or disable the contact guard.

Record the background training range, maximum applied coverage, and number of extrapolated pixels as ordinary diagnostics. Keep the estimates labeled as model-based/conditional where appropriate. Usability for admission is not a claim that their observational uncertainty or false-admission probability has been empirically calibrated.

## 4. Focused checks and release to execution

Add or update small deterministic tests demonstrating that:

- A valid positive prediction remains usable just above and farther above the training maximum, including the reported 458/475 versus 456 case. The removed cutoff has not merely moved elsewhere.
- Noise values and decisions on previously supported inputs are unchanged; existing high-coverage numerical predictions are unchanged when already well behaved. Source contact with these otherwise valid pixels no longer stops admission.
- A pixel with high coverage **and an independent real defect** remains invalid. Missing MAP support, bad variance/covariance, and actual geometric boundary contact retain their previous behavior.
- BASE and supplemental policies, including saved/reloaded operands, distinguish extrapolation metadata from unusable support consistently. Accounting, source rejection, candidate replacement, and native projection tests remain intact.

Use existing tests and compact/synthetic fixtures; do not retain the erased failure bundle as a new fixture. Obtain one bounded independent preflight review of the patch and execution plan, then proceed. Ordinary implementation/test repairs do not require another owner decision. No new Monte Carlo calibration or separate coverage-transfer campaign is authorized or required.

## 5. Start fresh on the same acquisition

Use a new dedicated output root, for example:

```text
~/work_toltec/local_data/citlali-validation/development-runs/
  successor-point-152430-a2000-no-upper-coverage-veto-20260926/
```

Reverify the preserved, previously reduction-ready inputs. **No download is expected.** Only if a genuinely required input is now missing may the original directive’s narrow Unity read/download authorization be used to recover its exact recorded identity. Do not fetch additional arrays, a different observation, or old RTC/PTC checkpoints.

Regenerate RTC/CAL/AST/PTC and ordinary M0 from those inputs. Do not restart from the deleted attempt’s maps or learned covariance. Learn the new run’s M0 admission-noise scale, full covariance C0, core K0, and measurement geometry once using the unchanged construction recipes; freeze them across policies and branches. Evaluate the continuation sanity check on this M0, not a separate trial reduction. Do not transplant numerical noise fields or source geometry from 152389.

Bind the new source/configuration, binary/environment, exact inputs, full-precision policy thresholds, and execution limits in the existing compact run record. The new run is not an unchanged-method replay of the discarded attempt.

Keep the 2-arcsec grid, AST-centered 40-arcsec feedback radius, containing-pixel projection, accepted uniform-coefficient/eligibility conventions, unity feedback gain, complete replacement, observed-entry rank-10 PTC, fresh residual-specific learning, and paired-r behavior unchanged. Keep the seven-parameter Gaussian-plus-constant guard, fit conventions and limits, amplitude S/N > 10, both sigmas > 1.5 arcsec, and centroid within 20 arcsec of the expected origin. All three policies use the full C0 source guard; DIAG differs only in its supplemental covariance treatment.

## 6. Execute the full original matrix

BASE-POINT keeps threshold-3 source-connected admission. CORR and DIAG retain their respective supplements and exact full-precision a2000 thresholds from the accepted calibration. Do not recalibrate or adjust them. Candidates contain selected **measured map brightness**, not a fitted Gaussian or neighborhood prediction, except for the specified initial seed intervention.

Run BASE-POINT REF through T10 first. From its accepted T10 fit construct and freeze the shared test shapes before examining the other policy endpoints or executing their response/seed branches:

- `G`: unit-peak Gaussian using the BASE-POINT REF T10 centroid, widths, and orientation; `A` is its fitted amplitude above the constant.
- `H`: twice both Gaussian widths, displaced by 0.5 times the original M0 major FWHM along the positive first physical tangent axis.
- Radial taper `w`: unity through 30 arcsec; `[1 + cos(pi*(r-30)/6)]/2` from 30 to 36 arcsec; zero beyond 36 arcsec. No post-taper renormalization.

For each policy run:

| Branch | Immutable pre-PTC data | Initial feedback model |
|---|---|---|
| REF | Original `y` | Its normal policy candidate |
| CORE-PLUS | `y + P(+0.10*A*G*w)` | Its candidate from the perturbed M0 |
| CORE-MINUS | `y + P(-0.10*A*G*w)` | Its candidate from the perturbed M0 |
| SHOULDER | `y + P(+0.05*A*H*w)` | Its candidate from the perturbed M0 |
| SEED-SHOULDER | Original `y` | Its own REF initial candidate plus the same shoulder raster, once only |

`P` is the unchanged native AST projection. Preserve invalidity and include valid source-bearing entries used by PTC even when not used in final MAP. Genuine missing shape support still blocks the dependent test; high-coverage metadata alone does not.

At each feedback application start from that branch’s immutable input, subtract its current candidate, freshly learn/apply PTC, restore exactly the subtracted operand, and accumulate MAP. No cumulative injection, accumulated old models, or borrowed cleaning basis. The seed has no exemption after its first application.

**Hard execution limits:** 15 trajectories, at most 150 feedback application attempts including failed calls, four distinct initial physical inputs, at most nine T10 ideal-addition controls, and at most three REF T10 endpoint covariance analyses. Share mathematically identical M0 calculations. Each completed trajectory includes M0 and T1–T10; T10 uses m9, and m10 is saved but not applied. No hidden retries, cycle 11, extra seeds, new observation, or additional array.

## 7. Assess the science, not just whether admission runs

Retain compact M0–T10 histories and summarize M0, T1, T3, T5, and T10. Use the original directive’s measurement definitions and frozen common geometry: peak, centroid, gauge-safe shape/widths, constant, fixed-template/aperture measurements, accepted/candidate support, and terminal drift.

Compare each response or seed branch with its **own policy REF**. Use the corresponding injection-only ordinary-MAP controls and T10 ideal-addition controls. Do not fit signed difference maps with the positive-source guard. Run the existing endpoint covariance companion once per completed REF T10; it is diagnostic and must never alter feedback. No new noise/refit ensemble.

Lead with signed measurement changes, their fraction of the whole source, and their size relative to the corresponding conditional measurement uncertainty. Distinguish pixel noise from peak, template, and aperture errors; distinguish injection-relative response from whole-source error. Label model-transfer qualifications without automatically turning them into another campaign. A paired difference divided by an uncertainty is a practical size comparison, not a calibrated significance test.

The scientific questions remain whether the shoulder benefit and compact-response tradeoff recur, and whether full covariance adds useful information beyond the DIAG supplement. More admitted pixels or a brighter map is not by itself success. Do not pursue inconsequential seed/peak/drift details solely because they are measurable. Explain any material source loss, bias, coherent residual, or instability.

## 8. Resources, genuine failures, and closeout

Use one shared upstream parent where practical and useful parallelism without nested thread oversubscription. Establish and enforce a peak disk ceiling from actual inputs/allocations and available space; use any already adopted workspace hard budget. The prior 16,159,001,984-byte campaign ceiling is a planning reference to verify, not a global quota. Avoid whole-timestream copies per policy/cycle. No environment migration or general cache framework.

High coverage alone is no longer a stop condition. Genuine source rejection, missing inputs, unusable numerical predictions, failed subtract/restore accounting, new processing errors, actual unsupported/geometric contact, and resource-limit violations retain their appropriate stops. A failed BASE REF blocks dependent shape tests. Other failures block only genuinely dependent work. Do not lower thresholds, replace the observation, or repeatedly rerun until the result looks favorable.

Deliver one current assessment and full 15-cell completion/status table, with useful maps/masks, compact measurements, shared injection rasters, noise-model descriptions, exact inputs/configuration and reproduction commands, resource totals, and a bounded final review. Keep the new source and useful compact result/reproduction record in the permanent repository, outside disposable local_data. Remove inactive bulk RTC/PTC arrays and per-cycle restart payloads at closeout; do not accumulate hidden copies.

If a new real blocker prevents completion, report the exact cause and completed/unavailable work, but do not manufacture another elaborate historical failure archive. The prior stopped attempt stays deleted. A brief cleanup summary is enough; do not preserve its detailed result merely to explain this rerun.

Commit the reviewed code, current result, and tracked deletions locally. Report actual execution counts, input transfers if any, retained storage, and the result location/commit. No push or merge. CUT3, JINC, other arrays/observations, new threshold/covariance families, and further uncertainty campaigns remain outside scope.

**Finish after the replacement comparison, concise scientific assessment, final review, and temporary-product cleanup.**
