# Codex directive: POINT 152389 — scientific importance and archive compaction

**Owner:** Grant Wilson  
**Date:** September 26, 2026  
**Status:** Assess saved results, establish a compact reproducible archive, execute verified cleanup within the boundaries below, review, and stop.  
**Scope:** Existing POINT 152389 validation work and its experiment-owned local products. No new scientific comparison campaign.

> Put differences on the scale of the noise and the measurements we actually care about. Record scientifically negligible effects and move on. Preserve the evidence and the ability to reproduce the work—not every intermediate array. “It might be useful later” is not sufficient justification for indefinite retention.

## 1. Priority, authority, and existing work

This is the next work item, ahead of another scientific experiment. Park the unexecuted remainder of the a2000 three-pixel counterfactual and the proposed second-observation comparison. Use any already completed results as saved evidence; do not finish or extend those experiments merely for closure. Do not interfere with running jobs or remove their inputs or outputs.

This order authorizes two activities:

- **Scientific-importance assessment:** use existing products to relate reported differences to matching noise/uncertainty scales and scientific consequences.
- **Archive compaction:** inventory, preserve a sufficient reproduction package, validate the relevant regeneration routes, and remove explicitly identified inactive, reproducible bulk products.

For those verified derivative files only, this order supersedes earlier requirements to preserve every intermediate, keep an entire sealed payload online, or perform no destructive cleanup. It does not authorize deleting original observations, indispensable calibration/metadata, unrelated work, unique evidence, or inputs whose recovery has not been established. Historical records must remain truthful after compaction.

Read applicable repository instructions and recover the actual worktree, latest completed status, accepted methods, and product dependencies. Principal records are the completed counterparts of:

```text
CODEX_POINT_152389_POST_FRUIT_NOISE_AND_UNCERTAINTY_AB.md
CODEX_POINT_152389_RESPONSE_AND_ROBUSTNESS_CD.md
CODEX_POINT_152389_GUARDED_FEEDBACK_COMPARISON.md
CODEX_POINT_152389_A2000_SAVED_PRODUCT_DIAGNOSIS.md
CODEX_POINT_152389_A2000_THREE_PIXEL_M1_COUNTERFACTUAL.md
```

Include the original MAP/FRUIT, uncertainty, JINC-comparison, and admission/guard records only as needed for the assessment or archive dependencies. Recover completed numerical products, not just directive text.

Discovery anchors, not an assumption about current HEAD:

```text
Validation root:
/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/

Previously reported worktree / branch:
/private/tmp/citlali-point-additive-20260923
codex/map-successor-first-forward-001

Guarded-comparison completion:
26a3047c654386e38adac860b91491136888bc64

Saved-product diagnosis completion:
11b4d2c07a81cc5876bd3da833c4168731e34aea
```

Resolve and record the exact owned directories before cleanup. A directory name, age, or location under the validation root is not sufficient deletion authority.

## 2. Assess importance in measurement units

### 2.1 Reuse the existing evidence

Lead with the completed guarded BASE-POINT / CORR / DIAG comparison across the supported arrays, including a2000's seed dependence, compact signed response, shoulder recovery, and late drift. Do not repeat the detailed pixel investigation. Use each branch's own policy-matched reference and the existing ideal-addition controls, masks, apertures, and measurement definitions.

Allowed assessment calculations are cached-result extraction; deterministic map differences, sums, existing fixed-template/aperture measurements and background statistics; and arithmetic propagation through already saved covariance/measurement operators. For an existing linear measurement with weights `w`, `sigma_Q² = wᵀ C w` is permitted when the required saved operands and validity assumptions are available. Do not invent new weights, masks, templates, or covariance parameters to obtain a more favorable comparison.

No new source fits, covariance fits, noise ensembles, candidate evaluations, or scientific processing are authorized for this assessment. A report cache miss must not silently launch inference. The strictly bounded regeneration checks in section 5 are separate exceptions for testing reproducibility, not additional scientific evidence.

### 2.2 Compare like with like

| Reported effect | Appropriate uncertainty or reference scale |
|---|---|
| Local map brightness residual | Local map noise at matching angular resolution, coverage, and support |
| Fitted peak-amplitude change | Uncertainty of the corresponding fitted-peak estimator |
| Frozen-template amplitude change | Uncertainty of that same template measurement |
| Fixed-aperture change | Uncertainty of that same aperture, including correlations and its background convention |
| Centroid or width change | Matching parameter uncertainty where available, plus absolute and fractional physical changes |

Do not substitute map RMS for fitted-peak uncertainty, fitted-peak uncertainty for local pixel noise, or independent-pixel noise for a correlated aperture sum. Assess broad coherent residuals using the existing measurement-scale diagnostics, not their pixel amplitudes alone. A full-map maximum is a descriptive extreme, not a preselected single-pixel significance test.

For each leading effect, report:

```text
measurement / array / policy / branch / stage
signed absolute effect and units
matching uncertainty or reference scale and its provenance
noise-scale ratio |effect| / sigma_Q, where interpretable
fractional change relative to the relevant source measurement
applicability and practical disposition
```

Distinguish matched endpoint estimates, conditional model-based estimates, historical reference scales, and unavailable scales. Include source-location transfer, coverage, estimator, and background qualifications where relevant. Prefer the best supported existing scale; do not conceal uncertainty by choosing the largest available value.

Paired reductions share the same observed noise. Their deterministic difference is not an independent measurement error. A noise-scale ratio here expresses practical size; it is not a p-value or calibrated significance of the paired difference. Do not add two map variances in quadrature to manufacture that significance.

### 2.3 Numerical context to recover, not new calibration

The earlier A+B results, quoted in the C+D record, give conditional fitted-peak scatters of approximately **7.91 / 5.68 / 6.63 mJy/nominal beam** for a1100/a1400/a2000. The saved diagnosis, as quoted in the three-pixel directive, reports an a2000 seed-dependent fitted-peak change of **−0.103601 mJy/nominal beam** and a maximum full-map difference of **4.706088 mJy/nominal beam**.

Authenticate the underlying values and definitions. The approximately `0.104 / 6.63 ≈ 0.016` peak comparison illustrates why that peak shift alone should be low priority. It does not validate 6.63 as the new CORR endpoint uncertainty or as the noise at the largest residual pixel. If a matching scale is unavailable, retain a clearly labeled contextual comparison rather than a false significance claim.

Preserve explicit denominators. A seed coefficient expressed relative to an artificial shoulder is not a whole-source error. The reported 9.28% compact-aperture discrepancy is relative to an injected increment, not the whole source. Show absolute measurement effects and, where meaningful, their scale relative to the source and its uncertainty. Do not compare aperture brightness-area units directly with mJy/beam.

## 3. Practical triage, not another acceptance framework

Assign a short, reasoned disposition to the leading findings:

- **Park:** comfortably below the relevant uncertainty and well below applicable scientific tolerances, with no demonstrated consequential instability or correctness issue.
- **Retain as a consequential question:** a material measurement effect, coherent residual, repeated bias, source-loss behavior, or instability warrants further attention.
- **Scale unresolved:** the available uncertainty is insufficiently matched to decide. State the limitation and whether it could plausibly change the priority; this does not automatically trigger another experiment.

Do not introduce a universal sub-mJy cutoff, a new numerical sigma threshold, or a combined quality score. The approximately 5% final calibration/PSF goals provide context; these partial-boundary tests do not become absolute calibration validations.

A small effect can still matter when it coherently accumulates or exposes a genuine implementation fault. Name the concrete consequence rather than invoking that possibility to keep every anomaly active. Numerical accounting, source-rejection semantics, data validity, and reproducibility checks retain their existing requirements. Scientific smallness does not excuse broken accounting or justify loosening replay tolerances.

Update the local owner-goals/status record and the reporting convention so future experiment proposals lead with the affected measurement, relevant noise scale, expected practical consequence, and decision the experiment could change. Do not retune admission thresholds, noise models, PTC, fitting, mapmaking, or feedback to implement this reporting change.

## 4. Inventory and define the retained archive

### 4.1 Measure the actual footprint

Inventory completed, experiment-owned POINT 152389 bundles and their referenced shared inputs. Identify the reported quarter-terabyte rather than assuming a total or an exact tenfold saving. Show per-bundle and product-class sizes, unique/shared dependencies, active-use status, and the largest contributors.

Distinguish logical file sizes from allocated/reclaimable storage where the filesystem permits. Account for hard links, clones, compressed copies, and symlinks without following them outside the approved scope. Moving files into a same-disk quarantine or backup is not storage reclaimed. Do not delete filesystem snapshots or unrelated backups.

Classify products into:

| Class | Treatment |
|---|---|
| Authoritative observations and indispensable inputs | Protect; reference the authoritative copy rather than duplicating it per experiment |
| Compact scientific evidence and reproduction records | Retain |
| Shared processing cache with a specific reuse benefit | Retain selectively, name the benefit and size, and identify its upstream recovery route |
| Inactive, reproducible bulk intermediates | Eligible for manifest-controlled removal after verification |
| Unknown ownership, dependencies, recoverability, or active use | Retain as a named exception |

Use **approximately 25 GB as a provisional target for retained derived evidence**, not a hard deletion quota. List shared caches separately but report the combined footprint and achieved reduction. Do not relabel most of the old archive as a cache to claim success. Protected originals remain outside the derived-evidence target but visible in dependency accounting.

### 4.2 Preserve a compact, usable scientific record

Retain existing assessments, reviews, directives, manifests, numerical tables, useful figures, failure summaries, and small regression examples. Preserve final maps and associated support/noise/measurement products for completed comparisons. Keep inexpensive all-cycle maps, masks, candidates, and measurements when useful; the goal is to remove large working arrays, not destroy small evidence indiscriminately.

Retain exact code revisions and necessary source changes, configurations, numerical environment/dependency identities, relevant thread/determinism settings, input/calibration identities, original hashes, and executable reproduction commands. Keep required code history reachable; do not prune branches or environments in this task. Record numerical dependencies, not credentials.

Preserve the exact injection/seed definitions and small operands, frozen thresholds/covariance descriptions, RNG identity and trial seeds, geometry, masks, and state needed to regenerate each retained result. A hash identifies missing bytes but does not recreate them. A commit and a seed are not sufficient when additional state affects the calculation.

Potential bulk-removal classes include per-cycle residual/cleaned/projected timestreams, redundant per-sample diagnostics, large working matrices, full synthetic-map banks, temporary renderings, and redundant experiment-local input copies. Verify their actual roles before classifying them. Keep trial-level measurement tables and sufficient exact generation records when removing simulated fields.

A single shared post-RTC/CAL parent may be a justified cache. Do not delete that cache unless recovery from the protected upstream inputs is established; the replay allowance below does not authorize rebuilding RTC/CAL. One cache must not multiply into one copy per branch.

### 4.3 Do not require instant restart everywhere

An inactive experiment may be reproduced from its retained original processing boundary; instant restart at every historical iteration is not required. Keep extra checkpoints only for a specific near-term use or a documented regeneration-cost benefit.

Conversely, do not present a saved map alone as an exact restart state. Preserve any additional operational state required by the chosen restart route, or explicitly require replay from the beginning. The reproduction dependency chain must terminate in accessible protected inputs, not another product scheduled for deletion.

## 5. Bounded regeneration verification

Freeze the retained-input list, deletion candidates, representative checks, and inherited comparison tolerances before pruning. Reuse existing replay evidence only when it actually demonstrates regeneration through the same code/configuration route using the proposed retained dependencies alone.

Otherwise, this order permits **at most one already completed a2000 CORR REF trajectory replay**, from its retained post-RTC/CAL parent through the original M0 and T1–T10. The ceiling is one initial PTC/MAP calculation plus ten feedback applications and the route's existing single cached fit per full map. No changed input, new intervention, extra iteration, or alternative policy is permitted. If that exact completed branch cannot be authenticated, retain the affected bulk class and report the limitation rather than substitute another experiment.

For synthetic-field storage, permit regeneration of **at most three already archived trials in total**, chosen prospectively to cover the retained generation route(s). Use their original parameters and seeds; generate no new trials or ensembles. Reuse existing small fixtures for other distinct product paths. Do not rerun the whole campaign to justify cleanup, and do not extrapolate a passing route to a materially different producer without supporting evidence.

Run regeneration in a separate scratch destination with only the proposed retained dependencies available to the runner, plus its declared code/runtime. Candidate bulk files may remain physically present for comparison but must be inaccessible to the generation process. Give the comparison step access afterward. Demonstrate the absence of hidden cache reads; changing an output directory alone is not sufficient.

Compare regenerated arrays, support, candidates, measurements, and statuses using the accepted exact or field-specific numerical rules. Preserve discrete-state agreement where required; handle basis sign/gauge freedom using existing rules. Do not relax tolerances because a difference is below observational noise. Record inputs actually read, commands, counts, runtime, and comparison results.

A mismatch or incomplete dependency proof blocks removal of the affected class, not independent verified cleanup. Do not retry large numerical runs automatically or rebuild the environment as a side project. Retain a concrete exception and finish the supported work. Remove successful verification scratch outputs after saving their compact evidence; retain only the minimal diagnostic evidence needed for failures.

## 6. Execute safe compaction and preserve truthful provenance

Prepare a machine-readable action manifest with exact paths, identities/hashes, sizes, classification, retained replacement or regeneration command, verification evidence, and disposition. Use existing hashes where appropriate; avoid repeatedly hashing unrelated large parents.

Before removal, confirm:

1. Each path is an explicitly listed, inactive experiment-owned product inside the resolved cleanup scope; it has not changed since inventory.
2. Its necessary scientific evidence and exact generation recipe survive, and all retained dependencies are accessible.
3. The relevant regeneration proof or verified duplicate identity is sufficient, and downstream retained reports/reproduction routes will not silently depend on the removed copy.

Original authoritative inputs are protected. A redundant experiment-local copy of an input may be removed only when its authoritative counterpart is byte-verified, persistently accessible, and correctly referenced by the retained reproduction route. Unknowns remain exceptions.

Use a dry run and the existing independent-review process to check this manifest and its regeneration evidence before applying it. Then **execute the verified removals without another routine owner-approval loop**. Do not delete active products, cross symlink targets, sweep directories by age/glob, run broad `git clean`, prune Git history, or touch unrelated repositories, observations, remote archives, or backups. Partial verified cleanup is preferable to unsafe completion.

Preserve original manifests, seals, assessments, and original hashes unchanged as historical records. Add a compaction record that explicitly identifies retained, deduplicated, and intentionally pruned/regenerable payloads. Seal the compact archive as a new retained representation; do not claim the original full-payload seal still verifies an intentionally incomplete directory.

Update the current archive index and affected readers/reproduction commands narrowly so they distinguish a retained artifact, an intentional regeneration requirement, and a genuinely missing/corrupt file. Do not silently regenerate large products while reading a report. Preserve historical provenance rather than rewriting old outcomes.

Verify retained-product integrity, report accessibility, and dependency references after cleanup. Report both removed logical bytes and observed storage reduction, including limitations from filesystem sharing or snapshots. Keep no hidden replacement copy that negates the claimed saving.

## 7. Deliverables and stopping point

Create one compact closeout bundle, for example:

```text
successor-point-152389-importance-and-archive-compaction-20260926/
```

Deliver:

- **One owner-facing assessment:** the noise-relative importance table, what is parked/consequential/unresolved, the practical interpretation, before/after storage totals, and any exceptions.
- **A reproduction and retention record:** inventory/action manifest, retained dependency list, exact commands and numerical checks, compaction provenance, and measured storage accounting.
- **Focused implementation evidence:** tests of units/denominators/applicability labels, protected-path and changed-file rejection, retained dependency closure, report access after pruning, and the required independent review. Reuse unaffected scientific validation; do not start a new broad test campaign.

Record actual assessment calculations, reused evidence, regeneration applications/trials, removals, and blocked items. Report a missed storage target honestly with named reasons. Do not spend unlimited computation or discard irreplaceable inputs to reach 25 GB.

Make the future retention convention explicit in the existing local workflow: compact evidence is durable; large intermediate arrays are temporary unless tied to a named use; regeneration is an explicit operation. Do not build a general archival platform, scheduler, or automatic deletion daemon.

Update local goals/status/handoff records and commit reviewed code/documentation changes using the established owned-worktree practice. No merge, push, production/default change, or new external storage service.

**Stop after the importance assessment, bounded regeneration checks, verified compaction, review, and closeout.** Recommend at most one next scientific action only if the retained evidence establishes its practical value; recommending no further investigation of the small effects is a valid outcome. No new observation campaign, pixel-removal experiment, parameter tuning, covariance adaptation, injection family, JINC work, applied m10, or cycle 11 is authorized.
