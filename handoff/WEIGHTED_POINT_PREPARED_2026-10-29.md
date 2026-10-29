# Weighted POINT: two local trials ready for Grant

The full agreed suite is implemented and locally checked: observation-local optical
response rejection with configurable default 10%, inverse-noise cleaning and map
weights, fresh original-CAL FRUIT passes, and detector diagnostics. **No reductions
were launched.** Grant alone runs the commands below. Existing maps, inputs, APTs,
reference directories and frozen runtimes remain preserved.

## Launch

152389:

```bash
bash /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weighted-152389-20261029/02_redu.sh --run
```

152418:

```bash
bash /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weighted-152418-20261029/02_redu.sh --run
```

Omit `--run` for a check-only pass. The scripts check configuration and frozen
software identities, then use the existing TolTECA preparation/launch layer and
TolProj executable snapshot/node-local launcher. They contain no output cleanup.
Each invocation reserves the next unused `reduNN`; the first successful outputs
will be in each trial's `reduced/redu00/<observation>/raw` directory.

The two workdirs were created through the established local TolTECA setup and
TolProj runtime APIs, copying the appropriate prior numbered configuration into
fresh locations. Both final merged configurations and three-worker resource
contracts pass check-only validation. Neither new `reduced` directory existed
at preparation handoff.

## Settings and required evidence

Both trials use a1100/a1400/a2000, ranks 3/4/6, 2 arcsec pixels, coverage cut 0.15,
three upstream workers, existing RTC filters and disturbance exclusion, quiet
output, and no optional retained binary transports. The relative optical-response
cut is 0.10, editable through `successor.detector_response.min_relative_response`
in the normal v2 YAML; changing a frozen trial requires normal revalidation and
snapshot preparation. FRUIT requests ten feedback applications after M0, with
new noise/basis learning each time and exact source restoration. This fixed
budget is not a convergence criterion. Source-guard failure stops feedback with
an explicit reason and preserves the last complete all-array maps.

Each `reduNN` contains `RUN.json`, `detector-drops.json` for the established
disturbance rule, `detector-health.json` for optical eligibility, and
`weighted-point.json` for the applied feedback trajectory and final contributors.
Native evidence and intermediate maps/models are under `.point-work`.
Normal weight ranges and unavailable-weight exceptions are retained without
large per-sample cycle dumps. All three final maps and qualified fit diagnostics
must publish together. The noise weights are empirical analysis coefficients,
not a measured full map covariance or precision.

The optical intervals are **conditional resampled 99% intervals**, with frozen
robust weights/reference scale and block-exchangeability assumptions. They do
not have demonstrated true 99% coverage or an observation-wide false-rejection
guarantee. Every contributing fit block remains in resampling; unsupported draws
produce unresolved uncertainty. Independent review caught and repaired an earlier
implementation that could discard sparse blocks and falsely declare a normal
detector weak. New tests reproduce the defect with the old line and pass with
the corrected `point-optical-heldout-network-block-panel-bootstrap-v2` method.

After Grant reports completion, inspect actual weak/reversed/unresolved losses,
noise availability and final contributors for each array; the source fits and
map backgrounds through the feedback sequence; preserved reference maps; and
elapsed time. Scientific improvement remains unmeasured. No acceptance threshold
was invented or changed to make the candidate pass. PPT calibration-consumer,
SCIENCE/coadd and Unity/Spack release qualification remain later milestones.

## Exact software and configuration identities

Reviewed frozen source: `331ff29ea1720b1454e671cbbab93814d8df8321`.
Tree: `c512f8e68b7dba3255467a9b06d06903aad43185`.
Branch: `codex/configurable-point-20261028`.
Implementation commit: `4f7d31957d2135601145c63387e310622fe16d64`;
statistical repair: `df550415789c5777d31fa37e59989021f4600f50`.
The first implementation candidate was never frozen or run on observations.

Project root:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`.

Runtime: `software/configured-point-weighted-331ff29ea` under that root.
Native version contains `g331ff29ea`; executable SHA256:
`197649397221c6b780605a732c4436ffde334c87faeaaec59e584984448f0852`.
Runtime manifest SHA256:
`7a39242d98e999f3a4c36ed79511b97d35af7b90b745749f9f0ccb069329be01`.
TolProj snapshot/launcher source SHA256:
`a013cac086724b7e778fd7e51ad4b7a91a2ec17ad763f52ab0f3950f237b24d5`.

Each observation has `evidence/weighted-point-<obs>-20261029/READY.json`, containing
every numbered-config and support-file digest, executable/snapshot paths, selected
APT bindings and reference configuration identities. READY digests:

| Observation | READY SHA256 |
| --- | --- |
| 152389 | `28fb64e8a874f16ae773effe504ca73adce7d80c6d2ec7fd6a56b958e4a4439b` |
| 152418 | `c2ad3e74065b8d0c525765690bf7a45b1bf1ef5bb3480a61b39c84e01ac892ee` |

The local data inventory has networks 0–5, 7–9, 11–12 for both observations;
networks 6 and 10 are absent in both supplied input sets. All three arrays are
represented. Existing matched-v2 APT manifests are intentionally shared inputs.
The reference configs and reductions remain in `pointings-configurable-20261028`
(152389 redu00) and `pointings-disturbance-152418-20261028` (152418 redu00/redu01).

Runtime Python is `/Users/gwilson/tolteca/bin/python`. The existing preparation
layer remains `software/preparation-20261028`. Python package contents and native
shared libraries are external dependencies; the native dependency still reports
`kids 04088da-dirty`. This is a local Mac AppleClang/Homebrew candidate, not a
portable deployment or a reproduction of the accepted Spack-backed V2 campaign.
Later documentation commits do not replace this reviewed frozen runtime.

## Validation and review

1,366 enabled CTest tests passed, zero failures; one pre-existing disabled test.
256 configured POINT/package Python tests, 207 baseline-tool tests, full require-all
configuration preflight and scientific-change ledger validation passed. Isolated
header compilation and required CLI build/source-version agreement passed.
Durable logs and their digests are under the local project's
`evidence/weighted-point-checks-20261029/checks.json`.

[Native review](../validation/weighted_point_2026-10-29/FINAL_NATIVE_REVIEW.md) and
[application review](../validation/weighted_point_2026-10-29/FINAL_APPLICATION_REVIEW.md)
pass the scientific/behavioral, architecture/ownership and repository/evidence
categories with explicit local/synthetic limitations. See the
[independent prepared-trial audit](../validation/weighted_point_2026-10-29/PREPARED_TRIAL_REVIEW.md),
[implementation](../validation/weighted_point_2026-10-29/IMPLEMENTATION.md),
[checks and review repair](../validation/weighted_point_2026-10-29/LOCAL_CHECKS.md)
and [work order](../validation/weighted_point_2026-10-29/WORK_ORDER.md).

No push, canonical integration, Unity access, observational processing or production
activation occurred. Actual two-observation validation waits for Grant's shell runs.
