# Output schemas

## `partial_observation_results.ecsv`

One row per opened pointing. It contains exact tau and whole-scan bootstrap
intervals, authenticated map tau, paired differences and covariance, support
census, derivative cross-check, sensitivity flag, held-out model-win counts,
and result digests. `gate_status` distinguishes observation-level passes from
the stopping observation. Rows are not a frozen inferential subset.

## `partial_observation_results.json`

Schema `sci-align-001-lissajous-timestream-partial-results-v1`. It mirrors the
ECSV rows and declares `corpus_inference_permitted=false`.

## `partial_input_identities.json`

Schema `sci-align-001-lissajous-timestream-partial-input-identities-v1`. Each
row binds PTC and PPT paths/digests, selection and protocol digests,
authenticated map-result identity, result digest, and per-observation checksum
manifest digest.

## `partial_stop_summary.json`

Schema `sci-align-001-lissajous-timestream-stop-summary-v1`. It records the
opened and unopened observations, exact stop condition, bootstrap count, KDE
peak count, optimizer-start pile-up count, and prohibition on corpus summary.

## Per-observation local result

Schema `sci-align-001-lissajous-timestream-observation-result-v1` is emitted as
`result.json` beneath the external visualization result root. Its companion
`SHA256SUMS` authenticates compact tables, objective/profile PDF, resumable
bootstrap work array, and result. Large realization arrays are intentionally
not copied into this Git package.

Revision-6 results also retain optimizer attempt, finite-result, converged-
result, and inherited-start fallback fields. A bootstrap checkpoint produced
before revision 6 is not reusable for the ObsNum 136280 rerun.

Instrumented successors add `progress.jsonl` and `run_state.json` without
changing the observation-result schema. Progress records carry UTC and
monotonic elapsed time, stage or fit identity, optimizer disposition, and
bootstrap counts where applicable. `run_state.json` uses schema
`sci-align-001-lissajous-runtime-state-v1`; only `status=complete` is a
completed run. A wall-limit stop publishes no `result.json`.

## Per-observation fit gate

`fit-gate` creates schema `sci-align-001-lissajous-fit-gate-v1` in
`fit_gate.json`, accompanied by `fit_gate_model_results.ecsv`,
`fit_gate_optimizer_audit.ecsv`, `fit_gate_scan_metrics.ecsv`, the exact
`fit_gate_progress.jsonl` slice, and
`lissajous_fit_gate_o<obsnum>.pdf`. `FIT_GATE_SHA256SUMS` authenticates this
immutable first-phase package. Its automatic status tests only structural and
numerical validity; tau is explicitly excluded from acceptance. A failed fit
still publishes a gate package, but cannot be resumed.

`resume-observation --owner-review-approved` reauthenticates that package and
loads its full-data fits without refitting. The source implementation digest is
part of the gate identity.

## Post-gate stage checkpoint

`stage_checkpoint.json` uses schema
`sci-align-001-lissajous-stage-checkpoint-v1`. Its identity binds the fit-gate
document, fit-gate checksum manifest, and source implementation. The associated
`STAGE_CHECKPOINT_SHA256SUMS` covers the state plus each completed stage file:
objective profile, derivative cross-check, held-out model comparison,
sensitivity fits, and network sensitivity. Files are written atomically and
verified before the stage is considered reusable.

## Runtime audit

`audit-runtime` projects any retained `progress.jsonl` into schema
`sci-align-001-lissajous-runtime-audit-v1`, plus stage, optimizer-family, and
optimizer-attempt ECSV tables. It is a read-only performance diagnostic and
does not reconstruct missing historical function-evaluation counts.

## Per-observation fit visualization

`visualize_sci_align_001_lissajous_fit.py` creates a separate checksum-bound
package from a completed result. Its identity, support audit, deterministic
fit-unit selection, exact model/objective validation, tables, PDFs, PNGs, and
copied renderer are covered by the package `SHA256SUMS`. Fit units are
contiguous True blocks of the frozen score mask for one detector and scan.
Retained PTC-weight residuals are explicitly labeled `sqrt(weight)`-scaled,
not standardized sigma residuals.
