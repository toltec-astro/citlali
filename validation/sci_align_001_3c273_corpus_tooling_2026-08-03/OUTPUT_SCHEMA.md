# SCI-ALIGN-001 3C273 corpus output contract

## General encoding and identity

All scientific JSON is UTF-8, sorted-key, two-space-indented JSON with a final
newline.  Scientific CSV uses a header, deterministic column order, sorted
rows, Unix newlines, and an empty field for a value that is unavailable by
contract.  JSON uses `null` for the same condition.  Non-finite floating-point
values are never serialized as `NaN` or infinity.

Every published directory contains `SHA256SUMS`.  Resume reuse is permitted
only when the candidate identity, selected-manifest digest, frozen-protocol
digest, tool digests, input identities/digests, and package checksums agree.
Source paths are recorded but source products are never copied or modified.
`candidate_manifest.example.json` is schema-shaped documentation with
deliberate placeholder paths and digests; it is not an executable manifest.

The stable observation identity is `observation_number`.  `candidate_id`
identifies one reduction/provenance realization and is not an independent
observation.  `network_id` is the integer TolTEC/ROACH network identity.  UID,
not detector row, is the detector identity.

## Stage 1: inventory

| File | Row grain or purpose |
| --- | --- |
| `candidate_inventory.csv` | one discovered reduction candidate |
| `candidate_inventory.json` | schema, roots, source policy, sorted candidate rows, and digest metadata |
| `candidate_table.md` | owner-readable rendering of all candidates |
| `selection_template.csv` | all candidate rows plus explicit owner selection/note fields |
| `duplicate_reduction_registry.csv` | one duplicate group/candidate membership row; no silent winner |
| `exclusion_registry.csv` | one candidate/reason row for every failed frozen eligibility check |
| `next_commands.txt` / `next_commands.json` | exact follow-on commands using the emitted paths |
| `digest_cache.json` | persistent path/size/mtime/digest facts used to avoid rehashing large inputs |

Required candidate facts include exact and normalized source identity,
observation number, observing date/time when authoritative, reduction/project
path, realized config identity and digest, software SHA when available,
detector-TOD/telescope/scan-registry availability, raw paths, raw timestamp
and counter-field availability, network coverage, and explicit core/enhanced
eligibility with exclusion reasons.  Raw header inspection records
`Header.Toltec.RoachIndex`, `FpgaFreq`, `AccumLen`, exact integer `T0` from
`Data.Toltec.Ts[:,0]`, and preserves column 5 separately.  A complete ordered
network-to-integer-`T0` vector and its digest are candidate session metadata;
the nanosecond field is not folded into that identity.

The frozen selection directory contains `selected_manifest.csv`,
`selected_manifest.json`, the owner selection file/digest, and `SHA256SUMS`.
The selected manifest preserves all input identities and contains exactly one
primary reduction per core-eligible observation. Every other core-eligible
reduction of that observation is included with `analysis_role=sensitivity`
and inherits the primary observation's held-out fold.

## Stage 2: one Beammap

Each candidate has a separate directory named by a sanitized `candidate_id`.
The runner publishes every completed file by atomic replacement and binds the
directory for safe resume; the directory itself is not renamed atomically.

| File | Row grain or purpose |
| --- | --- |
| `map_result.json` | complete compact map result, authority statements, model registry, controls, and scope |
| `map_summary.csv` / `map_summary.json` | one candidate result with explicit primary/sensitivity role |
| `network_map_results.csv` | one expected network per map, including explicit missing-network rows |
| `timing_phase_results.csv` | one timing model/group estimate |
| `fit_controls.csv` / `fit_controls.json` | independently frozen detector and direction fit acceptance evidence |
| `scan_registry.csv` | one realized stable scan window and trajectory-derived direction classification |
| `raw_counter_transitions.csv` | one delivered PPS-counter transition per network |
| `raw_phase_summary.csv` | one raw network mapping, phase, and counter summary or no rows for core-only analysis |
| `input_manifest.csv` / `input_manifest.json` | exact retained inputs and digests used by the candidate |
| `run.log` | deterministic diagnostic messages and any explicit enhanced-to-core fallback |
| `enhanced_failure.json` | fail-closed raw-linkage/analysis error when a valid core result is retained |
| `resume_binding.json` | exact manifest/protocol/tool/input digest binding |
| `SHA256SUMS` | all files in the candidate directory except itself |

The primary map fields include pooled timing estimate and scan-jackknife
uncertainty, left/right and excluded scan counts, matched detector and network
counts, scan-speed quantiles, equivalent parallel/perpendicular sky
displacement, FWHM fraction, beam major/minor widths, amplitudes, first- and
second-half estimates, and every preregistered model comparison.  Half changes
are labeled within-observation timing variation, not clock drift.

Each available network row includes the primary assigned-slot estimate and
uncertainty, detector/scan counts, sky and beam controls, all comparison-model
estimates, native-to-assigned-slot summaries, native detector-frame phase,
integer `T0`, and raw-counter anomaly counts where enhanced linkage succeeds.
The map result contains separate within-map regressions against native phase
and native-to-assigned-slot residual.

`raw_counter_transitions.csv` records zero-based delivered transition row,
PPS counts before/after, unsigned-32 internal-clock and PPS-time values
before/after, packet counts before/after, transition spacing, paired
PPS-count/PPS-time row offset when it is provable, and phase geometry.  Every
row states that metadata-to-integration association is unproved.  The network
summary reports the 122/123 spacing test, exact 128-PPS/15,625-row repeat,
modulo-`2^32` clock-step check, packet-step check, PPS-time increment check,
same/adjacent/variable transition association, and beginning/middle/end phase.

Enhanced failure never creates a raw reassociation claim.  When the retained
core products remain valid, the runner retains the core result and records the
enhanced failure explicitly.

## Stage 3: aggregation

The aggregation run consumes only the compact selected manifest, frozen split
protocol, and per-map outputs.

| File | Row grain or purpose |
| --- | --- |
| `map_summary.csv` | one independent primary map; duplicates are marked sensitivity-only |
| `network_map_results.csv` | one map/network result |
| `timing_phase_results.csv` | compact concatenation of retained per-map phase/linkage summaries |
| `fit_controls.csv` / `fit_controls.json` | concatenated acceptance diagnostics and counts |
| `exclusion_registry.csv` | inventory, execution, or aggregate exclusion with frozen reason |
| `duplicate_reduction_registry.csv` | duplicate sensitivity membership and canonical authority |
| `duplicate_reduction_sensitivity.csv` | deterministic primary-versus-sensitivity comparison by observation/network |
| `session_registry.csv` | frozen held-out group assignment and grouping authority |
| `candidate_model_results.csv` | M0--M4 fit, prediction, scatter, and support statistics; M5 is the no-model outcome |
| `heldout_predictions.csv` | one model/held-out map or map-network prediction and error |
| `variance_components.csv` | covariance-aware between-map, network, and interaction components |
| `network_repeatability.csv` | persistent-network repeatability summaries |
| `slot_regression_results.csv` | within-map, session, and corpus native-phase/slot predictor fits |
| `drift_results.csv` | group-aware within-observation timing-variation summaries |
| `corpus_summary.json` | selected category, limitations, producer authority, and no-correction scope |
| `REPORT.md` | owner-readable result, sky/FWHM scales, exclusions, and decision rationale |
| `input_digests.csv` / `input_digests.json` | verified compact-input, manifest, protocol, and tooling identities |
| `plots/` | deterministic compact diagnostic plots only |
| `SHA256SUMS` | aggregate package checksums |

The freeze subcommand creates a separate checksum-bound
`frozen_analysis_protocol.json` and `session_registry.csv` before reading
timing results.  Session precedence
is complete integer-`T0` vector, other provenance initialization identity,
date group, then deterministic observation-number fallback.  At least three
independent groups are required for held-out claims; exactly three permits
reporting all preregistered models but prohibits data-driven model selection.

Candidate model statistics include weighted held-out timing error, sky error,
beam-FWHM fraction, between-map intrinsic scatter, persistent network scatter,
network-by-map interaction, within-map and corpus regressions for both native
phase and native-to-assigned-slot residual, the free slot coefficient and its
interval relative to `-1`, session support, and within-observation variation.
Unsupported levels and missing predictors remain explicit.

## Scope fields required in every final result

Every `map_result.json`, `corpus_summary.json`, and `REPORT.md` states:

- no Citlali reduction was launched;
- no source product was modified;
- Codex did not contact Unity;
- raw row reassociation is not claimed;
- physical timestamp event semantics and upstream FPGA association are
  unresolved;
- arbitrary millisecond NTP error and differential oscillator drift are
  strongly disfavored by the producer account;
- distinct stable per-network integration phase remains possible; and
- no production correction is authorized.
