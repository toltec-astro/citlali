# POINT152420 APT calibration lineage — 2026-09-30

The scale divergence originates upstream of POINT CAL: historical redu09 selects
an observation-matched descendant of **3c273 beammap152307**; the successor selects
an observation-matched canonical descendant of **3c273 beammap148670**, explicitly
requested in the August28 matching command. The different recorded source fluxes
and detector-response measurements reconstruct the selected factors. No missing
pointing correction, duplicated conversion, or wrong-row application is demonstrated.
Neither absolute scale is accepted by this finding.

For a1100, historical `F_H = 3896.3679050705327 / amp_152307`; successor
`F_S = 3127.2160728457216 / cal_amp_148670`, in mJy/beam per native dimensionless xs.
The source-flux ratio is **0.8025977395**. For the same target acquisition/Tune
channels, the associated successor calibration amplitudes have medians about
1.80–2.00 times their historical associated amplitudes. Thus the factor ratios
are around0.40–0.45, with substantial detector variation. This comparison follows
each target's recorded associations; it does not assert equal source-channel
numbers identify the same physical detector in different beammaps.

| Network | Common valid target rows | F_S/F_H median | Median cal_amp_S/amp_H | Median cal_amp_S/amp_S |
|---|---:|---:|---:|---:|
|1|310|0.4205351616|1.90851663|0.96111913|
|2|327|0.4464489147|1.79773702|0.97159087|
|3|360|0.4471234313|1.79502550|0.99221141|
|4|345|0.4003555273|2.00471252|0.99122916|

All1,342 common valid rows use successor empirical-template method1. Its difference
from that beammap's own Gaussian amplitude is only a few percent at the median;
it does **not** alone explain the factor-of-two contrast. Source observation,
response measurement and beammap reduction differ together. No causal allocation
among atmosphere, detector response, association quality and beammap processing
is established by comparing these two fitted-amplitude columns.

The reconstructed ratio's maximum relative residual is **6.661338147750939e-16**.
Successor source-flux/cal_amp reproduces selected factors exactly; historical
source-flux/amp agrees within7e-16 relative (legacy decimal serialization).
The independent audit uses8 binary64 epsilons, justified by serialization and
arithmetic rounding. Parent→selected factors are required to match exactly.
Twenty previously saved representative samples trace both source rows, both
factors, native x and actual CAL; the target-atmosphere term remains separately
inferred from the earlier actual-CAL audit, not reimplemented as a new oracle.
Full distributions, including network0 and outliers, are in
[results/apt-lineage-audit.json](results/apt-lineage-audit.json).

## Side-by-side chain and evidence

Path abbreviations below are exact local roots:

- `H=/Users/gwilson/work_toltec/local_data/2025-C1-COM-01`
- `V=/Users/gwilson/work_toltec/local_data/citlali-validation/v2`
- `L=/Users/gwilson/work_toltec/local_data/_apt_library`
- `R=/Users/gwilson/work_toltec/local_data/citlali-validation/successor-rtc-point-transfer-completion-20260930`
- `W=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor`

| Boundary | Historical redu09 | Successor B |
|---|---|---|
| Observation / role |152420/0/2, Tune152420/0/1,1146+399 POINT. Ordinary earlier-state pointing input; historical fluxscale_report.json produces science children152390/392/419/431/433, not152420. Cohort project file and original matching invocation not recovered locally.|Same acquisition/Tune; named bright-POINT development reference for RTC/PTC/feedback response. TolProj match report records pointing cohort1146+399:pt_none_none_obs_152420. This experiment does not issue calibration evidence or independently validate absolute flux. `W/validation/rtc_point_transfer_2026-09-28/inputs/reference.json`, `R/B-REF/freeze/prospective.json`.|
| Source calibration |`L/raw/3c273/152307/apt_commissioning_beammap_152307_citlali.ecsv`, SHA2560f7bb194dc8a74046b70e0f738eb3b4d5f5f47e9b73d462c82a5b95afd640e0c; source dateFeb18, issuedMarch1. Library manifest identifies original remote redu00, copiedMarch1. Source metadata records3896.3679050705327mJy and Gaussian amp in xs.|`V/beammaps/3c273/reduced/redu00/148670/raw/apt_commissioning_beammap_148670_citlali.apt-v2/manifest.ecsv`; baseline semantic944a82d92231408d416012701d41a670b02b4e18c9fb59a4decb64d867641f51, occurrence2f407f35…; source observationJan13, producerAug28/c31a60a0. Parent-level beammap_provenance.yaml records TolProj flux authority3127.2160728457216mJy and phase-aware measurement iterations. At c31a60a0, beammap_flux_conversion_impl.h explicitly uses positive cal_amp, otherwise Gaussian amp, then flux/amp.|
| Matching / transformations |`H/apts/apt_152420_matched.ecsv` records det_id_right; all5,031 matched rows agree exactly with the library source's UID,nw,amp,flxscale,f_out,x_t,y_t.487 lack recorded matches. `H/apts/hero/hero_apply_summary.yaml` and hero audit apply geometry/flags; all5,518 flxscale values remain unchanged. No pointing-flux correction embodied.|`V/science/apts/v2/requests/apt_152420_matched_match_request.json`, relation and baseline snapshot:4,806 matches,712 unmatched,0 ambiguous,0 exceptions; flxscale rule copy-seed-or-null. All declared matched factors copy exact source_uid factors. Producer module hash62e3ec… and matcher hash970159… both match TolProj5fc334cd3d9313106246b0de68b0f7a07ab86909 source bytes. Target KMP fields are preserved; no fluxcal operation.|
| Requested / selected |`H/pointings/02_redu.sh` invokes TolTECA reduce; numbered72_reduce.yaml explicitly selects hero APT for152420. `H/pointings/reduced/redu09/citlali_o152389_0_2_c8.yaml` confirms effective selection. HST apt_flxscale snapshot equals selected table. Later73_reduce_hero_v2_152420.yaml names another directory and is **not** redu09's input.|`V/science/logs/matchapts-v2.log` line6 explicitly supplies baseline148670 and the actual Python/module/config command; job63731641 records issuance; mainlog line15 records reread. Selected product is observation-matched, profilecitlali-observation-matched-apt-v2, semantic7ba4b56addd3e6ce5221682b3789e899f936b4a479ac291fb16b609ac2bb2768, root transport3861c9d10e70b04dacac4d35cb74cbf2b782fb3a77758d5f9315ccfdf59a5d3c. Named POINT config line58 explicitly carries it through point_pipeline_inputs.py to R/B-REF/inputs/effective-point.yaml and network inputs. No automatic nearest-beammap selection here.|
| Runtime / multiplication |Historical consumerffc6b907, July19 log; legacy ECSV Calib route. Actual stats snapshot confirms consumed factors; mJy/beam fcf=1, extinction disabled; x×F_H. Original loaded TolTECA Python module bytes unavailable, so no exact Python build claim.|Source518e05891cf8eb2e84621cdcfcea83525f3d1c49; R/BINARY.json and B-REF/upstream/RUN.json bind CLI SHA1d08e7fe… and command. prospective.json records /Users/gwilson/tolteca/bin/python; repository run_campaign prepends W/tools/timestream_successor, imports point_transfer_experiment/point_pipeline_inputs rather than installed TolProj selection. Native verified relation maps network/channel→output_uid, validates ordinary dimensionless gainlintrend x and flxscale units. Selected F once × target atmosphere (~1.0333) × x; prior audit verifies all1,153 available detector streams against saved CAL.|
| Safeguards actually implicated |Explicit legacy ECSV compatibility route predates August v2. No canonical receipt/ancestry guard governed this July run. Source-row recovery now supports lineage, but cannot retroactively certify original matcher ambiguity handling or module identity.|Executed native route invokes verify_bundle_filesystem(manifest,true) and admit_canonical_apt_detector_relation_v2, including shared observation owner path. Raw/Tune identity/digest, units, keyed rows, availability and exact parent-copy checks apply. Relevant reader code unchanged from executed518e05891. TolProj setup/fluxcal is not invoked by this harness, but native canonical admission is. Fresh read-only validation of actual matched and baseline bundles succeeds; this corroborates current integrity and is distinct from original execution evidence.|

The historical library points to a March1 product at remote
`/work/toltec/wilson/citlali_testing/beammaps/3c273/reduced/redu00/152307/raw/apt_commissioning_beammap_152307_citlali.ecsv`.
Present local redu00 config/log refer to a June4 rerun (compressed log truncated);
local redu01, new_reductions and152307_redu50 APT hashes differ. They cannot be
substituted for the March producer. The retained library source itself is recovered
and all recorded associations/factors reconcile; original producer executable,
matcher invocation and original automatic-versus-explicit historical seed-choice
rationale remain specific unavailable links. No remote access was used.

## Workflow legitimacy and safeguards

Verified August24 implementation leads: Citlali07b60205b55f2782aabef583f08f4771506b6a6f
and TolProj2fe2af8fae681608893a0b491bd4b3e0b340ba03. Current accepted specification
is doc/CANONICAL_APT_V2.md; Stage7 status records matched→fluxcal issuance and
SCIENCE152390 validation. That is capability/acceptance evidence, not proof that
POINT152420 used a fluxcal child.

TolProj setup_pointing_reductions validates ordinary matched manifests.
calibrate_flxscale._science_cohorts selects science/mixed roles with science
observations and brackets; it derives corrections from pointing products and
issues children for those **science** observations. Science setup chooses the
calibrated binding after flxscale_done; matched inputs are also permitted before
that state. Both POINT inputs here are appropriate earlier-state inputs for their
measurement/development role. Applying a correction derived from this same POINT
back to it cannot establish independent absolute recovery.

Schema v2, unique artifact identity, observation matching, and flux-calibrated
state are separate facts. `selected_child_APT` in CAL receipts is a generic selected
artifact label, not proof of the fluxcal profile. Existing parent pointers already
provide the required distinction; no second calibration registry is introduced.

The guard checks conformance to the **declared** baseline. It does not choose the
best astrophysical calibration or require different valid baselines to have equal
factors. Explicit selection of148670 rather than152307 is therefore accepted
variation at that boundary. Hashes do not establish that148670 is scientifically
preferable for the February target. No recovered record provides that justification.
This is an open scientific-selection question, not evidence that a required
pointing correction was omitted.

The current V/science/project.yaml lacks the152420 apt_v2 binding despite immutable
issuance evidence. TolProj's later documented reconcile-apts-v2 workflow addresses
lost concurrent project bindings; the historical cause is not proved here. Its
legacy apt_seed remains unresolved by design because match-apts-v2 uses explicit
baseline input. Neither field resolved this successor run: its named manifest is
explicit. No authoritative project file was edited and this anomaly is not assigned
as the cause of the factor ratio.

## Repair, coverage, affected claims, and next discriminator

No production calibration algorithm or state-selection change is justified by
these results. Added a focused canonical integration regression in
`tests/test_canonical_apt_v2.cpp`: two independently valid parents with different
factors both publish/reread/admit as matched POINT relations; stale factors from
the first parent cannot be issued while declaring the second. Existing keyed
presentation-permutation and invalid-join tests also pass. New fixture development
initially exposed fixture-only field sorting/mandatory-flag omissions; both were
corrected before acceptance. No observed pipeline defect was hidden by relaxing a
check. New audit and clarified handoff are the implemented changes.

Validation: local CLI and test builds;12 focused tests;1,321 enabled CTests pass,
0 failures (one pre-existing MapFitterLifecycle test disabled);207 baseline tests;
full require-all config gate; both actual canonical bundles validate. These are
local AppleClang/Homebrew development results, not Unity Spack reproduction.
Compact gate evidence is results/apt-lineage-gates.json. The independent milestone
review is continued for this substantive result, separate from earlier prelaunch
review; see APT_LINEAGE_REVIEW.md when recorded.

All eight prior B/Q roles and their response controls depend on the same selected
148670-derived APT and its geometry; their conditional incremental-response claims
remain conditional on that lineage. They never established absolute calibration.
B/Q parking and saved map/feedback arithmetic remain supported. Earlier supplied
POINT152420 comparisons using this same explicit manifest share this limitation;
no other observation is reclassified or rerun. Calibration reconstruction alone
does not allocate the291.81 versus574.59mJy peak, fitted-area ratio1.61149, feedback,
network0/population, geometry, noise weighting, or profile differences.

The unstarted three-role M0 batch is **deferred**, original FREEZE/READINESS/review
unchanged, STATUS/DEFERRAL record the new ordering; **zero of six planned plus two
replacement starts used**. Old launch command is not ready against this source.
The existing common-current versus common-historical frozen-operator contrast,
with native replay first, remains the smallest prepared discriminator of the
*conditional map consequence* of the explained heterogeneous factors. It cannot
choose the scientifically correct baseline. It requires a new reviewed freeze
and current resource accounting before owner foreground launch; the preserved
old freeze must not be rewritten. No map run is necessary to establish the lineage
result. Absolute calibration and the unexplained choice/appropriateness of148670
remain explicitly open; neither agreement with574.59 nor a detector median can
resolve them.
