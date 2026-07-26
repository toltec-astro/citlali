# Fruit-loop calibration-reference investigation

Date: 2026-07-26

Status: existing evidence analyzed; additional Unity reductions proposed but
not requested

Population extension: the subsequent
[108-observation quality-stratified plan](FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md)
supersedes the small R0 real-observation subset below. The controlled
amplitude, position, and science-transfer experiments remain applicable.

## Outcome

The iteration-18 checkpoint-v2 result is a stable point-source-response
plateau in amplitude, shape, and centroid, but it is not demonstrably converged
under the strict all-diagnostic 1%, 2%, or 5% candidate rules. The last two
successive whole-map changes include 6.50% in a1400. All arrays pass all tested
diagnostics only at 10%. No production stopping rule or attenuation correction
is justified by this one observation, one source position, and one injected
amplitude.

The calibration-reference verdicts are deliberately separate:

| Use | Verdict for current processed pointing products | Evidence and boundary |
|---|---|---|
| Astrometric pointing offset | **Qualified yes only at a demonstrated stable endpoint; not yet for every five-iteration map** | Injected-source centroids at iteration 18 differ from the realized kernels by 0.010, 0.012, and 0.035 arcsec. The largest final-step centroid change among the five real iteration-4 pointings is 0.082 arcsec, but cumulative seed-to-iteration-4 displacement reaches 0.494 arcsec. The iteration-9 obsnum 133410 changes are below 0.001 arcsec. This supports stable *relative* offset extraction from a well-detected source only after the centroid stopping gate passes. It does not independently establish the absolute WCS/source-coordinate zero point. |
| Effective processed PSF | **Qualified yes, after a demonstrated stopping point** | At iteration 18, injected-minus-control major/minor FWHM ratios to the realized kernel are within 2.1% of unity. This is evidence for the effective processed point-source core under the same pointing configuration. It is not a physical telescope-beam measurement, and several five-iteration real maps still change by 2–4% in FWHM on their final step. |
| Photometric amplitude or universal transfer correction | **No** | Kernel-normalized recovery at iteration 18 is 0.9583, 0.9486, and 0.9825, leaving measured attenuation of 4.17%, 5.14%, and 1.75%. The endpoint is tested at one amplitude, one map position, and one observation. The remaining attenuation has no approved cause or correction model. |
| Prediction of associated science-observation response | **Not determined; current products are insufficient** | No associated science observation/configuration was recorded in the local project, and the injected-source seam is currently accepted only for pointing/OOF reductions. The decisive pointing-versus-science paired injection has therefore not been measured. |

These verdicts do not change production defaults.

## Evidence set and reproducibility

The machine-readable evidence package is
`validation/fruit_loop_calibration_reference_2026-07-26/`. Its manifest
records product counts, retained executable hashes, the exact-restart result,
and the hash of the archived TolAPT metrics.

The inventory covers:

- five real-source sequences (obsnums 133410, 144176, 148434, 151718, and
  153481), each through absolute iteration 4;
- all obsnum 133410 feedback ablations, including the ten-iteration unchanged
  policy and the intentional no-op `snr_only_s200` result;
- the quarantined checkpoint-v1 injected pair;
- the checkpoint-v2 uninterrupted reference, exact-gated five-iteration pair,
  and exact-gated extension through absolute iteration 18;
- low-level-config hashes, Citlali versions, checkpoint schemas, retained
  executable SHA256 values, source/time/weather metadata, and matched APTs.

The archived 75-row TolAPT `iteration_metrics.csv` was regenerated from the
downloaded products and matched byte for byte (SHA256
`3f6ce63535d0fabff7b82610780ac05c90ecdf4babf22f1f7a883c5b13a1bbc3`).
The extended checkpoint-v2 pair was independently regenerated as 30
array/iteration rows; the pair comparator first proved exact equality of
signal, kernel, and weight maps between the uninterrupted and restarted
controls at absolute iteration 9.

`real_iteration_metrics.csv` reports, for every retained real observation,
array, and iteration:

- fitted and kernel-normalized amplitude;
- major/minor fitted and kernel FWHM;
- fitted centroid and S/N;
- mean/median positive map weights;
- whole-map RMS and a robust 40–100 arcsec background estimate;
- successive whole-map absolute and relative RMS;
- seed ratios, step changes, and feedback-support diagnostics.

`injected_source_iteration_metrics.csv` additionally reports raw amplitude
recovery, realized-kernel-normalized recovery, full-map kernel projection,
transfer/kernel FWHM, centroid separation, pair kernel/weight equality, and
ordinary control/injected pointing-fit values.

The analysis is reproduced with:

```bash
MPLCONFIGDIR=/tmp/citlali-fruitloop-mpl \
  $HOME/tolteca/bin/python \
  tools/fruit_loops/analyze_calibration_reference.py \
  --output validation/fruit_loop_calibration_reference_2026-07-26 \
  --legacy-metrics \
    ../tolapt/outputs/hero-pointing-comparison/fruitloops5-rc1-convergence/analysis/iteration_metrics.csv \
  --legacy-reproduction /path/to/regenerated/iteration_metrics.csv
```

The legacy reproduction is made with
`../tolapt/scripts/analyze_fruitloop_convergence.py` against the same local
project root. The calibration analyzer refuses a supplied legacy reproduction
unless it is byte-identical to the archived file.

## Observation and flux authority audit

| Obsnum | Source | UTC date | Tau | Matched APT source | Local APT status |
|---:|---|---|---:|---|---|
| 133410 | 3C273 | 2025-04-11 | 0.041 | 3C273, obsnum 133411 | source-matched, contemporaneous reference |
| 144176 | Neptune | 2025-10-22 | 0.007 | 3C273, obsnum 148670 | wrong source for planetary truth |
| 148434 | Uranus | 2026-01-10 | 0.037 | 3C273, obsnum 148670 | wrong source for planetary truth |
| 151718 | 3C273 | 2026-02-12 | 0.101 | 3C273, obsnum 151719 | source-matched, contemporaneous reference |
| 153481 | 3C279 | 2026-03-04 | 0.062 | 3C279, obsnum 153142 | source-matched, four days earlier |

The repeated Beammap config values `[2329, 2800, 1923] mJy/beam` are config
inputs, not authoritative truth for these pointing maps, and are explicitly
marked non-authoritative in the inventory.

For 3C273 and 3C279, the external authority should be a date- and
frequency-matched fit from the
[ALMA Calibrator Source Catalogue](https://almascience.nrao.edu/alma-data/calibrator-catalogue)
and its
[flux service](https://almascience.nrao.edu/documents-and-tools/latest/flux-service-of-the-alma-source-catalogue),
integrated over the TolTEC bandpasses with uncertainty propagation. For
Neptune and Uranus, the authority should be an epoch-specific ephemeris and
bandpass-integrated planetary brightness model; the
[CASA `setjy` documentation](https://casadocs.readthedocs.io/en/stable/api/tt/casatasks.imaging.setjy.html)
identifies Butler-JPL-Horizons 2012 models as the time/frequency-aware standard
for solar-system objects.

No exact external catalogue query and TolTEC bandpass integration is archived
with these products. A source-matched APT is useful as a contemporaneous
calibrated reference, but it is neither the known injected truth nor an
independent absolute-flux validation. The planetary rows cannot use their
matched APTs as source truth because the APT source is 3C273.

## Convergence trajectory

No exponential form was assumed. The measured transfer sequence is reported
directly, including increment signs, contraction, endpoint span, and
threshold-window tests.

At absolute iteration 18:

| Array | Kernel-normalized amplitude recovery | Full-map kernel projection | Major FWHM/kernel | Minor FWHM/kernel | Centroid error | Successive map relative RMS |
|---|---:|---:|---:|---:|---:|---:|
| a1100 | 0.958322 | 0.956928 | 1.001471 | 1.003304 | 0.0105 arcsec | 0.9576% |
| a1400 | 0.948569 | 0.955556 | 1.013845 | 1.020091 | 0.0124 arcsec | 1.6884% |
| a2000 | 0.982536 | 0.986626 | 1.008536 | 1.006730 | 0.0351 arcsec | 1.0987% |

Recovery increases monotonically across iterations 9–18. The last amplitude
increments are only 0.031%, 0.123%, and 0.018%; their magnitudes are
0.07–0.49% of the first measured increments. The last-three-iteration
amplitude spans are 0.079%, 0.341%, and 0.055%. This is strong plateau
evidence for amplitude and shape, despite a non-zero asymptotic attenuation.

It is not sufficient evidence for strict whole-product convergence:

| Candidate tolerance | a1100 | a1400 | a2000 | All arrays |
|---:|---|---|---|---|
| 1% | fail (map) | fail (map) | fail (map) | fail |
| 2% | pass | fail (map) | pass | fail |
| 5% | pass | fail (map) | pass | fail |
| 10% | pass | pass | pass | pass |

This table uses the last two transitions, 16→17 and 17→18, and requires
amplitude, both FWHM axes, ordinary injected-map S/N, and whole-map relative
RMS below the stated tolerance, plus centroid error below 0.1 arcsec. The a1400
map changes are 6.50% and 1.69%; the endpoint alone must not erase the prior
failed transition.

The real maps reinforce the distinction. At iteration 4 the five-observation
sequences still include final amplitude changes up to 9.4%, FWHM changes up to
4.35%, and whole-map changes up to 16.8%, while centroid changes remain below
0.082 arcsec. The obsnum 133410 ten-iteration real sequence is much closer:
the final kernel-normalized amplitude changes are −0.043%, −0.005%, and
+0.006%, and FWHM changes are below 0.46%, but whole-map changes remain
1.04–1.69%.

Ordinary fitted S/N is not a monotone convergence objective: from iteration 9
to 18 it decreases by 14.2%, 16.3%, and 8.1% in the injected branch, while the
corresponding control decreases by 3.1%, 1.9%, and 1.2%. The final two S/N
steps are below 1%, but the cumulative loss is scientifically material enough
to retain as a separate guard.

## Candidate stopping policy

Do not adopt a production stopping policy yet. For the diagnostic continuation
only, evaluate after short blocks and call the endpoint stable at a
project-owner-selected tolerance `epsilon` only when all of the following hold
for every array across at least two successive transitions:

1. absolute kernel-normalized amplitude change is below `epsilon`;
2. absolute major and minor FWHM changes are below `epsilon`;
3. centroid displacement is below an approved pointing threshold (0.1 arcsec
   is the current test value, not an adopted requirement);
4. successive transfer-map relative RMS is below `epsilon` on a frozen,
   documented support mask as well as on the whole map;
5. fitted S/N step and two-transition slope show no scientifically material
   continuing degradation;
6. all required products exist, the exact-restart gate passes, and logs contain
   zero unexpected error-level messages.

Use `epsilon` = 1%, 2%, 5%, and 10% in the report, regardless of which value is
eventually selected. Continue by three iterations when the rule fails but the
trajectory still changes materially. Stop as “measured plateau but criterion
not met” when amplitude and shape increments contract for two blocks while a
different diagnostic remains systematically above tolerance. Always retain a
predeclared maximum-iteration safety cap.

The missing generality across amplitude, position, conditions, and science
processing is why this remains a diagnostic recommendation rather than a
production default.

## Minimum additional Unity matrix

The machine-readable version is `planned_unity_run_matrix.csv`. “Jobs” below
counts Citlali invocations, not scheduler wrappers.

| ID | Question resolved | Minimum experiment | Jobs | Readiness |
|---|---|---|---:|---|
| C0 | Is iteration 18 truly stable at the selected tolerance? | Continue the existing exact-gated obsnum 133410 control/injected checkpoint-v2 lineages in three-iteration blocks. | 2/block | Ready after tolerance selection |
| L0 | Is response linear? | From the same obsnum 133410 iteration-8 v2 checkpoint, run exact pairs at 0.1× and 3× the existing per-array truth. Existing 1× is the midpoint. | 4 | Ready |
| P0 | Is response position dependent? | One exact 1× pair at a frozen off-center position inside the well-covered region. | 2 | Blocked on diagnostic design approval |
| R0 | Does response generalize across conditions? | Fresh ten-iteration v2 references plus five-iteration exact 1× pairs for obsnum 144176 (tau 0.007) and 151718 (tau 0.101). | 6 | Ready to prepare |
| S0 | Does pointing transfer predict associated science transfer? | Inject identical sky-position/per-array truth into exact control/injected pointing and science branches; compare recovery by array. | 4 plus any required references | Blocked on association and science-injection approval |

The matrix is intentionally minimal: it reuses the existing representative
amplitude, uses the dry and highest-tau pointings for the widest available
condition span, and requests only one off-center position and one matched
pointing/science pair before broadening.

### Acceptance and failure controls for every run

- Use checkpoint schema v2 and one immutable Citlali executable; record version
  and SHA256 before launch.
- Freeze the input YAML, generated control/injected YAMLs, checkpoint path and
  hash, amplitude vector, source coordinate, and run manifest.
- Control and injected configs may differ only in output root and the injection
  enable switch (plus an approved position field for P0/S0).
- Prove equality of the first restarted control signal, kernel, and weight
  products against its uninterrupted reference.
- Give every save-all-iterations job a unique workspace; never launch two jobs
  into the same output tree.
- Missing iteration products, non-contiguous absolute iteration headers, or
  unexpected error-level log messages fail the run.

## Unity launch handoff

Do not launch these yet. First freeze the tolerance for C0 and approve or defer
the P0/S0 diagnostic design. Paths below are templates; each `<RUN_ROOT>` must
be a new, unique Unity directory.

### C0: short endpoint continuation

Prepare one control and one injected continuation from the existing absolute
iteration-18 checkpoint, with `start_iteration: 19` and
`additional_iterations: 3`, preserving the existing amplitude vector and
each branch's own checkpoint state. Launch both with the retained v2 executable:

```bash
<SHA256_VERIFIED_CITLALI> -l info <RUN_ROOT>/setup/citlali_control.yaml
<SHA256_VERIFIED_CITLALI> -l info <RUN_ROOT>/setup/citlali_injected.yaml
```

Download and append the three iterations to the existing 9–18 metric sequence.
Evaluate the four tolerances and the S/N guard before preparing another block.
Because the existing lineage already passed the exact iteration-9
uninterrupted/restarted equality gate, changing the binary, base config, or
checkpoint family invalidates this shortcut and requires a fresh uninterrupted
reference.

### L0: faint and bright linearity pairs

Use the existing v2 obsnum 133410 iteration-8 reference checkpoint and the
current setup tool twice:

```bash
$HOME/tolteca/bin/python \
  tools/fruit_loops/prepare_injected_source_pair.py \
  --input <FROZEN_OBS133410_CONFIG> \
  --restart-path <V2_REFERENCE>/reduced/redu08 \
  --output-dir <RUN_ROOT>/faint/setup \
  --runtime-output-root <RUN_ROOT>/faint \
  --start-iteration 9 --additional-iterations 10 \
  --amplitudes-mjy-beam 398.127 479.970 633.159

$HOME/tolteca/bin/python \
  tools/fruit_loops/prepare_injected_source_pair.py \
  --input <FROZEN_OBS133410_CONFIG> \
  --restart-path <V2_REFERENCE>/reduced/redu08 \
  --output-dir <RUN_ROOT>/bright/setup \
  --runtime-output-root <RUN_ROOT>/bright \
  --start-iteration 9 --additional-iterations 10 \
  --amplitudes-mjy-beam 11943.817 14399.093 18994.771
```

Launch the generated control and injected YAML for each amplitude. Compare each
first control iteration to `<V2_REFERENCE>/reduced/redu09`, then measure
recovery-versus-input slope and intercept by array. A single shared control
may be recognized as numerically duplicate only after its config and products
pass the same equality checks; do not silently substitute it in the manifests.

### R0: dry and high-tau pointing pairs

For each of obsnums 144176 and 151718, generate and launch a fresh uninterrupted
ten-iteration v2 reference:

```bash
$HOME/tolteca/bin/python \
  tools/fruit_loops/prepare_injected_source_reference.py \
  --input <FROZEN_OBS_CONFIG> \
  --output-dir <RUN_ROOT>/obs<OBSNUM>/setup_reference \
  --runtime-output-root <RUN_ROOT>/obs<OBSNUM> \
  --iterations 10

<SHA256_VERIFIED_CITLALI> -l info \
  <RUN_ROOT>/obs<OBSNUM>/setup_reference/citlali_injected_source_reference.yaml
```

Then generate a five-iteration control/injected pair from `redu08` with a
predeclared synthetic per-array amplitude vector. For comparability, prefer the
existing obsnum 133410 1× vector unless the map dynamic range requires a
documented lower value. Launch both generated YAMLs and compare their first
control iteration exactly to reference `redu09`.

### P0: off-center pair, after approval

After the position-aware diagnostic described below is reviewed, generate one
otherwise identical obsnum 133410 pair with a sky position chosen inside the
intersection of the three arrays' frozen high-weight masks. Record both sky
coordinates and resulting pixel coordinates. Launch the generated control and
injected YAML and apply the standard exact-restart gate. Do not select the
position after inspecting the recovery.

### S0: matched pointing/science pair, after approval and association

The user must identify one science observation and its governing pointing
observation/config. Freeze one sky coordinate covered by both products and one
per-array truth vector. Prepare four unique output roots:

```text
<RUN_ROOT>/pointing/control
<RUN_ROOT>/pointing/injected
<RUN_ROOT>/science/control
<RUN_ROOT>/science/injected
```

Launch all branches with one immutable approved executable. Require exact
restart equality independently for the pointing and science controls. The
decisive table is, by array:

```text
pointing_recovery
science_recovery
science_recovery - pointing_recovery
science_recovery / pointing_recovery
prediction_uncertainty
selected_tolerance_pass
```

Do not infer science response from pointing response until S0 passes.

## Science-injection generalization design (approval required)

Current validation rejects `injected_source_test` outside pointing/OOF, and
the injection is wired into the pointing fruit-loop implementation. No C++ was
changed in this investigation.

A bounded generalization should:

1. move the diagnostic configuration to a reduction-mode-neutral typed
   contract while keeping it disabled by default;
2. describe source identity explicitly: ICRS sky coordinate, per-array
   amplitude in mJy/beam, spectral/array ordering, injection start iteration,
   and expected map support;
3. insert the unit-kernel source into the science fruit-loop timestream at the
   same semantic seam as pointing—after pristine input construction and
   immediately before subtraction of the previous map—without altering RTC,
   PTC, projection, or production defaults;
4. keep requested injection, effective injection plan, and realized
   map/pixel position distinct in provenance;
5. generate exact control/injected configs and manifests with unique outputs,
   identical immutable inputs, and checkpoint-v2 enforcement;
6. extend the comparator to celestial WCS alignment and a frozen
   weight/support mask, reporting the same amplitude, kernel projection, PSF,
   centroid, S/N, weight, noise, and map-change fields in both modes;
7. test startup rejection, coordinate conversion, array ordering, config-pair
   isolation, disabled-mode bit identity, and exact restart before requesting
   Unity reductions.

This is diagnostic scaffolding, not a new science calibration path. Its code
and run review should be separate from any numerical algorithm or build-system
change.

## Remaining limitations and shortest resolving experiment

| Limitation | Why it matters | Shortest resolution |
|---|---|---|
| Strict endpoint stability is not sustained | Iteration 17→18 alone looks better than the two-transition window, especially in a1400. | C0: one three-iteration continuation block; extend only if the selected criterion still fails and changes remain material. |
| Only one injected amplitude | A constant recovery fraction and nonlinear response are indistinguishable. | L0: add 0.1× and 3× exact pairs around the existing 1× point. |
| Only the map center is tested | Projection and coverage may change response away from center. | P0: one frozen off-center 1× exact pair after position-aware diagnostic approval. |
| Only obsnum 133410 has transfer truth | Weather/elevation/source dependence is unknown. | R0: dry obsnum 144176 and high-tau obsnum 151718 exact pairs. |
| No independent absolute-flux truth is archived | Real-source amplitude cannot validate photometric accuracy. | Archive date/frequency queries, bandpass integration, uncertainties, and model versions for each source before photometric interpretation. |
| No pointing/science association or science injection | The calibration-reference claim itself is untested. | S0: one approved matched pointing/science exact pair with identical injected sky source. |
| Centroid stability is relative to the realized kernel | Absolute astrometric zero point is not tested. | Cross-match the converged fitted centroid to the authoritative source coordinate with the final WCS and pointing model. |
| S/N falls while amplitude plateaus | A stopping rule could preserve transfer while unnecessarily degrading detection significance. | C0 reports both two-step and cumulative-tail S/N; owner selects a material-loss bound before adoption. |
