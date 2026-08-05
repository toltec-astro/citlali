# SCI-ALIGN-001 3C273 16-map replay campaign

On 2026-08-05 the owner authorized a bounded expansion from the completed
source-isolated 148670 replay to a 16-independent-beammap diagnostic campaign.
The completed 148670 replay counts as one member. This runbook prepares the
remaining fifteen replays in four batches of at most four concurrent Slurm
jobs. It creates diagnostic products only; it does not alter raw inputs,
historical reductions, APTs, production configuration, or timing assignments.

The selected new ObsNums are stratified over the retained corpus time range:

| Batch | New ObsNums |
| --- | --- |
| 1 | 113862, 131925, 136279, 152882 |
| 2 | 128588, 133543, 150819, 152451 |
| 3 | 129687, 134643, 151126, 151950 |
| 4 | 130922, 135397, 151600 |

The existing 148670 replay is retained as the sixteenth member. Do not use an
unlisted ObsNum without a new owner decision.

## 1. Stage and verify the numbered-config authority

The campaign derives direct Citlali inputs from the established numbered-YAML
contract: `70_reduce.yaml` supplies the shared low-level Beammap policy and
`72_reduce.yaml` supplies the exact per-ObsNum astrometry/photometry entries.
The latter's historical duplicate `select` keys are illustrative run selection,
not campaign authority: this generator makes one explicit config per ObsNum.

On the local source machine, stage the two named authorities for owner transfer:

```bash
shasum -a 256 \
  /Users/gwilson/work_toltec/local_data/beammaps/3c273/70_reduce.yaml \
  /Users/gwilson/work_toltec/local_data/beammaps/3c273/72_reduce.yaml

ssh unity_toltec 'mkdir -p /work/toltec/wilson/sci_align_001_campaign_authority_2026-08-05'

scp /Users/gwilson/work_toltec/local_data/beammaps/3c273/70_reduce.yaml \
  /Users/gwilson/work_toltec/local_data/beammaps/3c273/72_reduce.yaml \
  unity_toltec:/work/toltec/wilson/sci_align_001_campaign_authority_2026-08-05/
```

On Unity, verify the received files before preparation:

```bash
export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export CITLALI_BIN="$SCI_REPO/build/bin/citlali"
export SCI_RUN_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_corpus_run_2026-08-03
export SCI_OUTPUT_ROOT="$SCI_RUN_ROOT/output"
export SCI_CAMPAIGN_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_replay_campaign_16_2026-08-05
export SCI_DATA_ROOT="$HOME/c2025t/beammaps/data"
export SCI_AUTHORITY_ROOT=/work/toltec/wilson/sci_align_001_campaign_authority_2026-08-05
export SCI_BASE_CONFIG="$SCI_AUTHORITY_ROOT/70_reduce.yaml"
export SCI_CALIBRATION_CONFIG="$SCI_AUTHORITY_ROOT/72_reduce.yaml"

cd "$SCI_REPO"
test -z "$(git status --short)"
test -f "$CITLALI_BIN"
test ! -e "$SCI_CAMPAIGN_ROOT"
test -d "$SCI_DATA_ROOT"
test -f "$SCI_BASE_CONFIG"
test -f "$SCI_CALIBRATION_CONFIG"

shasum -a 256 "$SCI_BASE_CONFIG" "$SCI_CALIBRATION_CONFIG"
# Expected values from the local source authority:
# 71273b3199bc762e406f51d34f9e31c8cf51d42ac0d30889d2bf4cace1ebdaff  70_reduce.yaml
# 7fec4b95842ecdaa0414592dac116120ee88fae52ff0e881368072876fe6ddfa  72_reduce.yaml

python tools/diagnostics/prepare_sci_align_001_3c273_replay_campaign.py \
  describe --base-config "$SCI_BASE_CONFIG" \
  --calibration-config "$SCI_CALIBRATION_CONFIG"

```

Stop if either SHA differs, the printed ObsNum/batch list differs from the
table above, or the campaign root already exists.

## 2. Resolve source identity and render all fifteen replays

This step does not invoke Citlali or submit work. It reads the 70 low-level
policy and the 72 per-ObsNum calibration, selects exact scannum-2 raw files,
telescope product, and matched-input APT by filename, hashes every selected
input, and emits a separate output root. It changes only runtime output/thread
settings, the absolute raw fit-report directory, the active refactor prior
path, and the existing detector-TOD sidecar request.

It can read hundreds of GiB while calculating checksums, so tens of minutes to
several hours of filesystem time is normal. Do not interrupt it merely because
it is quiet. If it fails, preserve the partial root as evidence; do not delete
or reuse that root.

```bash
python tools/diagnostics/prepare_sci_align_001_3c273_replay_campaign.py \
  prepare --base-config "$SCI_BASE_CONFIG" \
  --calibration-config "$SCI_CALIBRATION_CONFIG" \
  --analysis-root /work/toltec/wilson/citlali_testing/beammaps/3c273 \
  --raw-root "$SCI_DATA_ROOT" --repo-root "$SCI_REPO" \
  --output-root "$SCI_CAMPAIGN_ROOT" \
  --citlali-bin "$CITLALI_BIN" \
  --threads 6

(cd "$SCI_CAMPAIGN_ROOT" && shasum -a 256 -c PREPARATION_SHA256SUMS)
sed -n '1,220p' "$SCI_CAMPAIGN_ROOT/campaign_preparation.json"
for script in "$SCI_CAMPAIGN_ROOT"/submit_batch_*.sh; do bash -n "$script"; done
```

An absent or ambiguous raw/telescope/APT binding, wrong-ObsNum raw path,
unreadable numbered configuration, or output overlap is a hard stop before any
job is submitted. Return the error and the intact partial root for review.

## 3. Submit and review one batch at a time

After reviewing the generated configuration and input manifest for all four
members, submit only one batch script. Each script calls `sbatch` at most four
times; it contains no array and cannot start a fifth replay.

```bash
bash "$SCI_CAMPAIGN_ROOT/submit_batch_01.sh" | tee "$SCI_CAMPAIGN_ROOT/batch_01_job_ids.txt"
```

Record the returned ObsNum/job-id pairs. Once all four jobs are complete,
verify the corresponding products before starting the next batch:

```bash
for obs in 113862 131925 136279 152882; do
  replay="$SCI_CAMPAIGN_ROOT/replay_o${obs}"
  (cd "$replay" && shasum -a 256 -c SHA256SUMS) || exit 1
  find "$replay/reduced" -type f \
    \( -name '*_ptc_detector_tod.nc' -o -name 'timestream_output_provenance.yaml' \
       -o -name 'raw_timestream_provenance.yaml' \) -print | sort
done
```

Repeat with `submit_batch_02.sh`, `submit_batch_03.sh`, and
`submit_batch_04.sh` only after the preceding batch verifies. A failed job or a
missing required sidecar is a stop for that observation: preserve its evidence,
do not overwrite the replay root, and do not substitute a different ObsNum.

## 4. Rebuild inventory only after all successful replays are retained

After the authorized campaign completes, rerun the inventory while continuing
to exclude the old corpus run root, but **do not exclude the campaign root**:
its `replay_o*/reduced` directories are the deliberately generated candidates
to be discovered. The inventory only treats recognized reduced products as
candidates; campaign scripts and evidence remain non-candidates. Freeze a new
selection and rerun the existing per-map and aggregate pipeline only after
owner review of its candidate and omission tables.

The producer interpretation remains unchanged: integer-second `T0` is a
session identity candidate, shared reference/PPS does not imply shared sample
phase, and all raw-counter results concern delivered metadata. No result from
this campaign authorizes a raw-row reassociation, a clock-drift claim, or a
production timing correction.
