# SCI-ALIGN-001 66-pointing Unity fit campaign

This package turns the checksum-bound 66-observation source-quality cohort
into an owner-run Unity campaign. It reuses the validated map-space and direct
PTC estimators. It does not regenerate a reduction, edit production Citlali,
select on a timing result, or prescribe a correction.

The lifecycle is intentionally staged. All 66 map-space prerequisites may run
first. ObsNum 150818 then runs alone through the bounded `fit-gate` stage. The
remaining 65 gates are not launched until its PDF and machine record look
normal. Every fit gate stops before held-out refits, sensitivity fits, network
tests, and bootstraps. Although a resume array is generated, submitting it is
prohibited until the project owner reviews the complete gate census and gives
an explicit new approval.

The four direct-PTC fits are `constant`, `lag`, `hysteresis` (independent Az
and El direction-sign half-offsets), and `joint`. The map prerequisites use
the same frozen PTC/PPT identity and retain `constant`, `time_lag`,
`axis_sign`, and `joint` results for the eventual paired comparison.

## 1. Unity environment and freeze job

After fetching the associated commit on Unity, start from a fresh shell:

```bash
source /work/toltec/toltec_shared/toltec_astro/venvs/toltec_20241023/bin/activate

export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export SCI_SCHEMA_AUDIT=/work/toltec/wilson/sci_align_001_existing_ptc_inventory_2026-08-12/pointing_standard_trial_schema_audit.csv
export SCI_CAMPAIGN_ROOT=/work/toltec/wilson/citlali_testing/pointing/sci_align_001_pointing_fit_campaign_66_2026-08-12
export SCI_PACKAGE="$SCI_REPO/validation/sci_align_001_pointing_fit_campaign_2026-08-12"

command -v python
python -c 'import astropy, netCDF4, scipy; print(astropy.__version__, scipy.__version__)'
test ! -e "$SCI_CAMPAIGN_ROOT"

freeze_job_id=$(sbatch --parsable --export=ALL \
  "$SCI_PACKAGE/freeze_on_unity.sbatch")
freeze_job_id=${freeze_job_id%%;*}
printf 'freeze_job_id=%s\n' "$freeze_job_id"
```

Monitor it with:

```bash
squeue -j "$freeze_job_id" -o "%.18i %.30j %.8T %.10M %.4C %R"
tail -F "/work/toltec/wilson/sci-align-pointing-freeze_${freeze_job_id}.out"
```

Completion is the line `campaign frozen: pointings=66 pilot=150818`. Verify:

```bash
(cd "$SCI_CAMPAIGN_ROOT" && shasum -a 256 -c PREPARATION_SHA256SUMS)
(cd "$SCI_CAMPAIGN_ROOT/jobs" && shasum -a 256 -c JOB_SHA256SUMS)
(cd "$SCI_CAMPAIGN_ROOT/frozen" && shasum -a 256 -c SELECTION_SHA256SUMS)

python - "$SCI_CAMPAIGN_ROOT/campaign_preparation.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
print("repository_commit=", doc["repository_commit"])
print("observation_count=", doc["observation_count"])
print("pilot_obsnum=", doc["pilot_obsnum"])
print("lifecycle_stop=", doc["lifecycle_stop"])
PY
```

The freeze job reads and hashes the 66 canonical Unity PTCs, so it is run under
Slurm rather than tied to a login terminal. It also verifies every PPT against
the prior quality package and records the descriptive aberration covariates.

## 2. Map-space prerequisites

```bash
map_array_job_id=$(sbatch --parsable \
  "$SCI_CAMPAIGN_ROOT/jobs/run_map_array.sbatch")
map_array_job_id=${map_array_job_id%%;*}

map_aggregate_job_id=$(sbatch --parsable \
  --dependency="afterok:${map_array_job_id}" \
  "$SCI_CAMPAIGN_ROOT/jobs/run_map_aggregate.sbatch")
map_aggregate_job_id=${map_aggregate_job_id%%;*}

printf 'map_array_job_id=%s map_aggregate_job_id=%s\n' \
  "$map_array_job_id" "$map_aggregate_job_id"
```

Monitor with `squeue -j "$map_array_job_id,$map_aggregate_job_id"`. After the
aggregate completes:

```bash
test "$(find "$SCI_CAMPAIGN_ROOT/map_results" -mindepth 2 \
  -maxdepth 2 -name result.json -type f | wc -l)" -eq 66
(cd "$SCI_CAMPAIGN_ROOT/map_results" && shasum -a 256 -c SHA256SUMS)
```

Each `o<obsnum>` directory also contains and authenticates its own
`SHA256SUMS`. A failed task must be diagnosed or rerun into a fresh campaign;
do not delete a partial directory and silently replace it under the same
frozen identity. The aggregate job writes only corpus summary files into the
already populated `map_results` root; it does not rerun any observation.

## 3. Pilot fit gate, then stop and inspect

```bash
pilot_job_id=$(sbatch --parsable \
  "$SCI_CAMPAIGN_ROOT/jobs/run_fit_gate_pilot.sbatch")
pilot_job_id=${pilot_job_id%%;*}
squeue -j "$pilot_job_id" -o "%.18i %.30j %.8T %.10M %.4C %R"
```

Expected runtime is roughly seven minutes; the numerical runner has a
1,800-second wall limit and Slurm has a 45-minute outer limit. Completion is
`fit gate complete: obs=150818`. Then verify:

```bash
export SCI_PILOT="$SCI_CAMPAIGN_ROOT/fit_results/o150818"
(cd "$SCI_PILOT" && shasum -a 256 -c FIT_GATE_SHA256SUMS)
python - "$SCI_PILOT/fit_gate.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
print("status=", doc["quality_gate"]["automatic_structural_status"])
for name, fit in doc["point_model_results"].items():
    print(name, fit["status"], "tau_ms=", fit["tau_ms"],
          "parameters=", fit.get("parameters", {}))
PY
```

Download the pilot PDF from the local machine:

```bash
export SCI_REMOTE_CAMPAIGN=/work/toltec/wilson/citlali_testing/pointing/sci_align_001_pointing_fit_campaign_66_2026-08-12
scp unity_toltec:"$SCI_REMOTE_CAMPAIGN/fit_results/o150818/lissajous_fit_gate_o150818.pdf" .
```

The fitted value is not an acceptance criterion. Inspect source crossings,
model/data overlays, objective behavior, optimizer census, residuals, and the
structural gate. Stop here if the pilot is malformed or pathological.

## 4. Remaining 65 bounded fit gates

Only after the pilot review:

```bash
fit_gate_array_job_id=$(sbatch --parsable \
  "$SCI_CAMPAIGN_ROOT/jobs/run_fit_gate_remaining_array.sbatch")
fit_gate_array_job_id=${fit_gate_array_job_id%%;*}
printf 'fit_gate_array_job_id=%s\n' "$fit_gate_array_job_id"

squeue -j "$fit_gate_array_job_id" \
  -o "%.18i %.30j %.8T %.10M %.4C %R"
```

The `%4` array throttle permits four simultaneous observations. When it has
left the queue, inspect task states with `sacct -X -j "$fit_gate_array_job_id"`
and make the checksum-bound census:

```bash
export SCI_GATE_AUDIT="$SCI_CAMPAIGN_ROOT/fit_gate_audit_v1"
test ! -e "$SCI_GATE_AUDIT"
python "$SCI_REPO/tools/diagnostics/prepare_sci_align_001_pointing_fit_campaign.py" \
  audit-gates \
  --selection "$SCI_CAMPAIGN_ROOT/frozen/selected_pointings.json" \
  --fit-root "$SCI_CAMPAIGN_ROOT/fit_results" \
  --output "$SCI_GATE_AUDIT"

(cd "$SCI_GATE_AUDIT" && shasum -a 256 -c SHA256SUMS)
column -s, -t < <(grep -v '^#' "$SCI_GATE_AUDIT/fit_gate_status.ecsv") | less -S
```

## Mandatory stopping point

Return the pilot PDF, `fit_gate_audit_v1`, and any failed task logs for owner
review. Do **not** submit `run_resume_array.sbatch` yet. That script begins the
expensive checkpointed held-out, sensitivity, network, and paired-bootstrap
work and includes `--owner-review-approved`; its presence is preparation, not
approval. No corpus timing inference is produced by this package before that
separate decision.
