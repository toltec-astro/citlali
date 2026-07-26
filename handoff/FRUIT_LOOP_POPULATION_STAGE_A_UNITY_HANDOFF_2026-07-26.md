# Fruit-loop population Stage A Unity handoff

Date: 2026-07-26

Status: setup prepared and locally validated; no upload, Unity snapshot, or
Slurm submission has been performed

## Frozen run

Stage A contains 16 independent pointing reductions selected before examining
their fruit-loop behavior:

- 8 normal, 5 marginal, and 3 stress observations;
- all six source classes represented in the 108-observation population;
- ten saved iterations (`redu00` through `redu09`);
- processed-kernel and fruit-loop diagnostic output enabled;
- checkpoint-v2 fields present with no restart;
- injected-source testing disabled; and
- one observation and one output workspace per process.

The upload source is:

```text
validation/fruit_loop_population_stage_a_2026-07-26/
```

Its manifest records the 108-input source-config SHA256, population-run-matrix
SHA256, per-config SHA256 values, ranks, strata, APTs, and output paths.

## 1. Upload from the local machine

Run from the Citlali repository:

```bash
cd /Users/gwilson/GitHub/citlali-refactor

LOCAL_SETUP="$PWD/validation/fruit_loop_population_stage_a_2026-07-26"
UNITY_PROJECT="/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1"
UNITY_SETUP="${UNITY_PROJECT}/diagnostics/fruitloop_population_v1/stage_a/setup"

ssh unity_toltec "mkdir -p '${UNITY_SETUP}'"
rsync -av --checksum \
  "${LOCAL_SETUP}/" \
  "unity_toltec:${UNITY_SETUP}/"
```

The command intentionally has no `--delete`.

## 2. Freeze the binary and preflight on Unity

Log into Unity and run:

```bash
ssh unity_toltec

export PROJECT_ROOT=/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1
export OUTPUT_ROOT="${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_a"
export SETUP_DIR="${OUTPUT_ROOT}/setup"
export CITLALI_SOURCE=/work/toltec/citlali_dev/citlali_refactor/build/bin/citlali

cd "${SETUP_DIR}"
./snapshot_binary.sh
./preflight_stage_a.sh
```

`snapshot_binary.sh` copies the executable into `setup/bin/` under its SHA256
name and writes `binary.env`. Every array task uses that copy, so rebuilding
the development binary while the array is running cannot change later tasks.

The preflight refuses to proceed if:

- the snapshot or a config checksum differs;
- any APT or raw input path is missing;
- any observation output directory is nonempty; or
- the Stage A filesystem has less than 30 GiB free; or
- the job table does not contain exactly 16 entries.

The earlier five-observation/five-iteration run used about 1.4 GiB. Linear
scaling suggests roughly 9 GiB for this 16-by-ten run, but the 30 GiB preflight
floor leaves room for quality-dependent product size and logs. Override
`MIN_FREE_KIB` only after a deliberate storage check.

## 3. Submit

The default concurrency is four tasks (24 requested CPUs and 256 GiB aggregate
requested memory when all four run):

```bash
ARRAY_CONCURRENCY=4 ./submit_stage_a.sh
```

For a more conservative first launch:

```bash
ARRAY_CONCURRENCY=2 ./submit_stage_a.sh
```

Record the Slurm array job ID printed by `sbatch`. Do not rerun the submit
script after any task has created products; the collision preflight will
correctly refuse a mixed restart.

## 4. Monitor

From the same Unity shell:

```bash
./status_stage_a.sh
squeue -u "$USER" -n flpop-a
sacct -j ARRAY_JOB_ID \
  --format=JobID,State,Elapsed,MaxRSS,ExitCode \
  --units=G
```

`products_present` means ten `redu??` directories exist. It is not the
scientific gate: all logs still require the unexpected-error audit and all
products require trajectory analysis.

If a task fails, preserve its output and log. Diagnose first, then create a
separate retry root and job; do not overwrite or append to the partial output.

## 5. Download and verify from the local machine

After all jobs leave the queue:

```bash
LOCAL_STAGE="$HOME/work_toltec/local_data/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_a"
UNITY_STAGE="/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_a"

mkdir -p "${LOCAL_STAGE}"
rsync -av --checksum --partial \
  "unity_toltec:${UNITY_STAGE}/" \
  "${LOCAL_STAGE}/"

rsync -avn --checksum \
  "unity_toltec:${UNITY_STAGE}/" \
  "${LOCAL_STAGE}/"
```

The final dry run should list no files requiring transfer. Keep the Unity copy
until the local analysis has passed the Stage A gate and a second backup or
archive exists.

## Cleanup boundary

No active Stage A data is safe to remove yet. Never remove the same evidence
tree from both local storage and Unity unless a verified third copy exists.

### Protect now

Keep these on at least one authoritative filesystem, and keep the RC1 and
Stage A items locally available through analysis:

| Tree | Local size on 2026-07-26 | Reason |
|---|---:|---|
| `apts/` | 2.2 GiB | ordinary and modeled-frequency calibration inputs |
| `pointings_rc1/` | 14.4 GiB | current RC1 baseline subset and source config |
| `diagnostics/hero_trial_remaining/rc1/` | 18.7 GiB | completes the 108-observation RC1 baseline |
| `diagnostics/standard_trial/` | 32.8 GiB | distinct ordinary-APT control, not a duplicate of RC1 |
| `diagnostics/fruitloops5_rc1_ablation/` | 5.0 GiB | mechanistic feedback evidence |
| `diagnostics/fruitloops5_rc1_injected_source_v2/` | 2.2 GiB | valid exact-restart transfer evidence |
| new `diagnostics/fruitloop_population_v1/stage_a/` | unknown | active experiment |

Also protect project `data/`, the external beammap raw-data tree referenced by
the configs, setup/config/manifest files, binary SHA/version records, logs, and
the small TolAPT/validation metric tables.

### Safe to remove now, but low return

`diagnostics/137372_v4_smoke/` is the incomplete 25 MiB predecessor to the
successful `137372_v4_smoke_retry`. It can be removed locally and on Unity
after copying its YAML, log, profile, and runtime-provenance files into the
project audit archive. Because the saving is small, leaving it in place is
also reasonable.

### Safe only after evidence/archive checks

These are the first substantial candidates, in this order:

1. `diagnostics/fruitloops5_rc1_injected_source/` (0.56 GiB) is the quarantined
   checkpoint-v1 control/injected experiment. Keep its setup, executable SHA,
   logs, and failure report; its bulky map products can go after confirming
   the checkpoint-v2 evidence is complete.
2. `pointings_v4/` (14.4 GiB) together with
   `diagnostics/hero_trial_remaining/v4/` (18.6 GiB) is the complete historical
   v4 baseline. The derived v4-versus-RC1 tables exist, but their rows point
   back to these FITS files. Remove this 33.0 GiB only after freezing the
   derived tables, manifests, plots, configs, logs, and one checksummed archive
   of the source products—or explicitly accepting that map-level re-analysis
   will require a rerun.
3. `diagnostics/standard_trial/` (32.8 GiB) may be removed from Unity after its
   local mirror is verified byte-for-byte and the ordinary-APT comparison is
   closed. It is not currently safe to delete from both locations.
4. The old five-observation real-source products
   (`diagnostics/fruitloops5_rc1/`, 1.4 GiB) become superseded only after the
   common-binary ten-iteration Stage A reruns are downloaded, analyzed, and
   accepted.
5. Smoke-retry, equivalence, and intermediate ablation map products may be
   thinned after their conclusions and provenance are captured. Their
   configs, logs, manifests, binary checksums, and final comparison tables
   should remain.

### Read-only audit commands

Run locally:

```bash
PROJECT="$HOME/work_toltec/local_data/2026-ENG-hero-multiyear-pointings-v1"
du -sh \
  "${PROJECT}/apts" \
  "${PROJECT}/pointings_v4" \
  "${PROJECT}/pointings_rc1" \
  "${PROJECT}/diagnostics/standard_trial" \
  "${PROJECT}/diagnostics/hero_trial_remaining/v4" \
  "${PROJECT}/diagnostics/hero_trial_remaining/rc1" \
  "${PROJECT}/diagnostics/fruitloops5_rc1" \
  "${PROJECT}/diagnostics/fruitloops5_rc1_ablation" \
  "${PROJECT}/diagnostics/fruitloops5_rc1_injected_source" \
  "${PROJECT}/diagnostics/fruitloops5_rc1_injected_source_v2"
```

Run the same `du -sh` list on Unity after setting:

```bash
PROJECT=/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1
```

Before deleting a completed Unity tree, first run an `rsync -avn --checksum`
comparison against its intended local mirror. A size match alone is not a
content verification.
