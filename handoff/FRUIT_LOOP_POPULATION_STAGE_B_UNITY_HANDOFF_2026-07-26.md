# Fruit-loop population Stage B Unity handoff

Date: 2026-07-26

Status: Stage A gate passed; 92-observation Stage B bundle prepared and locally
validated; no Stage B upload or Slurm submission has been performed

## Frozen run

Stage B contains the 92 observations not present in Stage A. It retains the
same ten-iteration policy and requires the exact Stage A executable:

```text
SHA256 0f7685ad2b89cc2fc2cbe330c9e5ed75fc8972dc1bf60ab37e3a4b9209965330
v4.0.0-3596-g6ce35c14 (2026-07-26T01:20:09)
kids 04088da (2026-07-25T22:56:46)
```

The upload source is:

```text
validation/fruit_loop_population_stage_b_2026-07-26/
```

## 1. Upload from the local machine

Run from the Citlali repository:

```bash
cd /Users/gwilson/GitHub/citlali-refactor

LOCAL_SETUP="$PWD/validation/fruit_loop_population_stage_b_2026-07-26"
UNITY_PROJECT="/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1"
UNITY_SETUP="${UNITY_PROJECT}/diagnostics/fruitloop_population_v1/stage_b/setup"

ssh unity_toltec "mkdir -p '${UNITY_SETUP}'"
rsync -av --checksum \
  "${LOCAL_SETUP}/" \
  "unity_toltec:${UNITY_SETUP}/"
```

The upload intentionally has no `--delete`.

## 2. Snapshot the Stage A binary and preflight

```bash
ssh unity_toltec

export PROJECT_ROOT=/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1
export OUTPUT_ROOT="${PROJECT_ROOT}/diagnostics/fruitloop_population_v1/stage_b"
export SETUP_DIR="${OUTPUT_ROOT}/setup"

cd "${SETUP_DIR}"
./snapshot_binary.sh
./preflight_stage_b.sh
```

The snapshot script defaults to the SHA-named executable retained in the Stage
A setup and refuses any other SHA256. The preflight checks all 92 configs,
inputs, APTs, empty output workspaces, and at least 350 GiB free.

Stage A occupies about 50 GB for 16 observations. Linear scaling predicts
about 288 GB for Stage B; the 350 GiB gate leaves headroom. If the free-space
gate fails, inspect storage rather than overriding it reflexively.

## 3. Submit

The default concurrency is four tasks:

```bash
ARRAY_CONCURRENCY=4 ./submit_stage_b.sh
```

For a conservative launch:

```bash
ARRAY_CONCURRENCY=2 ./submit_stage_b.sh
```

Record the Slurm array job ID. Do not submit again after any task has created
products; the collision gate will refuse a mixed restart.

Each successful task restores the frozen setup config's mode on all ten copied
configs, verifies owner readability, and checks their content against the
setup checksum. This closes the isolated Stage A `0200` copied-config
permission failure without changing file content or science policy.

## 4. Monitor

```bash
./status_stage_b.sh
squeue -u "$USER" -n flpop-b
sacct -j ARRAY_JOB_ID \
  --format=JobID,State,Elapsed,MaxRSS,ExitCode \
  --units=G
```

`products_present` means ten `redu??` directories exist; it is not a
scientific convergence verdict.

If a task fails, preserve its output and log. Diagnose it before creating a
separate retry root; do not overwrite or append to partial products.

## 5. Download and verify from the local machine

After all tasks leave the queue:

```bash
LOCAL_STAGE="$HOME/work_toltec/local_data/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_b"
UNITY_STAGE="/work/toltec/commissioning2025-test/2026-ENG-hero-multiyear-pointings-v1/diagnostics/fruitloop_population_v1/stage_b"

mkdir -p "${LOCAL_STAGE}"
rsync -av --checksum --partial --info=progress2 \
  "unity_toltec:${UNITY_STAGE}/" \
  "${LOCAL_STAGE}/"

rsync -avn --checksum \
  "unity_toltec:${UNITY_STAGE}/" \
  "${LOCAL_STAGE}/"
```

The final dry run should list no files requiring transfer. Keep the Unity copy
until the full 108-observation analysis is complete and a second verified copy
or archive exists.
