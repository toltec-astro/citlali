# Unity owner runbook: split-direction Beammap products

This runbook is executed by the owner on Unity. Codex does not access Unity.
It creates new diagnostic roots and never edits raw data, retained reductions,
or the two completed replay configurations used as input authorities.

Run ObsNum 150819 first. Do not submit 148670 until all four 150819 jobs have
finished and their retained products pass the checks below.

## 1. Environment and source authorities

Load the normal TolTEC environment first so that `python` has `yaml` and the
usual science dependencies. Then define:

```bash
export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export CITLALI_BIN="$SCI_REPO/build/bin/citlali"
export SCI_ANALYSIS_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273
export SCI_CAMPAIGN_ROOT="$SCI_ANALYSIS_ROOT/sci_align_001_replay_campaign_16_2026-08-05_retry1"
export SCI_REPLAY_148670="$SCI_ANALYSIS_ROOT/sci_align_001_reproduction_148670_2026-08-03_retry1"
export SCI_SPLIT_ROOT="$SCI_ANALYSIS_ROOT/sci_align_001_split_direction_beammap_2026-08-06"

cd "$SCI_REPO"
test -z "$(git status --short)"
test -x "$CITLALI_BIN"
command -v python
python -c 'import yaml; print(yaml.__version__)'
test ! -e "$SCI_SPLIT_ROOT"
```

Resolve exactly one completed direct Citlali config for each observation:

```bash
mapfile -t SCI_CONFIGS_150819 < <(
  find "$SCI_CAMPAIGN_ROOT/replay_o150819/config" -maxdepth 1 \
    -type f -name '*.yaml' -print | sort
)
mapfile -t SCI_CONFIGS_148670 < <(
  find "$SCI_REPLAY_148670/config" -maxdepth 1 \
    -type f -name '*.yaml' -print | sort
)
test "${#SCI_CONFIGS_150819[@]}" -eq 1
test "${#SCI_CONFIGS_148670[@]}" -eq 1
export SCI_SOURCE_CONFIG_150819="${SCI_CONFIGS_150819[0]}"
export SCI_SOURCE_CONFIG_148670="${SCI_CONFIGS_148670[0]}"
printf '150819 source=%s\n148670 source=%s\n' \
  "$SCI_SOURCE_CONFIG_150819" "$SCI_SOURCE_CONFIG_148670"
shasum -a 256 "$SCI_SOURCE_CONFIG_150819" "$SCI_SOURCE_CONFIG_148670" \
  "$CITLALI_BIN"
```

Stop if either config is absent/ambiguous, the executable is missing, the Git
worktree is dirty, or the new diagnostic root already exists.

## 2. Render eight isolated mode configs and Slurm scripts

The following preparation reads the two completed configs, changes only
`beammap.direction_mode` and `runtime.output_dir`, and emits one isolated root
and Slurm script per observation/mode. It launches nothing.

```bash
python - "$SCI_SOURCE_CONFIG_150819" "$SCI_SOURCE_CONFIG_148670" \
  "$SCI_SPLIT_ROOT" "$CITLALI_BIN" <<'PY'
import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

import yaml

source_by_obs = {150819: Path(sys.argv[1]), 148670: Path(sys.argv[2])}
root = Path(sys.argv[3])
citlali = Path(sys.argv[4])
if root.exists():
    raise SystemExit(f"refusing existing output root: {root}")
if not citlali.is_file():
    raise SystemExit(f"missing Citlali executable: {citlali}")

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

root.mkdir(parents=True)
(root / "preparation").mkdir()
(root / "jobs").mkdir()
manifest = {
    "schema": "sci-align-001-split-direction-preparation-v1",
    "source_products_modified": False,
    "citlali": {"path": str(citlali), "sha256": sha256(citlali)},
    "runs": [],
}
for obsnum in (150819, 148670):
    source = source_by_obs[obsnum].resolve()
    base = yaml.safe_load(source.read_text())
    if not isinstance(base, dict):
        raise SystemExit(f"config root is not a mapping: {source}")
    for mode in ("standard", "left", "right", "all"):
        config = copy.deepcopy(base)
        beammap = config.setdefault("beammap", {})
        runtime = config.setdefault("runtime", {})
        if not isinstance(beammap, dict) or not isinstance(runtime, dict):
            raise SystemExit(f"beammap/runtime is not a mapping: {source}")
        mode_root = root / f"o{obsnum}" / mode
        config_path = root / "preparation" / f"citlali_o{obsnum}_{mode}.yaml"
        beammap["direction_mode"] = mode
        runtime["output_dir"] = str(mode_root / "reduced")
        rendered = yaml.safe_dump(config, sort_keys=False)
        config_path.write_text(rendered)
        script = root / "jobs" / f"run_o{obsnum}_{mode}.sbatch"
        script.write_text("\n".join([
            "#!/usr/bin/env bash",
            f"#SBATCH --job-name=sci-align-{obsnum}-{mode}",
            "#SBATCH --time=48:00:00",
            "#SBATCH --mem=64G",
            "#SBATCH --cpus-per-task=6",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            "#SBATCH --partition=toltec-cpu",
            f"#SBATCH --output={root}/jobs/%x_%j.out",
            f"#SBATCH --error={root}/jobs/%x_%j.err",
            "#SBATCH --parsable",
            "set -euo pipefail",
            f"cd {json.dumps(str(citlali.parent.parent.parent))}",
            f"{json.dumps(str(citlali))} {json.dumps(str(config_path))} --grppiex omp",
            "",
        ]))
        script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
        manifest["runs"].append({
            "observation_number": obsnum,
            "mode": mode,
            "source_config": str(source),
            "source_config_sha256": sha256(source),
            "rendered_config": str(config_path),
            "rendered_config_sha256": sha256(config_path),
            "output_root": str(mode_root),
            "submit_script": str(script),
        })
(root / "preparation" / "manifest.json").write_text(
    json.dumps(manifest, indent=2, sort_keys=True) + "\n")
PY

find "$SCI_SPLIT_ROOT/preparation" -maxdepth 1 -type f -print | sort
for script in "$SCI_SPLIT_ROOT"/jobs/*.sbatch; do bash -n "$script"; done
python - "$SCI_SPLIT_ROOT/preparation/manifest.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for row in doc["runs"]:
    print(row["observation_number"], row["mode"], row["output_root"])
PY
```

Review the eight output roots. Each must be below `SCI_SPLIT_ROOT`; no source
config or retained reduction may appear as an output.

## 3. Run 150819 first

Submit the four independent modes. They may run concurrently; each has its own
output root.

```bash
: > "$SCI_SPLIT_ROOT/jobs/o150819_job_ids.txt"
for mode in standard left right all; do
  job_id=$(sbatch "$SCI_SPLIT_ROOT/jobs/run_o150819_${mode}.sbatch")
  job_id=${job_id%%;*}
  printf 'obsnum=150819 mode=%s job_id=%s\n' "$mode" "$job_id" \
    | tee -a "$SCI_SPLIT_ROOT/jobs/o150819_job_ids.txt"
done
```

Monitor without relying on the current login shell remaining alive:

```bash
squeue -j "$(awk -F= '{print $4}' "$SCI_SPLIT_ROOT/jobs/o150819_job_ids.txt" \
  | paste -sd, -)" -o '%.18i %.28j %.8T %.10M %.4C %R'
tail -f "$SCI_SPLIT_ROOT"/jobs/sci-align-150819-*.out
```

When `squeue` is empty, require four successful accounting records:

```bash
sacct -X -j "$(awk -F= '{print $4}' "$SCI_SPLIT_ROOT/jobs/o150819_job_ids.txt" \
  | paste -sd, -)" --format=JobID,JobName%30,State,ExitCode,Elapsed,Start,End
```

Then inspect the products:

```bash
for mode in standard left right all; do
  mode_root="$SCI_SPLIT_ROOT/o150819/$mode"
  printf '\n===== 150819 %s =====\n' "$mode"
  find "$mode_root" -type f \
    \( -name '*_citlali*.fits' -o -name 'apt_*_citlali*.ecsv' \
       -o -name 'beammap_direction_scan_registry*.csv' \) \
    -print | sort
done
```

Require ordinary map FITS and APT products in every mode, no nonstandard
registry for `standard`, and exactly one matching registry for each of
`left`, `right`, and `all`. Stop and return the logs if any job failed, a
registry is absent, or Citlali reports an ambiguous/low-speed leg.

## 4. Replicate with 148670

Only after 150819 passes the preceding checks:

```bash
: > "$SCI_SPLIT_ROOT/jobs/o148670_job_ids.txt"
for mode in standard left right all; do
  job_id=$(sbatch "$SCI_SPLIT_ROOT/jobs/run_o148670_${mode}.sbatch")
  job_id=${job_id%%;*}
  printf 'obsnum=148670 mode=%s job_id=%s\n' "$mode" "$job_id" \
    | tee -a "$SCI_SPLIT_ROOT/jobs/o148670_job_ids.txt"
done
```

Use the same `squeue`, `sacct`, log, and product checks with `150819` replaced
by `148670`. Do not change classification rules or configuration fields after
viewing the 150819 maps.

## 5. Freeze the return

Record the exact code and executable identity, then create the recursive
checksum manifest described in `RETURN_BUNDLE_SPEC.md`:

```bash
cd "$SCI_REPO"
git rev-parse HEAD | tee "$SCI_SPLIT_ROOT/preparation/citlali_commit.txt"
shasum -a 256 "$CITLALI_BIN" \
  | tee "$SCI_SPLIT_ROOT/preparation/citlali_executable_sha256.txt"
```

No map comparison, recentering, timestamp modification, or mitigation decision
is part of this runbook.
