# Unity owner runbook: one-pass split-direction Beammap products

This runbook is executed by the owner on Unity. Codex does not access Unity.
It creates new diagnostic roots and never edits raw data, retained reductions,
or the completed replay configurations used as authorities.

One config with `beammap.direction_mode: all` launches one Citlali reduction.
After shared calibration/RTC/PTC processing, each scan fills the standard map
buffer and one directional buffer. The reduction writes standard, `_left`, and
`_right` products into one ordinary output tree. Run ObsNum 150819 first; do
not submit 148670 until 150819 passes the retained-product checks.

## 1. Environment and authorities

Load the normal TolTEC environment first so that `python` has `yaml` and the
usual science dependencies. Then define:

```bash
export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export CITLALI_BIN="$SCI_REPO/build/bin/citlali"
export SCI_ANALYSIS_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273
export SCI_CAMPAIGN_ROOT="$SCI_ANALYSIS_ROOT/sci_align_001_replay_campaign_16_2026-08-05_retry1"
export SCI_REPLAY_148670="$SCI_ANALYSIS_ROOT/sci_align_001_reproduction_148670_2026-08-03_retry1"
export SCI_SPLIT_ROOT="$SCI_ANALYSIS_ROOT/sci_align_001_split_direction_beammap_onepass_2026-08-06"

cd "$SCI_REPO"
test -z "$(git status --short)"
test -x "$CITLALI_BIN"
command -v python
python -c 'import yaml; print("python=", __import__("sys").executable); print("yaml=", yaml.__version__)'
test ! -e "$SCI_SPLIT_ROOT"
```

Resolve one completed direct Citlali config per observation:

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

Stop if a config is absent or ambiguous, the executable is missing, the Git
worktree is dirty, or the new root already exists.

## 2. Render configs and Slurm scripts

This preparation sets `beammap.direction_mode`, a new `runtime.output_dir`, and
the explicit compatibility value
`runtime.crop_detector_to_telescope_support: false`. The last value preserves
the historical behavior of the source replay configs while satisfying the
new typed-config requirement. It verifies detector grouping, launches nothing,
and does not create either observation output root.

```bash
python - "$SCI_SOURCE_CONFIG_150819" "$SCI_SOURCE_CONFIG_148670" \
  "$SCI_SPLIT_ROOT" "$CITLALI_BIN" <<'PY'
import copy
import hashlib
import json
import stat
import sys
from pathlib import Path

import yaml

source_by_obs = {150819: Path(sys.argv[1]), 148670: Path(sys.argv[2])}
root = Path(sys.argv[3])
citlali = Path(sys.argv[4])
if root.exists():
    raise SystemExit(f"refusing existing preparation root: {root}")
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
    "schema": "sci-align-001-split-direction-preparation-v4",
    "execution_model": "one shared timestream pass; three detector map buffers",
    "compatibility_policy": {
        "runtime.crop_detector_to_telescope_support": False,
        "reason": "preserve the source replay's historical non-cropping behavior",
    },
    "source_products_modified": False,
    "citlali": {"path": str(citlali), "sha256": sha256(citlali)},
    "runs": [],
}
for obsnum in (150819, 148670):
    source = source_by_obs[obsnum].resolve()
    config = copy.deepcopy(yaml.safe_load(source.read_text()))
    if not isinstance(config, dict):
        raise SystemExit(f"config root is not a mapping: {source}")
    beammap = config.setdefault("beammap", {})
    runtime = config.setdefault("runtime", {})
    mapmaking = config.get("mapmaking", {})
    if not all(isinstance(node, dict) for node in (beammap, runtime, mapmaking)):
        raise SystemExit(f"beammap/runtime/mapmaking is not a mapping: {source}")
    if mapmaking.get("grouping") != "detector":
        raise SystemExit(
            f"direction_mode=all requires mapmaking.grouping=detector: {source}")
    output_root = root / f"o{obsnum}"
    config_path = root / "preparation" / f"citlali_o{obsnum}_all.yaml"
    beammap["direction_mode"] = "all"
    runtime["output_dir"] = str(output_root)
    runtime["crop_detector_to_telescope_support"] = False
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    script = root / "jobs" / f"run_o{obsnum}_all.sbatch"
    script.write_text("\n".join([
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name=sci-align-{obsnum}-all",
        "#SBATCH --time=48:00:00",
        "#SBATCH --mem=192G",
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
        "requested_mode": "all",
        "realized_products": ["standard", "left", "right"],
        "source_config": str(source),
        "source_config_sha256": sha256(source),
        "rendered_config": str(config_path),
        "rendered_config_sha256": sha256(config_path),
        "output_root": str(output_root),
        "submit_script": str(script),
        "requested_memory": "192G",
        "crop_detector_to_telescope_support": False,
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
    print(row["observation_number"], row["requested_mode"],
          row["realized_products"], row["output_root"],
          row["requested_memory"])
PY

test ! -e "$SCI_SPLIT_ROOT/o150819"
test ! -e "$SCI_SPLIT_ROOT/o148670"
```

The first run requests 192 GB conservatively because `all` retains three
detector-map buffers. Record `MaxRSS`; later requests may be lowered from
evidence. This is one reduction, not three. Relative to the previous 2h40m
standard jobs, allow roughly 3–5 hours after allocation for the additional map
accumulation, normalization, fitting, and output. The 48-hour limit is a guard,
not an expected runtime.

## 3. Run and verify 150819

```bash
job_id=$(sbatch "$SCI_SPLIT_ROOT/jobs/run_o150819_all.sbatch")
job_id=${job_id%%;*}
printf 'obsnum=150819 mode=all job_id=%s\n' "$job_id" \
  | tee "$SCI_SPLIT_ROOT/jobs/o150819_job_id.txt"

squeue -j "$job_id" -o '%.18i %.28j %.8T %.10M %.4C %R'
tail -f "$SCI_SPLIT_ROOT"/jobs/sci-align-150819-all_"${job_id}".out
```

After it leaves `squeue`:

```bash
sacct -X -j "$job_id" \
  --format=JobID,JobName%30,State,ExitCode,Elapsed,MaxRSS,Start,End \
  | tee "$SCI_SPLIT_ROOT/jobs/o150819_sacct.txt"

find "$SCI_SPLIT_ROOT/o150819" -type f \
  \( -name '*_citlali*.fits' -o -name 'apt_*_citlali*.ecsv' \
     -o -name 'beammap_direction_scan_registry*.csv' \
     -o -name '*provenance*.yaml' \) \
  -print | sort

test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'beammap_direction_scan_registry_all.csv' | wc -l)" -eq 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali.ecsv' | wc -l)" -ge 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali_left.ecsv' | wc -l)" -ge 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali_right.ecsv' | wc -l)" -ge 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali_fit_qc.ecsv' | wc -l)" -ge 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali_left_fit_qc.ecsv' | wc -l)" -ge 1
test "$(find "$SCI_SPLIT_ROOT/o150819" -type f \
  -name 'apt_*_citlali_right_fit_qc.ecsv' | wc -l)" -ge 1
```

Require Slurm `COMPLETED`/`0:0`, one `_all` registry, and standard/left/right
map, APT, and fit-QC siblings in the same reduction tree. Search stdout and
stderr for unexpected error-level messages. Preserve and stop on failure; do
not retry in place.

The retained job 62656042 from commit 9730f0e2 is a known contract failure,
despite Slurm `COMPLETED`/`0:0`: YAML metadata aliasing caused the standard
APT write to use the `_right` identity. Preserve that root. Reproduce 150819
only in a new root with the metadata-isolation follow-up commit and rebuilt
executable; never fill its missing standard APT from an older reduction.

### Render the visual review product

Run this only after the reduction and product checks above have completed. It
is read-only with respect to the reduction. The default makes one page per
detector; `--detectors-per-page 2` is the only denser supported layout.

```bash
export SCI_VIZ_ROOT="$SCI_SPLIT_ROOT/review_o150819_a1100"
export SCI_VIZ_CACHE="$SCI_SPLIT_ROOT/_visualization_cache"
test ! -e "$SCI_VIZ_ROOT"
mkdir -p "$SCI_VIZ_CACHE/matplotlib" "$SCI_VIZ_CACHE/xdg"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_VIZ_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_VIZ_CACHE/xdg" \
python "$SCI_REPO/tools/diagnostics/render_sci_align_001_split_direction_beammaps.py" \
  --reduction-root "$SCI_SPLIT_ROOT/o150819" \
  --output "$SCI_VIZ_ROOT" \
  --array a1100 \
  --max-detectors 100 \
  --detectors-per-page 1

(cd "$SCI_VIZ_ROOT" && shasum -a 256 -c SHA256SUMS)
pdfinfo "$SCI_VIZ_ROOT/split_direction_beammaps_o150819_a1100.pdf" \
  | rg '^(Pages|Page size|File size)'
```

If an independently fixed hero UID table already exists, add
`--hero-selection /absolute/path/to/hero_uids.ecsv`. Do not construct that
table after looking at directional displacement. The automatic selection is
network-balanced and uses only standard-APT quality and S/N.

### Test the retained downstream transfer kernel

This step launches no Citlali reduction and does not alter the completed
products. Use the exact `selected_detectors.ecsv` frozen by the visual review.
The job fails closed if any selected standard/left/right detector lacks its
retained kernel plane or if a signal/kernel WCS identity differs.

For the completed retry2 root and its already frozen raw-frame review:

```bash
export SCI_SPLIT_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_split_direction_beammap_onepass_2026-08-06_retry2
export SCI_VIZ_ROOT="$SCI_SPLIT_ROOT/review_o150819_a1100_raw_frame_v2"
export SCI_TRANSFER_ROOT="$SCI_SPLIT_ROOT/transfer_o150819_a1100_v1"
export SCI_TRANSFER_CACHE="$SCI_SPLIT_ROOT/_transfer_cache"

test -f "$SCI_VIZ_ROOT/selected_detectors.ecsv"
test ! -e "$SCI_TRANSFER_ROOT"
mkdir -p "$SCI_TRANSFER_CACHE/matplotlib" "$SCI_TRANSFER_CACHE/xdg"
```

Submit the read-only diagnostic through Slurm so it survives a terminal
disconnect:

```bash
cat > "$SCI_SPLIT_ROOT/jobs/run_o150819_transfer_v1.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sci-align-150819-transfer
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=toltec-cpu
#SBATCH --output=$SCI_SPLIT_ROOT/jobs/%x_%j.out
#SBATCH --error=$SCI_SPLIT_ROOT/jobs/%x_%j.err
#SBATCH --parsable
set -euo pipefail
cd $SCI_REPO
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=$SCI_TRANSFER_CACHE/matplotlib \
XDG_CACHE_HOME=$SCI_TRANSFER_CACHE/xdg \
python $SCI_REPO/tools/diagnostics/analyze_sci_align_001_split_direction_transfer.py \
  --reduction-root $SCI_SPLIT_ROOT/o150819 \
  --selection $SCI_VIZ_ROOT/selected_detectors.ecsv \
  --output $SCI_TRANSFER_ROOT \
  --array a1100 \
  --minimum-clean-detectors 30
EOF

bash -n "$SCI_SPLIT_ROOT/jobs/run_o150819_transfer_v1.sbatch"
transfer_job_id=$(sbatch "$SCI_SPLIT_ROOT/jobs/run_o150819_transfer_v1.sbatch")
transfer_job_id=${transfer_job_id%%;*}
printf 'obsnum=150819 transfer_job_id=%s\n' "$transfer_job_id" \
  | tee "$SCI_SPLIT_ROOT/jobs/o150819_transfer_job_id.txt"
```

After the job leaves `squeue`:

```bash
sacct -X -j "$transfer_job_id" \
  --format=JobID,JobName%32,State,ExitCode,Elapsed,MaxRSS,Start,End

(cd "$SCI_TRANSFER_ROOT" && shasum -a 256 -c SHA256SUMS)
python - "$SCI_TRANSFER_ROOT/diagnostic_decision.json" <<'PY'
import json, sys
doc = json.load(open(sys.argv[1]))
for key in (
    "classification",
    "downstream_filtering_artifact_disposition",
    "signal_nuclear_right_minus_left_arcsec",
    "signal_core_plus_jet_right_minus_left_arcsec",
    "kernel_right_minus_left_arcsec",
):
    print(f"{key}={doc.get(key)}")
PY

pdfinfo "$SCI_TRANSFER_ROOT/split_direction_transfer_o150819_a1100.pdf" \
  | rg '^(Pages|Page size|File size)'
```

Require Slurm `COMPLETED`/`0:0`, checksum success, and two PDF pages. The
decision JSON is scoped only to filtering/cleaning/mapmaking downstream of
synthetic-kernel creation; it cannot exclude an FPGA/raw metadata-to-integration
association error.

### Audit the naive-map sampling against the retained full PTC

The mapmaker-dependence control and the repaired full-PTC replay are complete.
Use the latter only as sample/pointing authority and the former as the
standard/left/right map authority. This read-only job selects UID 199, derives
direction from the retained `az_phys` trajectory, and replays the exact naive
nearest-pixel support rule. It does not modify either reduction.

```bash
export SCI_NAIVE_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_split_direction_beammap_onepass_naive_2026-08-07
export SCI_PTC_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_naive_full_ptc_singlepass_150819_2026-08-07_retry1
export SCI_PTC_TOD="$SCI_PTC_ROOT/o150819/redu00/150819/raw/full_ptc/toltec_commissioning_beammap_150819_ptc_timestream.nc"
export SCI_SAMPLING_ROOT="$SCI_PTC_ROOT/review_uid199_naive_sampling_v1"
export SCI_SAMPLING_CACHE="$SCI_PTC_ROOT/_sampling_cache"

test -f "$SCI_PTC_TOD"
test ! -e "$SCI_SAMPLING_ROOT"
mkdir -p "$SCI_PTC_ROOT/jobs" "$SCI_SAMPLING_CACHE/matplotlib" "$SCI_SAMPLING_CACHE/xdg"

cat > "$SCI_PTC_ROOT/jobs/run_o150819_uid199_sampling_v1.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sci-align-150819-sampling
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=toltec-cpu
#SBATCH --output=$SCI_PTC_ROOT/jobs/%x_%j.out
#SBATCH --error=$SCI_PTC_ROOT/jobs/%x_%j.err
#SBATCH --parsable
set -euo pipefail
cd $SCI_REPO
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=$SCI_SAMPLING_CACHE/matplotlib \
XDG_CACHE_HOME=$SCI_SAMPLING_CACHE/xdg \
python $SCI_REPO/tools/diagnostics/analyze_sci_align_001_ptc_sampling.py \
  --ptc-tod $SCI_PTC_TOD \
  --map-reduction-root $SCI_NAIVE_ROOT/o150819 \
  --output $SCI_SAMPLING_ROOT \
  --uid 199 --array a1100 --half-width-arcsec 25
EOF

bash -n "$SCI_PTC_ROOT/jobs/run_o150819_uid199_sampling_v1.sbatch"
sampling_job_id=$(sbatch "$SCI_PTC_ROOT/jobs/run_o150819_uid199_sampling_v1.sbatch")
sampling_job_id=${sampling_job_id%%;*}
printf 'obsnum=150819 uid=199 sampling_job_id=%s\n' "$sampling_job_id" \
  | tee "$SCI_PTC_ROOT/jobs/o150819_uid199_sampling_job_id.txt"
```

After completion, require `COMPLETED`/`0:0`, successful checksums, and two PDF
pages. Interpret black support pixels as agreement, orange as accepted-PTC-hit
only, magenta as map-support only, and white as neither. Only the black/white
pattern supports ordinary sparse nearest-pixel coverage; colored disagreement
requires investigation of replay identity or retained-product semantics.

```bash
sacct -X -j "$sampling_job_id" \
  --format=JobID,JobName%34,State,ExitCode,Elapsed,MaxRSS,Start,End
(cd "$SCI_SAMPLING_ROOT" && shasum -a 256 -c SHA256SUMS)
pdfinfo "$SCI_SAMPLING_ROOT/ptc_sampling_audit_o150819_uid199.pdf" \
  | rg '^(Pages|Page size|File size)'
sed -n '1,100p' "$SCI_SAMPLING_ROOT/mode_support_metrics.ecsv"
```

The completed first audit is a pointing-continuity and geometric screen. Its
map comparison crosses two replays and therefore does not have same-run
signal flags, scan weights, or map-accumulation pointing. Do not interpret its
Jaccard values as an exact support failure.

### Join same-run selected PTC samples to the full-PTC pointing

This bounded follow-up uses the map reduction's retained UID 199 detector TOD
and PTC diagnostics for final-iteration signal, flags, and per-scan weights.
It joins them to the full-PTC pointing by the documented one-based original
scan identity and requires the repaired full PTC scan metadata to describe
each exact appended chunk. Products created before the 2026-08-08 append-bound
repair incorrectly preserve the first scan's length for every row and are not
accepted scan-bound authority. Only retained scans are tested. A
selected-hit-only pixel is exact disagreement; a map-only pixel is explicitly
untested because an unretained scan may support it.

```bash
export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export SCI_NAIVE_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_split_direction_beammap_onepass_naive_2026-08-07
export SCI_PTC_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_naive_full_ptc_singlepass_150819_2026-08-07_retry1
export SCI_PTC_TOD="$SCI_PTC_ROOT/o150819/redu00/150819/raw/full_ptc/toltec_commissioning_beammap_150819_ptc_timestream.nc"
export SCI_SELECTED_JOIN_ROOT="$SCI_PTC_ROOT/review_uid199_naive_selected_join_v1"
export SCI_SELECTED_JOIN_CACHE="$SCI_PTC_ROOT/_selected_join_cache"

test -f "$SCI_PTC_TOD"
test ! -e "$SCI_SELECTED_JOIN_ROOT"
mkdir -p "$SCI_PTC_ROOT/jobs" \
  "$SCI_SELECTED_JOIN_CACHE/matplotlib" "$SCI_SELECTED_JOIN_CACHE/xdg"

cat > "$SCI_PTC_ROOT/jobs/run_o150819_uid199_selected_join_v1.sbatch" <<EOF
#!/usr/bin/env bash
#SBATCH --job-name=sci-align-150819-selected-join
#SBATCH --time=04:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=toltec-cpu
#SBATCH --output=$SCI_PTC_ROOT/jobs/%x_%j.out
#SBATCH --error=$SCI_PTC_ROOT/jobs/%x_%j.err
#SBATCH --parsable
set -euo pipefail
cd $SCI_REPO
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR=$SCI_SELECTED_JOIN_CACHE/matplotlib \
XDG_CACHE_HOME=$SCI_SELECTED_JOIN_CACHE/xdg \
python $SCI_REPO/tools/diagnostics/analyze_sci_align_001_selected_sampling_join.py \
  --full-ptc-tod $SCI_PTC_TOD \
  --map-reduction-root $SCI_NAIVE_ROOT/o150819 \
  --output $SCI_SELECTED_JOIN_ROOT \
  --uid 199 --array a1100 --half-width-arcsec 25
EOF

bash -n "$SCI_PTC_ROOT/jobs/run_o150819_uid199_selected_join_v1.sbatch"
selected_join_job_id=$(sbatch \
  "$SCI_PTC_ROOT/jobs/run_o150819_uid199_selected_join_v1.sbatch")
selected_join_job_id=${selected_join_job_id%%;*}
printf 'obsnum=150819 uid=199 selected_join_job_id=%s\n' \
  "$selected_join_job_id" \
  | tee "$SCI_PTC_ROOT/jobs/o150819_uid199_selected_join_job_id.txt"
```

After completion, require `COMPLETED`/`0:0`, successful checksums, and two PDF
pages. The decisive per-mode fields are
`selected_hit_supported_fraction` and `selected_hit_only_pixels`.

```bash
sacct -X -j "$selected_join_job_id" \
  --format=JobID,JobName%36,State,ExitCode,Elapsed,MaxRSS,Start,End
(cd "$SCI_SELECTED_JOIN_ROOT" && shasum -a 256 -c SHA256SUMS)
pdfinfo "$SCI_SELECTED_JOIN_ROOT/selected_sampling_join_o150819_uid199.pdf" \
  | rg '^(Pages|Page size|File size)'
sed -n '1,100p' "$SCI_SELECTED_JOIN_ROOT/mode_selected_support.ecsv"
```

### Completed mapmaker-dependence control: naive mapmaking

After preserving the retained-kernel result, prepare a new ObsNum 150819
`direction_mode: all` reduction from the exact accepted retry2 rendered config.
Change only `mapmaking.method` from its retained value to `naive` and
`runtime.output_dir` to a new, absent root. Preserve detector grouping,
direction mode, all RTC/PTC/filtering settings, inputs, APT authority,
`runtime.crop_detector_to_telescope_support: false`, executable, and commit.
Before submission, write a manifest containing source/rendered config hashes
and a machine-checked recursive diff allowlisting exactly those two YAML
paths. Do not edit or reuse the accepted retry2 output root. The completed
control satisfied that contract and is the map authority used by the sampling
audit above.

## 4. Replicate with 148670

Only after 150819 passes:

```bash
job_id_148670=$(sbatch "$SCI_SPLIT_ROOT/jobs/run_o148670_all.sbatch")
job_id_148670=${job_id_148670%%;*}
printf 'obsnum=148670 mode=all job_id=%s\n' "$job_id_148670" \
  | tee "$SCI_SPLIT_ROOT/jobs/o148670_job_id.txt"
```

Use the same `squeue`, `tail`, `sacct`, and product checks with `150819`
replaced by `148670`. Do not change the classifier or any other configuration
after viewing 150819.

## 5. Freeze the return

```bash
cd "$SCI_REPO"
git rev-parse HEAD | tee "$SCI_SPLIT_ROOT/preparation/citlali_commit.txt"
shasum -a 256 "$CITLALI_BIN" \
  | tee "$SCI_SPLIT_ROOT/preparation/citlali_executable_sha256.txt"
```

Then create the recursive manifest and tarball in `RETURN_BUNDLE_SPEC.md`.
Timestamp modification and mitigation decisions remain outside this runbook.
