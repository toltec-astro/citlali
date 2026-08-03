# SCI-ALIGN-001 3C273 corpus owner runbook

This is an owner-executed, retained-product diagnostic workflow.  Codex did
not access Unity, run these commands, create a Citlali reduction, or authorize
a correction.  The commands read retained beammap/raw metadata and write only
under the versioned run directory.

## 1. Make the exact tooling available and verify identity

Codex does not push or perform network activity. If the Unity checkout lacks
the handoff commits, the owner may either fetch them after an owner push, or
create and transfer this bundle containing both commits after the known base:

```bash
# On the local source clone:
git bundle create /Users/gwilson/GitHub/citlali-refactor/sci_align_001_followup_8e4fcae2.bundle \
  a2b37924d612eb175821483523cc94dd233f2fea..8e4fcae2ff92078665b8fad992307a609236125a
# Owner transfer to the known Unity checkout (not performed by Codex):
scp /Users/gwilson/GitHub/citlali-refactor/sci_align_001_followup_8e4fcae2.bundle \
  unity_toltec:/work/toltec/citlali_dev/citlali_refactor/
# Then, on Unity from the repository clone:
git fetch /work/toltec/citlali_dev/citlali_refactor/sci_align_001_followup_8e4fcae2.bundle \
  8e4fcae2ff92078665b8fad992307a609236125a:refs/heads/codex/sci-align-001-3c273-corpus-tooling
```

```bash
export SCI_REPO=/work/toltec/citlali_dev/citlali_refactor
export CITLALI_BIN=/work/toltec/citlali_dev/citlali_refactor/build/bin/citlali
export SCI_ANALYSIS_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273
export SCI_RAW_ROOT=/work/toltec/wilson/citlali_testing/beammaps/data
export SCI_RUN_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273/sci_align_001_corpus_run_2026-08-03
export SCI_SCRIPT_ROOT="$SCI_RUN_ROOT/scripts"
export SCI_OUTPUT_ROOT="$SCI_RUN_ROOT/output"
export SCI_RUNTIME_CACHE="$SCI_RUN_ROOT/_runtime_cache"
export SCI_TOOLING_COMMIT=8e4fcae2ff92078665b8fad992307a609236125a
export SCI_PACKAGE=validation/sci_align_001_3c273_corpus_tooling_2026-08-03

source "$HOME/tolteca/bin/activate"
cd "$SCI_REPO"
test "$(git branch --show-current)" = codex/sci-align-001-3c273-corpus-tooling
test "$(git rev-parse HEAD)" = "$SCI_TOOLING_COMMIT"
git merge-base --is-ancestor 6776931e753488b902d178b177815a2762375e5c HEAD
test -z "$(git status --short)"
# CITLALI_BIN is retained in execution identity only; do not test or invoke it.
(cd "$SCI_PACKAGE" && shasum -a 256 -c SHA256SUMS)

mkdir -p "$SCI_SCRIPT_ROOT" "$SCI_OUTPUT_ROOT" \
  "$SCI_RUNTIME_CACHE/matplotlib" "$SCI_RUNTIME_CACHE/xdg"
test -w "$SCI_RUN_ROOT"
{
  echo "tooling_branch=$(git branch --show-current)"
  echo "tooling_commit=$(git rev-parse HEAD)"
  echo "citlali_bin_recorded_only=$CITLALI_BIN"
  echo "python=$(python --version 2>&1)"
} > "$SCI_RUN_ROOT/execution_identity.txt"
```

Stop before analysis if any identity or package checksum check fails.  No
command below invokes `CITLALI_BIN`, `citlali`, or a reduction configuration.

## 2. Inventory the authoritative corpus

The run root is passed as an explicit exclusion.  This prevents any generated
script, output, cache, or archive below it from becoming a candidate on a
repeat run.

```bash
python tools/diagnostics/inventory_sci_align_001_3c273_corpus.py \
  --reduction-root "$SCI_ANALYSIS_ROOT" --raw-root "$SCI_RAW_ROOT" \
  --exclude-path "$SCI_RUN_ROOT" \
  --obsnum-allowlist "$SCI_PACKAGE/authoritative_obsnums_2026-08-03.json" \
  --source-regex '(?i)^3c[ _-]?273$' --output "$SCI_OUTPUT_ROOT/inventory"

column -s, -t < "$SCI_OUTPUT_ROOT/inventory/authoritative_obsnum_status.csv" | less -S
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/network_availability.csv" | less -S
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/out_of_scope_3c273_discovery.csv" | less -S
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/duplicate_reduction_registry.csv" | less -S
```

The allowlist is exactly the owner-supplied 40 ObsNums and is checksum-bound
in the inventory and selected manifest.  `nw10` is structural/nonexistent;
`nw6` is recorded as intermittent; other absences remain visible and reduce
only relevant network support.  Missing retained products and raw metadata
remain deficiencies—never run a new reduction to fill them.

Canonical selection is provenance-only: a sole eligible candidate wins; an
eligible `redu00`/`redu01` pair selects `redu01` primary and retains `redu00`
as sensitivity.  Multiple candidates in either location, another location, or
ambiguous provenance fail closed.  Edit only the resulting template:

```bash
cp "$SCI_OUTPUT_ROOT/inventory/selection_template.csv" \
  "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv"
emacs -nw "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv"
python tools/diagnostics/inventory_sci_align_001_3c273_corpus.py \
  --freeze-selection "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv" \
  --inventory "$SCI_OUTPUT_ROOT/inventory/candidate_inventory.json" \
  --output "$SCI_OUTPUT_ROOT/inventory/frozen"
(cd "$SCI_OUTPUT_ROOT/inventory/frozen" && shasum -a 256 -c SHA256SUMS)
```

## 3. Generate checksum-bound serial and Slurm analysis scripts

Both scripts analyze retained products only.  They bind command table, selected
manifest, allowlist, protocol, and tool bytes by checksum; every candidate has
a unique output directory.  The six Slurm CPUs remain deliberately unused by
numerical libraries (all numerical thread counts are one).

```bash
mkdir -p "$SCI_OUTPUT_ROOT/slurm_logs"
python tools/diagnostics/generate_sci_align_001_3c273_slurm_array.py \
  --selected-manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol "$SCI_PACKAGE/frozen_analysis_protocol.json" \
  --output-root "$SCI_OUTPUT_ROOT/per_map" \
  --command-table "$SCI_SCRIPT_ROOT/3c273.commands.csv" \
  --output-script "$SCI_SCRIPT_ROOT/run_3c273_array.sh" \
  --serial-script "$SCI_SCRIPT_ROOT/run_3c273_serial.sh" \
  --python python --array-concurrency 8 \
  --sbatch-option output="$SCI_OUTPUT_ROOT/slurm_logs/%x_%A_%a.out" \
  --sbatch-option error="$SCI_OUTPUT_ROOT/slurm_logs/%x_%A_%a.err"

sed -n '1,240p' "$SCI_SCRIPT_ROOT/run_3c273_array.sh"
bash -n "$SCI_SCRIPT_ROOT/run_3c273_array.sh"
bash -n "$SCI_SCRIPT_ROOT/run_3c273_serial.sh"
```

The rendered array has exactly these owner defaults: `48:00:00`, `64G`, six
CPUs, one node, one task, `toltec-cpu`, `--parsable`, and `%8`; it has no
account directive.  Scheduler variables appear at the top and are editable.
Run one route, only after inspection:

```bash
"$SCI_SCRIPT_ROOT/run_3c273_serial.sh"
# Or, owner-run only:
sbatch "$SCI_SCRIPT_ROOT/run_3c273_array.sh"
```

## 4. Freeze the aggregate plan and aggregate compact results

```bash
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
python tools/diagnostics/aggregate_sci_align_001_3c273_corpus.py freeze \
  --selected-manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol-template "$SCI_PACKAGE/frozen_analysis_protocol.json" \
  --output "$SCI_OUTPUT_ROOT/aggregate_freeze"

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg \
MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
python tools/diagnostics/aggregate_sci_align_001_3c273_corpus.py run \
  --selected-manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --frozen-protocol "$SCI_OUTPUT_ROOT/aggregate_freeze/frozen_analysis_protocol.json" \
  --map-output-root "$SCI_OUTPUT_ROOT/per_map" --output "$SCI_OUTPUT_ROOT/aggregate"
```

The aggregate reports no science acceptance threshold, timing eligibility cut,
or correction decision.  It reports support counts (observations, networks,
scans, detectors), phase/session structure, and descriptive uncertainties.
The Stage-A lineage evidence begins with delivered `D[n]/Ts[n]`; it cannot
exclude an upstream FPGA metadata-to-integration association error.

For every available network the returned bundle includes mismatch denominators,
counts/rates, signed and absolute tick residuals, locations/adjacent geometry,
and field-unavailable status in `pps_time_increment_occurrence.csv` and
`raw_pps_time_increment_anomalies.csv`.  `nw9_timing_sensitivity.csv` contains
the nw9 estimate/residual/rate plus all-network versus leave-nw9-out difference
and uncertainty; its corpus association is descriptive, not causal.  No
row mask or repair is authorized where metadata semantics are ambiguous.

## 5. Verify and return the compact bundle

```bash
while IFS= read -r -d '' sci_sum; do
  sci_dir=$(dirname "$sci_sum")
  (cd "$sci_dir" && shasum -a 256 -c SHA256SUMS) || exit 1
done < <(find "$SCI_RUN_ROOT" -name SHA256SUMS -type f -print0)

python -m json.tool \
  "$SCI_OUTPUT_ROOT/aggregate/known_omissions.json" | less
find "$SCI_RUN_ROOT" -type f \( -name '*.nc' -o -name '*.fits' -o -name '*.fits.gz' \) -print

export SCI_ARCHIVE_PARENT=$(dirname "$SCI_RUN_ROOT")
export SCI_ARCHIVE_NAME=sci_align_001_3c273_corpus_run_2026-08-03.tar.gz
tar -C "$SCI_ARCHIVE_PARENT" -czf "$SCI_ARCHIVE_PARENT/$SCI_ARCHIVE_NAME" \
  --exclude="$(basename "$SCI_RUN_ROOT")/_runtime_cache" \
  "$(basename "$SCI_RUN_ROOT")"
(cd "$SCI_ARCHIVE_PARENT" && shasum -a 256 "$SCI_ARCHIVE_NAME" > "$SCI_ARCHIVE_NAME.sha256")
```

`known_omissions.json` is generated from inventory and task evidence.  It
enumerates listed ObsNum/product/raw-linkage deficiencies, network absence with
the nw10/nw6 distinctions, unresolved duplicates, task failures, unavailable
metadata, and intentionally skipped sensitivity duplicates; it also records
that raw timestreams and retained beammap products are intentionally absent
from the compact archive.

Stop and return the smallest relevant manifest/log excerpt if identity,
checksum, provenance, duplicate authority, required retained-product, or
raw-linkage checks fail; if a whole observation cannot be analyzed; or if a
command proposes reduction, application/configuration edits, source-root
writes, a timing correction, push, merge, or rebase.
