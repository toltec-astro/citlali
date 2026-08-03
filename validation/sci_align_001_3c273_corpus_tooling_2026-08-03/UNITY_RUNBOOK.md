# SCI-ALIGN-001 3C273 corpus owner runbook

This runbook is for the project owner to execute on Unity. Codex did not run
these commands and must not connect to Unity. Every analysis command is
read-only with respect to reduction and raw-data roots and writes only below
the owner-selected output root.

The current owner-provided 3C273 analysis location is:

```text
/work/toltec/wilson/citlali_testing/beammaps/3c273
```

Angle-bracketed values are deliberate owner choices. Do not copy a command
until every placeholder in it has been resolved.

## 1. Environment and exact repository identity

```bash
export SCI_REPO=<UNITY_CITLALI_REPOSITORY>
export SCI_TOOLING_COMMIT=<TOOLING_COMMIT_FROM_CODEX_HANDOFF>
export SCI_ANALYSIS_ROOT=/work/toltec/wilson/citlali_testing/beammaps/3c273
export SCI_RAW_ROOT=<UNITY_RAW_DATA_ROOT>
export SCI_OUTPUT_ROOT=<OWNER_WRITABLE_OUTPUT_ROOT>/sci_align_001_3c273_corpus
export SCI_RUNTIME_CACHE="$SCI_OUTPUT_ROOT/_runtime_cache"

source "$HOME/tolteca/bin/activate"
cd "$SCI_REPO"

test "$(git branch --show-current)" = codex/sci-align-001-3c273-corpus-tooling
test "$(git rev-parse HEAD)" = "$SCI_TOOLING_COMMIT"
git merge-base --is-ancestor \
  a2b37924d612eb175821483523cc94dd233f2fea HEAD
test -z "$(git status --short)"
"$HOME/tolteca/bin/python" --version
```

Stop if any identity check fails. Do not substitute a patched executable,
different branch, or uncommitted copy while retaining the expected identity.

Create only the separate owner output directory:

```bash
mkdir -p "$SCI_OUTPUT_ROOT"
mkdir -p "$SCI_RUNTIME_CACHE/matplotlib" "$SCI_RUNTIME_CACHE/xdg"
test -w "$SCI_OUTPUT_ROOT"

{
  echo "tooling_branch=$(git branch --show-current)"
  echo "tooling_commit=$(git rev-parse HEAD)"
  echo "frozen_predecessor=a2b37924d612eb175821483523cc94dd233f2fea"
  echo "python=$($HOME/tolteca/bin/python --version 2>&1)"
} > "$SCI_OUTPUT_ROOT/execution_identity.txt"
```

The output root must not be inside either the analysis or raw-data root.

## 2. Inventory dry run and inventory

First preview discovery without writing:

```bash
"$HOME/tolteca/bin/python" \
  tools/diagnostics/inventory_sci_align_001_3c273_corpus.py \
  --reduction-root "$SCI_ANALYSIS_ROOT" \
  --raw-root "$SCI_RAW_ROOT" \
  --output "$SCI_OUTPUT_ROOT/inventory" \
  --source-regex '(?i)^3c[ _-]?273$' \
  --dry-run
```

Then create the deterministic inventory:

```bash
"$HOME/tolteca/bin/python" \
  tools/diagnostics/inventory_sci_align_001_3c273_corpus.py \
  --reduction-root "$SCI_ANALYSIS_ROOT" \
  --raw-root "$SCI_RAW_ROOT" \
  --output "$SCI_OUTPUT_ROOT/inventory" \
  --source-regex '(?i)^3c[ _-]?273$'
```

The default inventory hashes small identity and configuration files. Very
large raw signal files are identified by path, size, producer metadata, and a
persistent digest-cache status rather than being repeatedly hashed. If the
owner deliberately wants complete raw SHA-256 values, repeat with
`--hash-large`; retain the digest cache so each physical file is read only
once.

Before analysis, the runner always authenticates the current bytes of every
retained input. A digest supplied by the frozen manifest must match. A large
raw file that was deliberately left `not_hashed_large` during inventory is
hashed once into the owner-output physical-file digest cache, then reused only
while its device/inode/size/mtime identity remains unchanged.

## 3. Inspect candidates, duplicates, and exclusions

```bash
sed -n '1,240p' "$SCI_OUTPUT_ROOT/inventory/candidate_table.md"
sed -n '1,240p' "$SCI_OUTPUT_ROOT/inventory/next_commands.txt"
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/candidate_inventory.csv" | less -S
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/duplicate_reduction_registry.csv" | less -S
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/exclusion_registry.csv" | less -S
"$HOME/tolteca/bin/python" -m json.tool \
  "$SCI_OUTPUT_ROOT/inventory/candidate_inventory.json" | less
```

Check every source spelling, observation, reduction/configuration identity,
software revision, network set, exact per-network integer-second T0 vector,
retained PPS/internal-counter field availability, eligibility class, and
exclusion reason.
Eligibility and canonical proposals are provenance-only; do not inspect a
candidate's inferred timing result to decide whether to select it.

For duplicate observations:

- retain every reduction in the inventory;
- select exactly one core-eligible reduction per eligible observation for
  primary independence;
- keep other reductions for reduction-sensitivity analysis;
- stop for an owner choice when configuration or provenance does not establish
  a defensible canonical authority.

## 4. Freeze the selected manifest

Copy the generated owner-selection template, change only its explicit
selection and owner-note columns, and preserve every candidate row:

```bash
cp "$SCI_OUTPUT_ROOT/inventory/selection_template.csv" \
  "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv"
<OWNER_EDITOR> "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv"
```

Validate and freeze that selection without rereading timing results:

```bash
"$HOME/tolteca/bin/python" \
  tools/diagnostics/inventory_sci_align_001_3c273_corpus.py \
  --freeze-selection "$SCI_OUTPUT_ROOT/inventory/owner_selection.csv" \
  --inventory "$SCI_OUTPUT_ROOT/inventory/candidate_inventory.json" \
  --output "$SCI_OUTPUT_ROOT/inventory/frozen"
```

Inspect the exact freeze and its digest:

```bash
column -s, -t < "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.csv" | less -S
"$HOME/tolteca/bin/python" -m json.tool \
  "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" | less
(cd "$SCI_OUTPUT_ROOT/inventory/frozen" && shasum -a 256 -c SHA256SUMS)
```

Do not continue if a primary observation has zero or multiple selected
reductions, an ineligible row is selected, or a duplicate choice remains
unresolved.

The frozen manifest assigns the chosen reduction `analysis_role=primary` and
automatically retains every other core-eligible reduction of that observation
as `analysis_role=sensitivity`. Those rows are executed by the same serial or
array command but never count as independent observations; they inherit the
primary observation's frozen held-out group. The exact edited owner-selection
file is copied into the frozen directory and covered by its `SHA256SUMS`.

## 5. Preview per-map analysis

```bash
"$HOME/tolteca/bin/python" \
  tools/diagnostics/run_sci_align_001_3c273_beammap.py \
  --manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output-root "$SCI_OUTPUT_ROOT/per_map" \
  --dry-run
```

This must list retained-product reads and owner-output writes only. It must not
list `citlali`, TolTECA reduction setup, `sbatch`, or writes under the source
roots.

For enhanced candidates, the planned outputs must include raw phase and
counter diagnostics: exact T0 by network, PPS transition rows, before/after
internal-counter values, 122/123-row spacing checks, the exact
128-second/15,625-row repeat check, and one-row metadata/counter anomaly
checks. Absence of a retained field must be reported explicitly rather than
inferred.

## 6A. Serial per-map execution

For a small corpus or an initial sentinel, run one selected candidate:

```bash
export SCI_CANDIDATE_ID=<CANDIDATE_ID_FROM_SELECTED_MANIFEST>

OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/run_sci_align_001_3c273_beammap.py \
  --manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --candidate-id "$SCI_CANDIDATE_ID" \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output-root "$SCI_OUTPUT_ROOT/per_map"
```

Run every selected map serially with:

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/run_sci_align_001_3c273_beammap.py \
  --manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output-root "$SCI_OUTPUT_ROOT/per_map"
```

## 6B. Configurable Slurm job array

Generate, but inspect before submitting, a scheduler script. Scheduler policy
is deliberately supplied by the owner rather than embedded in the toolkit:

```bash
mkdir -p "$SCI_OUTPUT_ROOT/slurm_logs"
"$HOME/tolteca/bin/python" \
  tools/diagnostics/generate_sci_align_001_3c273_slurm_array.py \
  --selected-manifest \
    "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --output-root "$SCI_OUTPUT_ROOT/per_map" \
  --output-script "$SCI_OUTPUT_ROOT/run_3c273_array.sh" \
  --command-table "$SCI_OUTPUT_ROOT/run_3c273_array.commands.csv" \
  --python "$HOME/tolteca/bin/python" \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --array-concurrency <MAX_CONCURRENT_MAPS> \
  --sbatch-option cpus-per-task=<CPUS_PER_TASK> \
  --sbatch-option mem=<MEMORY_REQUEST> \
  --sbatch-option partition=<PARTITION> \
  --sbatch-option account=<ACCOUNT> \
  --sbatch-option output="$SCI_OUTPUT_ROOT/slurm_logs/%x_%A_%a.out" \
  --sbatch-option error="$SCI_OUTPUT_ROOT/slurm_logs/%x_%A_%a.err" \
  --sbatch-option time=<TIME_LIMIT>
```

Review all rendered directives, environment lines, manifest paths, and array
bounds:

```bash
sed -n '1,260p' "$SCI_OUTPUT_ROOT/run_3c273_array.sh"
column -s, -t \
  < "$SCI_OUTPUT_ROOT/run_3c273_array.commands.csv" | less -S
```

The owner may then submit it:

```bash
sbatch "$SCI_OUTPUT_ROOT/run_3c273_array.sh"
```

This `sbatch` command is owner-run. Its presence in the runbook is not Codex
authorization to submit or inspect a Unity job.

## 7. Resume or retry

Completed map outputs are reusable only when their candidate, selected
manifest, frozen protocol, tool, and input digests match exactly.

Retry one failed candidate:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/run_sci_align_001_3c273_beammap.py \
  --manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --candidate-id <FAILED_CANDIDATE_ID> \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output-root "$SCI_OUTPUT_ROOT/per_map" \
  --resume
```

Retry the complete manifest while reusing digest-matched successes:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/run_sci_align_001_3c273_beammap.py \
  --manifest "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output-root "$SCI_OUTPUT_ROOT/per_map" \
  --resume
```

Never force reuse after a digest mismatch. Move the stale output aside and run
the candidate again into a fresh owner-output directory.

## 8. Freeze held-out grouping before aggregate timing analysis

This stage reads selected-manifest identity and provenance fields only:

```bash
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/aggregate_sci_align_001_3c273_corpus.py freeze \
  --selected-manifest \
    "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --protocol-template \
    validation/sci_align_001_3c273_corpus_tooling_2026-08-03/frozen_analysis_protocol.json \
  --output "$SCI_OUTPUT_ROOT/aggregate_freeze"
```

Inspect `aggregate_freeze/session_registry.csv` and
`aggregate_freeze/frozen_analysis_protocol.json` before running aggregation.
The complete ordered network-T0 vector is the first candidate
ROACH-initialization session identity. A session grouping is valid only when
that vector or other retained provenance supports it; otherwise the documented
date or observation fallback is used.

## 9. Aggregate compact per-map results

```bash
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
MPLBACKEND=Agg MPLCONFIGDIR="$SCI_RUNTIME_CACHE/matplotlib" \
XDG_CACHE_HOME="$SCI_RUNTIME_CACHE/xdg" \
"$HOME/tolteca/bin/python" \
  tools/diagnostics/aggregate_sci_align_001_3c273_corpus.py run \
  --selected-manifest \
    "$SCI_OUTPUT_ROOT/inventory/frozen/selected_manifest.json" \
  --frozen-protocol \
    "$SCI_OUTPUT_ROOT/aggregate_freeze/frozen_analysis_protocol.json" \
  --map-output-root "$SCI_OUTPUT_ROOT/per_map" \
  --output "$SCI_OUTPUT_ROOT/aggregate"
```

The aggregate report must retain all technically valid timing outliers, state
the independent-group count, expose unsupported held-out levels, compare
network timing against both measured native detector-frame phase and
native-to-assigned-slot residual, report the free coefficient and its interval
relative to `-1`, and translate held-out errors into arcseconds and beam-FWHM
fractions. It must not call a first/second-half change clock drift unless the
raw counters contradict the shared-clock producer account. A stable or
T0-session-predictable native detector-frame phase, or a genuinely held-out
slot predictor, favors later structural native-time or fractional-slot design,
not a fixed physical clock correction. The report must always state that no
production correction is authorized.

## 10. Verify checksums and compactness

Verify each package from its own directory:

```bash
while IFS= read -r -d '' sci_sum; do
  sci_dir=$(dirname "$sci_sum")
  (cd "$sci_dir" && shasum -a 256 -c SHA256SUMS) || exit 1
done < <(find "$SCI_OUTPUT_ROOT" -name SHA256SUMS -type f -print0)
```

Confirm that no retained reduction or raw product was copied into the output:

```bash
find "$SCI_OUTPUT_ROOT" -type f \
  \( -name '*.nc' -o -name '*.fits' -o -name '*.fits.gz' \) -print
find "$SCI_OUTPUT_ROOT" -type f -size +100M -print
```

Both commands must print nothing. Inspect total compact size:

```bash
du -sh "$SCI_OUTPUT_ROOT"
```

Expected disk use is approximately 15 MiB per analyzed map plus no more
than about 100 MiB for manifests, aggregate tables, reports, plots, and logs.
For `N` maps, reserve approximately `(20 * N + 100) MiB`. This is a disk-use
estimate only; the toolkit deliberately makes no execution-time estimate.

## 11. Create the compact transfer archive

```bash
export SCI_ARCHIVE_PARENT=$(dirname "$SCI_OUTPUT_ROOT")
export SCI_ARCHIVE_NAME=sci_align_001_3c273_corpus_bundle.tar.gz
export SCI_RETRIEVAL_UTC=$(date -u '+%Y-%m-%dT%H:%M:%SZ')

cat > "$SCI_OUTPUT_ROOT/RETURN_METADATA.txt" <<EOF
retrieval_timestamp_utc=$SCI_RETRIEVAL_UTC
source_unity_path=$SCI_OUTPUT_ROOT
scope=SCI-ALIGN-001 retained 3C273 beammap corpus diagnostic
question=stability, T0-session dependence, native-phase/slot predictability, within-observation variation, or unpredictability
known_omissions=<KNOWN_OMISSIONS_OR_NONE>
EOF

tar -C "$SCI_ARCHIVE_PARENT" -czf \
  "$SCI_ARCHIVE_PARENT/$SCI_ARCHIVE_NAME" \
  --exclude="$(basename "$SCI_OUTPUT_ROOT")/_runtime_cache" \
  --exclude="$(basename "$SCI_OUTPUT_ROOT")/per_map/_input_digest_cache.json" \
  "$(basename "$SCI_OUTPUT_ROOT")"
(cd "$SCI_ARCHIVE_PARENT" && \
  shasum -a 256 "$SCI_ARCHIVE_NAME" > "$SCI_ARCHIVE_NAME.sha256")
```

Resolve `<KNOWN_OMISSIONS_OR_NONE>` before creating the archive. The metadata
is intentionally outside the deterministic scientific tables but inside the
checksum-bound transfer archive. Runtime plotting caches and the host-specific
physical-file digest cache are deliberately excluded; verified per-candidate
`input_manifest` files retain the portable input digests.

From the owner's local machine, use the required SSH host alias:

```bash
scp unity_toltec:<UNITY_ARCHIVE_PATH>/sci_align_001_3c273_corpus_bundle.tar.gz .
scp unity_toltec:<UNITY_ARCHIVE_PATH>/sci_align_001_3c273_corpus_bundle.tar.gz.sha256 .
shasum -a 256 -c sci_align_001_3c273_corpus_bundle.tar.gz.sha256
```

Do not transfer raw timestreams, retained reduction products, project YAML,
APTs, or Unity logs outside the compact diagnostic list.

## Stop conditions

Stop and return the smallest relevant manifest/log excerpt if:

- source, observation, reduction, config, or software identity is ambiguous;
- duplicate reductions lack a defensible canonical authority;
- a required retained product is missing or malformed;
- raw-row linkage fails or is ambiguous (core analysis may remain eligible,
  but no enhanced claim is allowed);
- common-support construction differs across compared timing models;
- a tool proposes writing below the reduction/raw root;
- a new Citlali reduction or application/configuration change appears needed;
- a checksum, manifest, protocol, or resume binding fails; or
- the independent corpus cannot support the frozen held-out protocol.

Do not weaken a fit cut, remove an unusual timing result, select a duplicate
after viewing timing, or recommend a fixed correction to make the analysis
complete.
