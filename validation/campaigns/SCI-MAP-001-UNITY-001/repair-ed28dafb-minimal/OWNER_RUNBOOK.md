# Grant's minimal Unity runbook

These are human commands for Grant to run after confirming the candidate,
operational paths, TolProj site configuration, and canonical inputs. They are
not authorization for Codex or for unattended execution. Every remote command
uses `unity_toltec`.

## 1. Transfer the package

### Unity terminal, before the local transfer

```sh
mkdir -p "$HOME/c2025t/2026-ENG-citlali-MAP"
```

### Local

```sh
LOCAL_PACKAGE=/Users/gwilson/.codex/worktrees/aa31/citlali-refactor/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-minimal
UNITY_RUN_ROOT=/home/toltec_umass_edu/c2025t/2026-ENG-citlali-MAP
UNITY_PACKAGE="$UNITY_RUN_ROOT/repair-ed28dafb-minimal"
rsync -a --checksum --protect-args "$LOCAL_PACKAGE/" \
  "unity_toltec:$UNITY_PACKAGE/"
```

### Unity terminal

Paste these into the already logged-in Unity terminal (and, when appropriate,
the allocated compute-node shell):

```sh
UNITY_RUN_ROOT="$HOME/c2025t/2026-ENG-citlali-MAP"
UNITY_PACKAGE="$UNITY_RUN_ROOT/repair-ed28dafb-minimal"
cd "$UNITY_PACKAGE"
sha256sum -c SHA256SUMS
```

## 2. Create the two TolProj workspaces

The agreed Unity root is `$HOME/c2025t/2026-ENG-citlali-MAP`. Set the two
remaining operational values in Unity. This runbook uses Unity's default
TolProj reduction configuration. Do not reuse an old reduction or copy an old
reduction tree into it.

### Local

No additional local command is needed after the transfer.

### Unity terminal

```sh
TOLPROJ='<approved TolProj executable on Unity>'
GRANT_USER='<Grant Unity/TolProj user>'

set -euo pipefail
point="$UNITY_RUN_ROOT/SCI-MAP-001-POINT-SOURCE"
science="$UNITY_RUN_ROOT/SCI-MAP-001-SCIENCE-SOURCE"
test ! -e "$point"; test ! -e "$science"
"$TOLPROJ" init-test "$UNITY_PACKAGE/tolproj-point-source.json" --root "$UNITY_RUN_ROOT" --user "$GRANT_USER"
"$TOLPROJ" init-test "$UNITY_PACKAGE/tolproj-science-source.json" --root "$UNITY_RUN_ROOT" --user "$GRANT_USER"
test -f "$point/project.yaml"; test -f "$science/project.yaml"
```

Stage only the canonical raw files and matched APT/PPT inputs whose observation
identities appear in `campaign.json`: Point APT 152389; Science APTs 152390 and
152392; Science pointing support 152389, 152391, and 152393. Stop if a required
input is absent or ambiguous. Do not copy an old reduction, and do not change
the selected observations or groups.

## 3. Set up fresh, named reductions

The generic `pointings/02_redu.sh` generated with `--cpus 1` is not a MAP
case: it leaves the default `parallel_policy: omp` in force and produces the
unsupported `omp`/one-thread combination. Preserve that failed directory as a
diagnostic; do not rerun it or edit its generated merged YAML. Create fresh,
named case directories and install exactly one supplied case overlay in each.
The overlay name must be `99_zzz_sci_map_case.yaml`, which loads after TolProj's
submission-runtime overlay. Do not combine `omp` with one CPU/thread.

### Local

No local command is needed; keep the Unity compute-node terminal active.

### Unity terminal

```sh
POINT_PROJECT="$UNITY_RUN_ROOT/SCI-MAP-001-POINT-SOURCE"
SCIENCE_PROJECT="$UNITY_RUN_ROOT/SCI-MAP-001-SCIENCE-SOURCE"
CASE_OVERLAYS="$UNITY_PACKAGE/case-overlays"

# Science pointing support is not an acceptance case. It must be an ordinary
# valid OMP run because its outputs are needed to prepare the Science cases.
# The existing directory must have no completed reduNN output before setup.
test ! -d "$SCIENCE_PROJECT/pointings/reduced/redu00"
"$TOLPROJ" setup-pointing-reductions "$SCIENCE_PROJECT" \
  --refactor --source 1146+399 --pointings-dir pointings --apt-dir apts --cpus 6
"$TOLPROJ" validate-reduction "$SCIENCE_PROJECT/pointings"
"$TOLPROJ" submit-reduction "$SCIENCE_PROJECT/pointings"

# Wait for the support reduction. For every required support obsnum
# 152389/152391/152393, its selected reduNN/raw directory must contain exactly
# one ppt_*.ecsv before continuing. Substitute that same reduNN below.
SCIENCE_POINTING_REDUCTION='reduNN'

# The two Point acceptance cases are fresh sibling directories. Their first
# execution is permitted only while reduced/redu* is absent; do not rerun them,
# because TolProj's generated Point launcher clears its own prior redu* output.
for CASE in P-SEQ P-OMP; do
  test ! -e "$POINT_PROJECT/$CASE"
done
"$TOLPROJ" setup-pointing-reductions "$POINT_PROJECT" \
  --refactor --source 1146+399 --pointings-dir P-SEQ --apt-dir apts --cpus 1
cp "$CASE_OVERLAYS/P-SEQ.yaml" "$POINT_PROJECT/P-SEQ/99_zzz_sci_map_case.yaml"
"$TOLPROJ" setup-pointing-reductions "$POINT_PROJECT" \
  --refactor --source 1146+399 --pointings-dir P-OMP --apt-dir apts --cpus 6
cp "$CASE_OVERLAYS/P-OMP.yaml" "$POINT_PROJECT/P-OMP/99_zzz_sci_map_case.yaml"
for CASE in P-SEQ P-OMP; do
  "$TOLPROJ" validate-reduction "$POINT_PROJECT/$CASE"
  "$TOLPROJ" submit-reduction "$POINT_PROJECT/$CASE"
done

# Prepare a clean sequential Science template, clone it before executing, and
# then prepare a clean OMP template. These copies are unrun TolProj setup
# directories, not copied reductions or returned evidence.  TolProj writes an
# absolute template path into 02_redu.sh, so every clone must have that path
# rewritten before it can be submitted.  Without this rewrite every clone
# changes directory back to $SCIENCE_BASE, bypasses its case overlay, and runs
# the same default reduction.
SCIENCE_BASE="$SCIENCE_PROJECT/$GRANT_USER/NGC4449"
finalize_science_launcher() {
  local case_id="$1"
  local cpus="$2"
  local case_dir="$SCIENCE_PROJECT/$case_id"
  local launcher="$case_dir/02_redu.sh"
  test -f "$launcher"
  perl -0pi -e "s|\\Q$SCIENCE_BASE\\E|$case_dir|g; s|^#SBATCH --job-name=.*$|#SBATCH --job-name=$case_id|m; s|^#SBATCH --cpus-per-task=.*$|#SBATCH --cpus-per-task=$cpus|m" \
    "$launcher"
  grep -Fx "#SBATCH --job-name=$case_id" "$launcher"
  grep -Fx "#SBATCH --cpus-per-task=$cpus" "$launcher"
  grep -Fx "cd $case_dir" "$launcher"
}
for CASE in S-C-SEQ S-E-SEQ S-X-SEQ S-C-OMP S-E-OMP; do
  test ! -e "$SCIENCE_PROJECT/$CASE"
done
"$TOLPROJ" setup-science-reductions "$SCIENCE_PROJECT" \
  --refactor --user "$GRANT_USER" --pointing-reduction "$SCIENCE_POINTING_REDUCTION" \
  --apt-product matched --cpus 1
for CASE in S-C-SEQ S-E-SEQ S-X-SEQ; do
  cp -a "$SCIENCE_BASE" "$SCIENCE_PROJECT/$CASE"
  cp "$CASE_OVERLAYS/$CASE.yaml" "$SCIENCE_PROJECT/$CASE/99_zzz_sci_map_case.yaml"
  finalize_science_launcher "$CASE" 1
done
"$TOLPROJ" setup-science-reductions "$SCIENCE_PROJECT" \
  --refactor --user "$GRANT_USER" --pointing-reduction "$SCIENCE_POINTING_REDUCTION" \
  --apt-product matched --cpus 16
for CASE in S-C-OMP S-E-OMP; do
  cp -a "$SCIENCE_BASE" "$SCIENCE_PROJECT/$CASE"
  cp "$CASE_OVERLAYS/$CASE.yaml" "$SCIENCE_PROJECT/$CASE/99_zzz_sci_map_case.yaml"
  finalize_science_launcher "$CASE" 16
done
for CASE in S-C-SEQ S-E-SEQ S-X-SEQ S-C-OMP S-E-OMP; do
  "$TOLPROJ" validate-reduction "$SCIENCE_PROJECT/$CASE"
  "$TOLPROJ" submit-reduction "$SCIENCE_PROJECT/$CASE"
done
```

If a previous package version has already created one or more named Science
directories, but its reduction failed before producing any output, repair only
their generated launchers before trying again. This keeps the case overlays
and the failed Slurm logs intact; it does not alter the selected observations
or scientific configuration.

### Unity terminal

With `SCIENCE_PROJECT`, `SCIENCE_BASE`, and `finalize_science_launcher` still
defined from the block above, run:

```sh
for CASE in S-C-SEQ S-E-SEQ S-X-SEQ; do
  test -d "$SCIENCE_PROJECT/$CASE"
  finalize_science_launcher "$CASE" 1
done
for CASE in S-C-OMP S-E-OMP; do
  test -d "$SCIENCE_PROJECT/$CASE"
  finalize_science_launcher "$CASE" 16
done
for CASE in S-C-SEQ S-E-SEQ S-X-SEQ S-C-OMP S-E-OMP; do
  "$TOLPROJ" validate-reduction "$SCIENCE_PROJECT/$CASE"
done
```

The `fruit_loops.sig2noise_limit` error from the earlier jobs is expected from
the unmodified template configuration: those launchers changed into
`$SCIENCE_BASE` and never loaded the named case's
`99_zzz_sci_map_case.yaml`, which disables fruit loops. After this repair, do
not change the supplied case overlay. Submit only the corrected named case
directory; do not submit `$SCIENCE_BASE`.

## 4. Prepare and submit the two auxiliary full/all captures

These are separate retained captures, not acceptance cases. `CAP-POINT` starts
from the exact P-SEQ configuration; `CAP-SCIENCE` starts from the exact S-E-SEQ
configuration. In each fresh directory, install the ordinary case overlay
first, then install the full/all output overlay as
`99_zzzz_processed_time_chunk_full.yaml`. This filename sorts after
`99_zzz_sci_map_case.yaml`, so it changes only:

```yaml
timestream.processed_time_chunk.output.enabled: true
timestream.processed_time_chunk.output.mode: full
timestream.processed_time_chunk.output.indices: all
```

### Local

No additional local command is needed after the package transfer.

### Unity terminal

Run this only after the Science pointing-support reduction has produced the
required `ppt_*.ecsv` files and `SCIENCE_POINTING_REDUCTION` is set to its
actual `reduNN` value. It creates fresh capture directories and never changes a
seven-case directory.

```sh
CAPTURE_OVERLAY="$UNITY_PACKAGE/processed-time-chunk-full-overlay.yaml"

# CAP-POINT: exact P-SEQ plus only full/all PTC output.
test ! -e "$POINT_PROJECT/CAP-POINT"
"$TOLPROJ" setup-pointing-reductions "$POINT_PROJECT" \
  --refactor --source 1146+399 --pointings-dir CAP-POINT --apt-dir apts --cpus 1
cp "$CASE_OVERLAYS/P-SEQ.yaml" \
  "$POINT_PROJECT/CAP-POINT/99_zzz_sci_map_case.yaml"
cp "$CAPTURE_OVERLAY" \
  "$POINT_PROJECT/CAP-POINT/99_zzzz_processed_time_chunk_full.yaml"
"$TOLPROJ" validate-reduction "$POINT_PROJECT/CAP-POINT"
"$TOLPROJ" submit-reduction "$POINT_PROJECT/CAP-POINT"

# CAP-SCIENCE: regenerate only the unrun sequential template, then copy it to
# a fresh capture directory. The capture receives exact S-E-SEQ plus full/all.
test ! -e "$SCIENCE_PROJECT/CAP-SCIENCE"
"$TOLPROJ" setup-science-reductions "$SCIENCE_PROJECT" \
  --refactor --user "$GRANT_USER" --pointing-reduction "$SCIENCE_POINTING_REDUCTION" \
  --apt-product matched --cpus 1
cp -a "$SCIENCE_BASE" "$SCIENCE_PROJECT/CAP-SCIENCE"
cp "$CASE_OVERLAYS/S-E-SEQ.yaml" \
  "$SCIENCE_PROJECT/CAP-SCIENCE/99_zzz_sci_map_case.yaml"
cp "$CAPTURE_OVERLAY" \
  "$SCIENCE_PROJECT/CAP-SCIENCE/99_zzzz_processed_time_chunk_full.yaml"
finalize_science_launcher CAP-SCIENCE 1
"$TOLPROJ" validate-reduction "$SCIENCE_PROJECT/CAP-SCIENCE"
"$TOLPROJ" submit-reduction "$SCIENCE_PROJECT/CAP-SCIENCE"
```

Do not install `99_zzzz_processed_time_chunk_full.yaml` in P-SEQ, P-OMP,
S-C-SEQ, S-C-OMP, S-E-SEQ, S-E-OMP, or S-X-SEQ. Do not clean either capture
after it has started or finished. TolProj freezes the selected executable at
submission; use the same ordinary candidate binary for both captures and every
acceptance case.

## 5. Collect ordinary artifacts

After all seven selected reductions finish, set `RESULTS_DIR` to a directory
under `UNITY_RUN_ROOT` containing those seven completed reduction directories,
named by case ID.

### Unity terminal

```sh
tar -C "$UNITY_RUN_ROOT" -czf "$UNITY_RUN_ROOT/SCI-MAP-001-UNITY-001-return.tar.gz" \
  SCI-MAP-001-POINT-SOURCE/project.yaml SCI-MAP-001-SCIENCE-SOURCE/project.yaml \
  "$(basename "$RESULTS_DIR")"
```

### Local

```sh
LOCAL_RETURN_DIR='<local destination for returned MAP artifacts>'
rsync -a --checksum --protect-args \
  "unity_toltec:/home/toltec_umass_edu/c2025t/2026-ENG-citlali-MAP/SCI-MAP-001-UNITY-001-return.tar.gz" \
  "$LOCAL_RETURN_DIR/"
```

Give the archive path and SHA-256 value to the coordinator for interpretation.
This package does not interpret results or close any finding.
