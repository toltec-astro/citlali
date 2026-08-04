#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 8 ]]; then
  echo "usage: case-job-wrapper.sh CASE_ROOT EVIDENCE_DIR MASTER_BINARY EXPECTED_MASTER_SHA SNAPSHOT EXPECTED_SNAPSHOT_SHA PRE_SUBMIT_INTEGRITY_MANIFEST COMPACT_AUTHORITY_MANIFEST" >&2
  exit 64
fi

case_root=$1
evidence_dir=$2
master_binary=$3
expected_master_sha=$4
snapshot=$5
expected_snapshot_sha=$6
integrity_manifest=$7
compact_manifest=$8

[[ -d "$case_root" ]]
[[ -x "$case_root/02_redu.sh" ]]
[[ -x "$master_binary" ]]
[[ -f "$snapshot" ]]
[[ -f "$integrity_manifest" ]]
[[ -f "$compact_manifest" ]]
[[ "$expected_master_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$expected_snapshot_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$expected_master_sha" == "$expected_snapshot_sha" ]]
[[ -d "$evidence_dir" ]]
[[ ! -L "$evidence_dir" ]]

wrapper_outputs=(
  started-at-utc.txt hostname.txt runtime-environment.txt affinity.txt
  pre-run-sha256.txt integrity-manifest.sha256 compact-authority-manifest.sha256
  stdout.txt stderr.txt exit-status.txt post-run-sha256.txt completed-at-utc.txt
)
for output_name in "${wrapper_outputs[@]}"; do
  output_path="$evidence_dir/$output_name"
  [[ ! -e "$output_path" ]]
  [[ ! -L "$output_path" ]]
done

actual_master_sha=$(sha256sum "$master_binary" | awk '{print $1}')
actual_snapshot_sha=$(sha256sum "$snapshot" | awk '{print $1}')
[[ "$actual_master_sha" == "$expected_master_sha" ]]
[[ "$actual_snapshot_sha" == "$expected_snapshot_sha" ]]
integrity_manifest_sha=$(sha256sum "$integrity_manifest" | awk '{print $1}')
compact_manifest_sha=$(sha256sum "$compact_manifest" | awk '{print $1}')
sha256sum -c "$integrity_manifest"
sha256sum -c "$compact_manifest"

export TOLPROJ_CITLALI_SNAPSHOT=$snapshot
export TOLPROJ_CITLALI_SHA256=$expected_snapshot_sha

date -u +%Y-%m-%dT%H:%M:%SZ > "$evidence_dir/started-at-utc.txt"
hostname > "$evidence_dir/hostname.txt"
env | LC_ALL=C sort | grep -E '^(OMP_|SLURM_|TOLPROJ_)' > "$evidence_dir/runtime-environment.txt" || true
if command -v taskset >/dev/null 2>&1; then
  taskset -pc "$$" > "$evidence_dir/affinity.txt" 2>&1 || true
else
  printf '%s\n' 'taskset unavailable' > "$evidence_dir/affinity.txt"
fi
sha256sum "$master_binary" "$snapshot" "$case_root/02_redu.sh" \
  > "$evidence_dir/pre-run-sha256.txt"
printf '%s  %s\n' "$integrity_manifest_sha" "$integrity_manifest" \
  > "$evidence_dir/integrity-manifest.sha256"
printf '%s  %s\n' "$compact_manifest_sha" "$compact_manifest" \
  > "$evidence_dir/compact-authority-manifest.sha256"

set +e
(
  cd "$case_root"
  /usr/bin/time -v bash ./02_redu.sh
) > "$evidence_dir/stdout.txt" 2> "$evidence_dir/stderr.txt"
rc=$?
set -e

printf '%s\n' "$rc" > "$evidence_dir/exit-status.txt"
sha256sum "$master_binary" "$snapshot" "$case_root/02_redu.sh" \
  > "$evidence_dir/post-run-sha256.txt"
cmp "$evidence_dir/pre-run-sha256.txt" "$evidence_dir/post-run-sha256.txt"
printf '%s  %s\n' "$integrity_manifest_sha" "$integrity_manifest" | sha256sum -c -
printf '%s  %s\n' "$compact_manifest_sha" "$compact_manifest" | sha256sum -c -
sha256sum -c "$integrity_manifest"
sha256sum -c "$compact_manifest"
date -u +%Y-%m-%dT%H:%M:%SZ > "$evidence_dir/completed-at-utc.txt"
exit "$rc"
