#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_citlali_native_reproducer.sh [--gdb] EXECUTABLE CONFIG

Copy Citlali to node-local storage, verify the copy, and run one low-level
config directly with native resource accounting. Use --gdb to stop on SIGBUS
and print all native thread stacks.
EOF
}

use_gdb=0
if [[ "${1:-}" == "--gdb" ]]; then
  use_gdb=1
  shift
fi
if [[ "$#" -ne 2 ]]; then
  usage >&2
  exit 2
fi

source_executable="$(realpath "$1")"
config="$(realpath "$2")"
scratch_parent="${SLURM_TMPDIR:-${TMPDIR:-/tmp}}"
run_id="${SLURM_JOB_ID:-manual}-$$"
scratch_dir="${scratch_parent%/}/citlali-native-reproducer-${run_id}"
scratch_executable="${scratch_dir}/citlali"

mkdir -p "${scratch_dir}"
cp "${source_executable}" "${scratch_executable}"
chmod 755 "${scratch_executable}"

source_sha="$(sha256sum "${source_executable}" | awk '{print $1}')"
scratch_sha="$(sha256sum "${scratch_executable}" | awk '{print $1}')"
if [[ "${source_sha}" != "${scratch_sha}" ]]; then
  echo "Citlali executable checksum mismatch after local staging" >&2
  exit 1
fi

echo "native_reproducer host=$(hostname)"
echo "native_reproducer slurm_job_id=${SLURM_JOB_ID:-unset}"
echo "native_reproducer slurm_job_nodelist=${SLURM_JOB_NODELIST:-unset}"
echo "native_reproducer slurm_cpus_per_task=${SLURM_CPUS_PER_TASK:-unset}"
echo "native_reproducer source_executable=${source_executable}"
echo "native_reproducer scratch_executable=${scratch_executable}"
echo "native_reproducer executable_sha256=${source_sha}"
echo "native_reproducer config=${config}"
stat "${source_executable}"
stat "${scratch_executable}"
ldd "${scratch_executable}" || true
"${scratch_executable}" --version

ulimit -c unlimited || true
export CITLALI_PROCESS_RESOURCE_DIAGNOSTICS=1
if [[ "${use_gdb}" -eq 1 ]]; then
  exec gdb \
    --batch \
    --return-child-result \
    -ex "set pagination off" \
    -ex "handle SIGBUS stop print nopass" \
    -ex "run" \
    -ex "thread apply all bt full" \
    --args "${scratch_executable}" -l debug "${config}"
fi

exec /usr/bin/time -v "${scratch_executable}" -l debug "${config}"
