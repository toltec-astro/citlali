#!/usr/bin/env bash
#SBATCH --job-name=citlali-spack
#SBATCH --partition=toltec-cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/citlali-spack-%j.out
#SBATCH --error=logs/citlali-spack-%j.err

set -euo pipefail

workspace="${WORKSPACE:-${HOME}/work_toltec/citlali_spack_acceptance}"
repository="${workspace}/citlali"
spack_root="${SPACK_ROOT:-${HOME}/work_toltec/spack-1.2.2}"
spack_python="${SPACK_PYTHON:-/usr/bin/python3.12}"
environment="${repository}/spack/environments/citlali-unity-gcc13"
jobs="${SLURM_CPUS_PER_TASK:-8}"
job_id="${SLURM_JOB_ID:-manual}"
manifest="${repository}/logs/citlali-spack-${job_id}.manifest.txt"

if [[ -z "${EXPECTED_CITLALI_SHA:-}" ]]; then
    echo "EXPECTED_CITLALI_SHA must contain the full source commit" >&2
    exit 2
fi

cd "${repository}"

actual_sha="$(git rev-parse HEAD)"
if [[ "${actual_sha}" != "${EXPECTED_CITLALI_SHA}" ]]; then
    echo "source SHA mismatch: expected ${EXPECTED_CITLALI_SHA}, got ${actual_sha}" >&2
    exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
    echo "source checkout is dirty" >&2
    git status --short >&2
    exit 2
fi

if [[ ! -x "${spack_root}/bin/spack" ]]; then
    echo "missing Spack executable ${spack_root}/bin/spack" >&2
    exit 2
fi
if [[ ! -x "${spack_python}" ]]; then
    echo "missing Spack Python ${spack_python}" >&2
    exit 2
fi

export SPACK_ROOT="${spack_root}"
export SPACK_PYTHON="${spack_python}"
# shellcheck disable=SC1091
source "${SPACK_ROOT}/share/spack/setup-env.sh"

"${SPACK_PYTHON}" tools/build/verify_spack_source_revisions.py \
    --workspace-root "${repository}/build/spack-sources" | tee "${manifest}"

{
    echo "started_at=$(date -Is)"
    echo "host=$(hostname -f)"
    echo "slurm_job_id=${job_id}"
    echo "slurm_cpus=${jobs}"
    echo "citlali_sha=${actual_sha}"
    for dependency in tula_cmake tula kidscpp; do
        echo "${dependency}_sha=$(git -C "${repository}/build/spack-sources/${dependency}" rev-parse HEAD)"
    done
    echo "spack_version=$(spack --version)"
    echo "spack_lock_sha256=$(sha256sum "${environment}/spack.lock" | awk '{print $1}')"
} | tee -a "${manifest}"

spack -e "${environment}" find -cvl
spack -e "${environment}" clean --stage citlali || true
time spack -e "${environment}" install \
    -j "${jobs}" -y --overwrite --show-log-on-error citlali

"${SPACK_PYTHON}" tools/build/run_spack_citlali_dev.py all \
    --profile unity-gcc13 \
    --spack "${SPACK_ROOT}/bin/spack" \
    --spack-python "${SPACK_PYTHON}" \
    --fresh \
    -j "${jobs}"

"${SPACK_PYTHON}" tools/build/test_spack_citlali.py \
    --profile unity-gcc13 \
    --spack "${SPACK_ROOT}/bin/spack" \
    --spack-python "${SPACK_PYTHON}"

prefix="$(spack -e "${environment}" location -i citlali)"
executable="${prefix}/bin/citlali"
{
    echo "package_prefix=${prefix}"
    echo "executable=${executable}"
    echo "executable_sha256=$(sha256sum "${executable}" | awk '{print $1}')"
    echo "completed_at=$(date -Is)"
    echo "version_begin"
    "${executable}" --version
    echo "version_end"
} | tee -a "${manifest}"
