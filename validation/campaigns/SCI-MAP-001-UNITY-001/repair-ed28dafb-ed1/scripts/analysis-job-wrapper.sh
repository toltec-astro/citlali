#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 13 ]]; then
  echo "usage: analysis-job-wrapper.sh PYTHON ANALYZER ANALYZER_SHA CAMPAIGN INPUTS OUTPUT_DIR EVIDENCE_DIR SOURCE_ROOT PRODUCT_CONTRACTS PROFILE_REGISTRY ACCEPTED_RUNS POINT_CONTRACT_ID SCIENCE_CONTRACT_ID" >&2
  exit 64
fi

python_bin=$1
analyzer=$2
expected_analyzer_sha=$3
campaign=$4
inputs=$5
output_dir=$6
evidence_dir=$7
source_root=$8
product_contracts=$9
profile_registry=${10}
accepted_runs=${11}
point_contract=${12}
science_contract=${13}

[[ -x "$python_bin" ]]
[[ -f "$analyzer" ]]
[[ -f "$campaign" ]]
[[ -f "$inputs" ]]
[[ -f "$product_contracts" ]]
[[ -f "$profile_registry" ]]
[[ -f "$accepted_runs" ]]
[[ -d "$source_root/.git" || -f "$source_root/.git" ]]
[[ "$expected_analyzer_sha" =~ ^[0-9a-f]{64}$ ]]
[[ "$point_contract" == sci-map-001-point-products-v1 ]]
[[ "$science_contract" == sci-map-001-science-products-v1 ]]
[[ "$(sha256sum "$analyzer" | awk '{print $1}')" == "$expected_analyzer_sha" ]]
[[ ! -e "$output_dir" ]]
[[ ! -e "$evidence_dir" ]]
mkdir -p "$evidence_dir"

date -u +%Y-%m-%dT%H:%M:%SZ > "$evidence_dir/started-at-utc.txt"
hostname > "$evidence_dir/hostname.txt"
env | LC_ALL=C sort | grep -E '^(OMP_|SLURM_)' > "$evidence_dir/runtime-environment.txt" || true
if command -v taskset >/dev/null 2>&1; then
  taskset -pc "$$" > "$evidence_dir/affinity.txt" 2>&1 || true
else
  printf '%s\n' 'taskset unavailable' > "$evidence_dir/affinity.txt"
fi
"$python_bin" -VV > "$evidence_dir/python-version.txt" 2>&1
"$python_bin" -c 'import astropy, numpy, scipy; print("astropy", astropy.__version__); print("numpy", numpy.__version__); print("scipy", scipy.__version__)' > "$evidence_dir/module-versions.txt"
sha256sum "$analyzer" "$campaign" "$inputs" "$product_contracts" \
  "$profile_registry" "$accepted_runs" > "$evidence_dir/pre-run-sha256.txt"

set +e
/usr/bin/time -v "$python_bin" "$analyzer" run \
  --inputs "$inputs" --output "$output_dir" --source-root "$source_root" \
  --python "$python_bin" --product-contracts "$product_contracts" \
  --profile-registry "$profile_registry" --accepted-runs "$accepted_runs" \
  --point-contract "$point_contract" --science-contract "$science_contract" \
  > "$evidence_dir/stdout.txt" 2> "$evidence_dir/stderr.txt"
rc=$?
set -e

printf '%s\n' "$rc" > "$evidence_dir/exit-status.txt"
sha256sum "$analyzer" "$campaign" "$inputs" "$product_contracts" \
  "$profile_registry" "$accepted_runs" > "$evidence_dir/post-run-sha256.txt"
cmp "$evidence_dir/pre-run-sha256.txt" "$evidence_dir/post-run-sha256.txt"
date -u +%Y-%m-%dT%H:%M:%SZ > "$evidence_dir/completed-at-utc.txt"
exit "$rc"
