#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "$0")/.." && pwd)
repo_root=$(cd "$package_root/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/Users/gwilson/tolteca/bin/python}
source_root=${SOURCE_ROOT:-$repo_root}
check_root=$(mktemp -d "${TMPDIR:-/tmp}/sci-map-001-ed2-check.XXXXXX")
trap 'rm -rf "$check_root"' EXIT
export PYTHONPYCACHEPREFIX="$check_root/pycache"

cd "$package_root"

# The ED1 stop return is immutable history.  Its three-file inventory and the
# checksum file itself are frozen independently of the active ED2 inventory.
printf '%s  %s\n' \
  293b21ec162d407496c22db0b022cc512e8e4ebc8ac0c6d15765e8bbd844cc60 \
  SHA256SUMS | sha256sum -c -
sha256sum -c SHA256SUMS

# SHA256SUMS.ed2 is exhaustive and deliberately excludes only itself.
sha256sum -c SHA256SUMS.ed2

while IFS= read -r json_path; do
  "$python_bin" -m json.tool "$json_path" >/dev/null
done < <(find . -type f -name '*.json' -not -path './__pycache__/*' | LC_ALL=C sort)

"$python_bin" -m py_compile \
  SCI-MAP-001-analysis.py \
  scripts/compact-evidence.py scripts/ed2-capture.py scripts/hash-tree.py \
  scripts/unity-campaign.py \
  tests/test_compact_evidence.py tests/test_ed2_capture.py \
  tests/test_package_contract.py

bash -n scripts/analysis-job-wrapper.sh scripts/case-job-wrapper.sh \
  scripts/verify-package.sh

"$python_bin" scripts/ed2-capture.py self-check >/dev/null
"$python_bin" scripts/compact-evidence.py self-check >/dev/null
"$python_bin" SCI-MAP-001-analysis.py self-check \
  --campaign campaign.json \
  --product-contracts "$source_root/validation/product_contracts.json" \
  --source-root "$source_root" \
  --output "$check_root/analysis-self-check.json" >/dev/null
"$python_bin" scripts/unity-campaign.py --campaign campaign.json \
  self-check --require-inventory

"$python_bin" -m unittest discover \
  -s tests -p 'test_*.py'

# Package programs are file-only preparation/verification tools.  They may
# emit human commands but must not contain executable network or Slurm calls.
if rg -n --glob '*.py' \
  'subprocess\.(run|Popen|call|check_call|check_output)\([^\n]*(ssh|scp|rsync|sbatch|srun)' \
  scripts SCI-MAP-001-analysis.py; then
  printf '%s\n' 'forbidden programmatic network/Slurm invocation found' >&2
  exit 1
fi

printf '%s\n' 'SCI-MAP-001-UNITY-001 MAP-UNITY-ED2 package checks passed'
