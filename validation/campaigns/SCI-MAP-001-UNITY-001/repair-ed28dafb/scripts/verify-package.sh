#!/usr/bin/env bash
set -euo pipefail

package_root=$(cd "$(dirname "$0")/.." && pwd)
repo_root=$(cd "$package_root/../../../.." && pwd)
python_bin=${PYTHON_BIN:-/Users/gwilson/tolteca/bin/python}
source_root=${SOURCE_ROOT:-$repo_root}
check_root=$(mktemp -d "${TMPDIR:-/tmp}/sci-map-001-package-check.XXXXXX")
trap 'rm -rf "$check_root"' EXIT

cd "$package_root"
sha256sum -c SHA256SUMS
"$python_bin" SCI-MAP-001-analysis.py self-check \
  --campaign campaign.json \
  --product-contracts "$source_root/validation/product_contracts.json" \
  --source-root "$source_root" \
  --output "$check_root/analysis-self-check.json" >/dev/null
"$python_bin" scripts/unity-campaign.py --campaign campaign.json \
  self-check --require-inventory
"$python_bin" -m json.tool campaign.json >/dev/null
"$python_bin" -m json.tool owner-values.schema.json >/dev/null
"$python_bin" -m json.tool owner-values.template.json >/dev/null
"$python_bin" -m json.tool raw-input-manifest.schema.json >/dev/null
"$python_bin" -m json.tool raw-input-manifest.point.template.json >/dev/null
"$python_bin" -m json.tool raw-input-manifest.science.template.json >/dev/null
"$python_bin" -m json.tool result-collection.schema.json >/dev/null
"$python_bin" -m json.tool result-collection.template.json >/dev/null
"$python_bin" -m json.tool sample-ledger-contract.json >/dev/null
PYTHONPYCACHEPREFIX="$check_root/pycache" "$python_bin" -m py_compile \
  SCI-MAP-001-analysis.py scripts/hash-tree.py scripts/unity-campaign.py
bash -n scripts/analysis-job-wrapper.sh scripts/case-job-wrapper.sh \
  scripts/verify-package.sh
printf '%s\n' 'SCI-MAP-001-UNITY-001 package checks passed'
