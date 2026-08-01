# SCI-MAP-001-UNITY-001 launch checklist

Run all seven cases against clean exact candidate
`ed28dafb37f9113c0d3c95297148157129a90886` and return the complete immutable
bundle and digest. This campaign is not a re-audit and closes none of ALIGN,
CAL, AST, PTC, or VAL.

All state-changing blocks and both emitted plans are single-shot. Preserve any
partial state and stop for owner inspection; do not rerun or overwrite it.

## Local package and owner values

```sh
PACKAGE=/private/tmp/citlali-repair-sci-map-001/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb
VALUES=/private/tmp/SCI-MAP-001-UNITY-001-owner-values.json
PYTHON=/Users/gwilson/tolteca/bin/python
test ! -e "$VALUES"
test ! -L "$VALUES"
install -m 0600 "$PACKAGE/owner-values.template.json" "$VALUES"
"${EDITOR:?set EDITOR}" "$VALUES"                         # OWNER INPUT
"$PYTHON" "$PACKAGE/SCI-MAP-001-analysis.py" \
  validate-owner-values --input "$VALUES"
SOURCE_ROOT=/private/tmp/citlali-repair-sci-map-001 \
  PYTHON_BIN="$PYTHON" "$PACKAGE/scripts/verify-package.sh"
```

Use lexically normalized absolute paths with no trailing slash. The request
root must be absent but its parent must exist. The required disconnected
dependency paths are exactly
`UNITY_SOURCE_CHECKOUT/build_unity_release/_deps/kidscpp-src` and
`dirname(UNITY_SOURCE_CHECKOUT)/tula`. The local retrieval destination must
already be an ordinary, non-symlink directory.

Export the exact completed JSON values; do not use display examples:

```sh
owner_value() {
  "$PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
    "$VALUES" "$1"
}
DEPLOYED_CAMPAIGN_PATH=$(owner_value deployed_campaign_path)
DEPLOYED_OWNER_VALUES="${DEPLOYED_CAMPAIGN_PATH}.owner-values.json"
UNITY_PYTHON=$(owner_value unity_python)
UNITY_SOURCE_CHECKOUT=$(owner_value unity_source_checkout)
REQUEST_ROOT=$(owner_value request_root)
LOCAL_RETRIEVAL_DESTINATION=$(owner_value local_retrieval_destination)
OWNER_VALUES_SHA256=$(shasum -a 256 "$VALUES" | awk '{print $1}')

# Preserve every remote argv element exactly; OpenSSH otherwise joins argv
# into one unquoted remote-shell command string.
unity_ssh() {
  SCI_MAP_REMOTE_COMMAND=$("$PYTHON" -c \
    'import shlex,sys
args=sys.argv[1:]
if any("\n" in value or "\r" in value for value in args):
    raise SystemExit("remote argument contains a line break")
print(shlex.join(args))' "$@") || return
  ssh unity_toltec "$SCI_MAP_REMOTE_COMMAND"
}
```

## Transfer, identity, prepare, and build

```sh
unity_ssh bash -s -- "$DEPLOYED_CAMPAIGN_PATH" "$DEPLOYED_OWNER_VALUES" <<'REMOTE'
set -euo pipefail
destination=$1
owner_values=$2
test "$owner_values" = "$destination.owner-values.json"
test ! -e "$destination"
test ! -L "$destination"
test ! -e "$owner_values"
test ! -L "$owner_values"
mkdir -p "$(dirname "$destination")"
mkdir "$destination"
REMOTE
rsync -a --checksum --protect-args \
  "$PACKAGE/" "unity_toltec:$DEPLOYED_CAMPAIGN_PATH/"
rsync -a --checksum --protect-args --ignore-existing "$VALUES" \
  "unity_toltec:$DEPLOYED_OWNER_VALUES"

unity_ssh bash -s -- \
  "$UNITY_PYTHON" "$DEPLOYED_CAMPAIGN_PATH" \
  "$UNITY_SOURCE_CHECKOUT" "$REQUEST_ROOT" "$DEPLOYED_OWNER_VALUES" \
  "$OWNER_VALUES_SHA256" <<'REMOTE'
set -euo pipefail
python_bin=$1
package=$2
source_root=$3
request_root=$4
values=$5
expected_values_sha=$6
test "$values" = "$package.owner-values.json"
test -f "$values"
test ! -L "$values"
printf '%s  %s\n' "$expected_values_sha" "$values" | sha256sum -c -
driver="$package/scripts/unity-campaign.py"
cd "$package"
sha256sum -c SHA256SUMS
SOURCE_ROOT="$source_root" PYTHON_BIN="$python_bin" scripts/verify-package.sh
"$python_bin" SCI-MAP-001-analysis.py self-check \
  --campaign campaign.json \
  --product-contracts "$source_root/validation/product_contracts.json" \
  --source-root "$source_root"
"$python_bin" "$driver" --campaign "$package/campaign.json" validate \
  --owner-values "$values" --require-existing --expect-request-root absent
"$python_bin" "$driver" --campaign "$package/campaign.json" identity \
  --owner-values "$values" --expect-request-root absent
"$python_bin" "$driver" --campaign "$package/campaign.json" prepare \
  --owner-values "$values"

frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
values="$request_root/owner-values.json"
driver="$frozen/scripts/unity-campaign.py"
"$python_bin" "$driver" --campaign "$frozen/campaign.json" build \
  --owner-values "$values"
REMOTE
```

## OWNER INPUT — raw manifests and ledgers

Before case preparation, supply the two approved exact raw manifests described
by `raw-input-manifest.schema.json`. These are mandatory owner facts; the
templates are not evidence.

```sh
: "${POINT_RAW_MANIFEST:?set exact approved Point raw-manifest path}"
: "${SCIENCE_RAW_MANIFEST:?set exact approved Science raw-manifest path}"

unity_ssh bash -s -- "$UNITY_PYTHON" "$REQUEST_ROOT" \
  "$POINT_RAW_MANIFEST" "$SCIENCE_RAW_MANIFEST" <<'REMOTE'
set -euo pipefail
python_bin=$1
request_root=$2
point_manifest=$3
science_manifest=$4
raw_root="$request_root/raw-input-manifests"
frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
values="$request_root/owner-values.json"
driver="$frozen/scripts/unity-campaign.py"
test -f "$point_manifest"
test ! -L "$point_manifest"
test -f "$science_manifest"
test ! -L "$science_manifest"
install_new() {
  source_path=$1
  destination_path=$2
  test ! -e "$destination_path"
  test ! -L "$destination_path"
  install -m 0444 "$source_path" "$destination_path"
}
install_new "$point_manifest" "$raw_root/P-SEQ.json"
install_new "$point_manifest" "$raw_root/P-OMP.json"
for case_id in S-C-SEQ S-C-OMP S-E-SEQ S-E-OMP S-X-SEQ; do
  install_new "$science_manifest" "$raw_root/$case_id.json"
done
"$python_bin" "$driver" --campaign "$frozen/campaign.json" prepare-cases \
  --owner-values "$values"
REMOTE
```

Before submission, supply all nine exact NPZ ledgers from the approved frozen
producer. They must satisfy `sample-ledger-contract.json` and must not be
derived from final FITS products. Missing producer/source/scan/detector/sample
authority is a stop condition. Set the external source directory, install all
nine without replacement, then emit the plan:

```sh
: "${LEDGER_SOURCE_DIR:?set exact nine-ledger producer-output directory}"

unity_ssh bash -s -- "$REQUEST_ROOT" "$LEDGER_SOURCE_DIR" <<'REMOTE'
set -euo pipefail
request_root=$1
source_dir=$2
destination_dir="$request_root/sample-ledgers"
ledger_names=(
  point-152389-a1100.npz point-152389-a1400.npz point-152389-a2000.npz
  science-152390-a1100.npz science-152390-a1400.npz science-152390-a2000.npz
  science-152392-a1100.npz science-152392-a1400.npz science-152392-a2000.npz
)
for name in "${ledger_names[@]}"; do
  source_path="$source_dir/$name"
  destination_path="$destination_dir/$name"
  test -f "$source_path"
  test ! -L "$source_path"
  test ! -e "$destination_path"
  test ! -L "$destination_path"
done
for name in "${ledger_names[@]}"; do
  install -m 0444 "$source_dir/$name" "$destination_dir/$name"
done
REMOTE

unity_ssh bash -s -- "$UNITY_PYTHON" "$REQUEST_ROOT" <<'REMOTE'
set -euo pipefail
python_bin=$1
request_root=$2
frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
values="$request_root/owner-values.json"
driver="$frozen/scripts/unity-campaign.py"
"$python_bin" "$driver" --campaign "$frozen/campaign.json" emit-submit-plan \
  --owner-values "$values"
sed -n '1,320p' "$request_root/plans/submit-seven-cases.sh"
REMOTE
```

## STOP — owner external-action point

Nothing above submits a job. Review the emitted account, optional Slurm
directives, paths, hashes, cases, and resources. Then the human runs:

```sh
unity_ssh bash -s -- "$REQUEST_ROOT/plans/submit-seven-cases.sh" <<'REMOTE'
set -euo pipefail
plan=$1
test -f "$plan"
test ! -L "$plan"
bash "$plan"
REMOTE
```

All seven cases must exit zero. Afterwards:

```sh
unity_ssh bash -s -- "$UNITY_PYTHON" "$REQUEST_ROOT" <<'REMOTE'
set -euo pipefail
python_bin=$1
request_root=$2
frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
values="$request_root/owner-values.json"
driver="$frozen/scripts/unity-campaign.py"
"$python_bin" "$driver" --campaign "$frozen/campaign.json" \
  build-result-collection --owner-values "$values"
"$python_bin" "$driver" --campaign "$frozen/campaign.json" emit-final-plan \
  --owner-values "$values"
sed -n '1,360p' "$request_root/plans/analyze-freeze-and-retrieve.sh"
REMOTE
```

Review the second plan, then run the bounded analysis and freeze:

```sh
unity_ssh bash -s -- \
  "$REQUEST_ROOT/plans/analyze-freeze-and-retrieve.sh" <<'REMOTE'
set -euo pipefail
plan=$1
test -f "$plan"
test ! -L "$plan"
bash "$plan"
REMOTE
```

The frozen analysis requires contract-derived mapdiag/histogram/PSD families
and writes lossless persisted-identity and seq/OpenMP residual NPZs with a
hashed manifest; aggregate maxima alone are not accepted.

Run the printed `rsync` retrieval command locally, or use the equivalent
fail-closed block below:

```sh
UNITY_BUNDLE=$("$PYTHON" -c \
  'import posixpath,sys
root=posixpath.normpath(sys.argv[1])
print(posixpath.join(posixpath.dirname(root), "SCI-MAP-001-UNITY-001.tar.gz"))' \
  "$REQUEST_ROOT")
UNITY_BUNDLE_DIGEST="$UNITY_BUNDLE.sha256"
test -d "$LOCAL_RETRIEVAL_DESTINATION"
test ! -L "$LOCAL_RETRIEVAL_DESTINATION"
test ! -e "$LOCAL_RETRIEVAL_DESTINATION/SCI-MAP-001-UNITY-001.tar.gz"
test ! -L "$LOCAL_RETRIEVAL_DESTINATION/SCI-MAP-001-UNITY-001.tar.gz"
test ! -e "$LOCAL_RETRIEVAL_DESTINATION/SCI-MAP-001-UNITY-001.tar.gz.sha256"
test ! -L "$LOCAL_RETRIEVAL_DESTINATION/SCI-MAP-001-UNITY-001.tar.gz.sha256"
rsync -a --checksum --protect-args \
  "unity_toltec:$UNITY_BUNDLE" \
  "unity_toltec:$UNITY_BUNDLE_DIGEST" \
  "$LOCAL_RETRIEVAL_DESTINATION/"
cd "$LOCAL_RETRIEVAL_DESTINATION"
shasum -a 256 -c SCI-MAP-001-UNITY-001.tar.gz.sha256
VERIFY_DIR=$(mktemp -d /tmp/SCI-MAP-001-UNITY-001-verify.XXXXXX)
tar -xzf SCI-MAP-001-UNITY-001.tar.gz -C "$VERIFY_DIR"
REQUEST_DIR=$(find "$VERIFY_DIR" -mindepth 1 -maxdepth 1 -type d)
test -n "$REQUEST_DIR"
test "$(printf '%s\n' "$REQUEST_DIR" | wc -l | tr -d ' ')" = 1
cd "$REQUEST_DIR"
shasum -a 256 -c SCI-MAP-001-UNITY-001-MANIFEST.sha256
```

Give the bundle, both digests, exact campaign-package commit, and every
nonzero/gap record to a fresh `codex/reaudit-sci-map-001` task. Do not push and
do not claim external evidence exists before the human completes and returns
this campaign.
