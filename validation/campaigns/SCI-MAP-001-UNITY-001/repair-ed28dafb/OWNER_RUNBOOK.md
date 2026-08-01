# Owner runbook: SCI-MAP-001-UNITY-001

This is the human-run procedure for exact Citlali candidate
`ed28dafb37f9113c0d3c95297148157129a90886`. Package preparation did not
access Unity, submit a job, or create external evidence. Every command that
contacts Unity uses the SSH alias `unity_toltec`.

Stop at the first missing, ambiguous, extra, dirty, mismatched, or unhashable
identity. Do not edit an installed numbered file except the initially empty
mode-specific expert override. Do not subset an input project to hide an extra
observation, array, network, detector, or scan. Do not infer an independent
processed-term ledger from final FITS products.

Every transfer, prepare, build, case-preparation, ledger-installation, plan,
submission, collection, analysis, and freeze block is single-shot. If a block
leaves partial state, preserve it and stop for owner inspection; never rerun
the whole block or replace an existing artifact.

## 1. Frozen request and disposition

- Request: `SCI-MAP-001-UNITY-001`
- Revision: `repair-sha-ed28dafb-2026-08-01`
- Candidate: `ed28dafb37f9113c0d3c95297148157129a90886`, clean only
- Parent: `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- Forbidden ancestor: `02a198cbfb379eaf6ab279c5a3d44ee73ff90435`
- Build: preset `unity_release`, disconnected FetchContent, target
  `citlali_cli`, eight build jobs
- Runtime: `toltec-cpu`; case CPUs are `1/6/1/16/1/16/1` in manifest order
- Repaired outcome: all seven cases, including `S-X-SEQ`, must exit zero
- Numerical comparison: exact inventory/support/WCS plus `atol=2e-8` and
  `rtol=1e-10`; all residuals are retained
- WCS sky bound: `1e-12` degrees

F009 and F010 remain `addressed_pending_reaudit`. F012 remains open until the
returned exact-SHA bundle is independently audited. MAP F013 remains
conditioned and this campaign closes none of ALIGN, CAL, AST, PTC, or VAL.

ALIGN-OD1 through ALIGN-OD8 and ALIGN-C001 are owner-approved at
`4f905f4f353e91847a303f4f3959654f3f03c302`; the canonical identity correction
is `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`, the bounded handoff is
`0309fd48a973a6e7e136224906ac49c02f0171be`, and coordination-ledger HEAD is
`846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. No ALIGN application-repair commit
or re-audit exists. ALIGN implementation remains nonconformant, validation is
in progress, and production remains `existing_use_only`.

## 2. Complete owner values locally

Copy the deliberately unresolved template outside the repository and fill
every value. Empty strings are accepted only for the optional Slurm QoS,
constraint, and reservation fields.

```sh
LOCAL_PACKAGE=/private/tmp/citlali-repair-sci-map-001/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb
LOCAL_OWNER_VALUES=/private/tmp/SCI-MAP-001-UNITY-001-owner-values.json
LOCAL_PYTHON=/Users/gwilson/tolteca/bin/python

test ! -e "$LOCAL_OWNER_VALUES"
test ! -L "$LOCAL_OWNER_VALUES"
install -m 0600 "$LOCAL_PACKAGE/owner-values.template.json" "$LOCAL_OWNER_VALUES"
"${EDITOR:?set EDITOR}" "$LOCAL_OWNER_VALUES"
"$LOCAL_PYTHON" "$LOCAL_PACKAGE/SCI-MAP-001-analysis.py" \
  validate-owner-values --input "$LOCAL_OWNER_VALUES"
SOURCE_ROOT=/private/tmp/citlali-repair-sci-map-001 \
  PYTHON_BIN="$LOCAL_PYTHON" "$LOCAL_PACKAGE/scripts/verify-package.sh"
```

The owner must supply these deployment facts; the package has no defaults for
them:

1. Clean Unity checkout at the exact candidate SHA.
2. A request root that does not exist, whose parent already exists.
3. A disjoint staging directory that does not exist before transfer.
4. Exact Unity Python and installed TolProj executable/site config.
5. Matched Point project/source filter/APT directory selecting only 152389.
6. Matched Science project/source basename/pointing `reduNN` selecting only
   152390 then 152392.
7. Evidence operator, Slurm account, and explicit optional Slurm values.
8. Clean kidscpp and tula source worktrees used by the disconnected build.
   The enforced locations are exactly
   `UNITY_SOURCE_CHECKOUT/build_unity_release/_deps/kidscpp-src` and
   `dirname(UNITY_SOURCE_CHECKOUT)/tula`.
9. An existing ordinary, non-symlink local retrieval destination.

All path values must be lexically normalized absolute paths with no trailing
slash. Derive the shell values used below from the completed JSON rather than
typing a second copy:

```sh
owner_value() {
  "$LOCAL_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]])' \
    "$LOCAL_OWNER_VALUES" "$1"
}
DEPLOYED_CAMPAIGN_PATH=$(owner_value deployed_campaign_path)
DEPLOYED_OWNER_VALUES="${DEPLOYED_CAMPAIGN_PATH}.owner-values.json"
UNITY_PYTHON=$(owner_value unity_python)
UNITY_SOURCE_CHECKOUT=$(owner_value unity_source_checkout)
REQUEST_ROOT=$(owner_value request_root)
LOCAL_RETRIEVAL_DESTINATION=$(owner_value local_retrieval_destination)
OWNER_VALUES_SHA256=$(shasum -a 256 "$LOCAL_OWNER_VALUES" | awk '{print $1}')

# Preserve every remote argv element exactly; OpenSSH otherwise joins argv
# into one unquoted remote-shell command string.
unity_ssh() {
  SCI_MAP_REMOTE_COMMAND=$("$LOCAL_PYTHON" -c \
    'import shlex,sys
args=sys.argv[1:]
if any("\n" in value or "\r" in value for value in args):
    raise SystemExit("remote argument contains a line break")
print(shlex.join(args))' "$@") || return
  ssh unity_toltec "$SCI_MAP_REMOTE_COMMAND"
}
```

## 3. Transfer and prove the standalone package

The staging destination must be absent. The owner-values file is placed beside
the package so package transfer cannot modify it.

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
  "$LOCAL_PACKAGE/" "unity_toltec:$DEPLOYED_CAMPAIGN_PATH/"
rsync -a --checksum --protect-args --ignore-existing "$LOCAL_OWNER_VALUES" \
  "unity_toltec:$DEPLOYED_OWNER_VALUES"
```

Run package, analyzer, owner-value, source, TolProj/TolTECA, and dependency
identity checks before creating the request root:

```sh
unity_ssh bash -s -- \
  "$UNITY_PYTHON" "$DEPLOYED_CAMPAIGN_PATH" \
  "$UNITY_SOURCE_CHECKOUT" "$DEPLOYED_OWNER_VALUES" \
  "$OWNER_VALUES_SHA256" <<'REMOTE'
set -euo pipefail
python_bin=$1
package=$2
source_root=$3
values=$4
expected_values_sha=$5
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
REMOTE
```

Do not proceed if any identity differs or if the candidate/dependency worktrees
are dirty. Conan authority is N/A only because this exact candidate has neither
`conanfile.py` nor `conan.lock`; an unexpected Conan file is a hard stop.

## 4. Initialize the isolated request and build once

`prepare` is the only command that creates the request root. It refuses an
existing path and freezes the package, owner values, governing authorities,
and identity record. All later commands use the request-local frozen package.

```sh
unity_ssh bash -s -- \
  "$UNITY_PYTHON" "$DEPLOYED_CAMPAIGN_PATH" "$REQUEST_ROOT" \
  "$DEPLOYED_OWNER_VALUES" <<'REMOTE'
set -euo pipefail
python_bin=$1
staging=$2
request_root=$3
staging_values=$4
test "$staging_values" = "$staging.owner-values.json"
staging_driver="$staging/scripts/unity-campaign.py"

"$python_bin" "$staging_driver" --campaign "$staging/campaign.json" prepare \
  --owner-values "$staging_values"

frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb"
values="$request_root/owner-values.json"
driver="$frozen/scripts/unity-campaign.py"
"$python_bin" "$driver" --campaign "$frozen/campaign.json" build \
  --owner-values "$values"
REMOTE
```

The build records configure/compile transcripts, cache/options,
`compile_commands.json`, compiler and module identities, dependency SHAs,
binary/version, and SHA-256. Every case later receives a snapshot byte-equal to
the one immutable request-local binary.

## 5. Supply raw-input authority before case preparation

This is the first owner-supplied scientific-evidence boundary. No approved raw
manifest producer is available in this local package, so the owner must supply
two already-generated manifests:

- one Point manifest for 152389; and
- one Science manifest for ordered observations 152390, 152392.

Each must conform to `raw-input-manifest.schema.json` and identify/hash the
frozen producer, complete invocation, every raw/KIDs/APT/calibration/pointing/
projection/sample-rate/FWHM/target source, exact arrays and networks, exact
scan identities/sample counts, exact detector UID/network order, processed
term cardinality, projection identity, map shape, target, and binary64 values.
The producer and source files must be outside all reduction-output roots.

Set these to the two exact Unity files. Do not use the unresolved templates as
evidence:

```sh
# OWNER ACTION REQUIRED: export both exact Unity paths before continuing.
: "${POINT_RAW_MANIFEST:?set exact approved Point raw-manifest path}"
: "${SCIENCE_RAW_MANIFEST:?set exact approved Science raw-manifest path}"
```

Copy identical bytes to the seven fixed request locations, then invoke native
case preparation:

```sh
unity_ssh bash -s -- \
  "$UNITY_PYTHON" "$REQUEST_ROOT" \
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

The driver now executes only the native TolProj `duplicate --refactor` and
mode-specific setup/preflight commands. It preserves exact `RuntimeContext`
ordering and requires nine recognized numbered sources: TolTECA-owned
`40_setup.yaml`, seven mode sources, and generated
`99_zz_tolproj_submission_runtime.yaml`. The expert override must precede the
generated prefix-99 overlay; a tenth source is rejected.

## 6. Supply nine independent processed-term ledgers

Before submitting or viewing any result, the owner must run the same approved,
frozen producer identified by the raw manifests and place nine NPZ files at:

```text
REQUEST_ROOT/sample-ledgers/point-152389-a1100.npz
REQUEST_ROOT/sample-ledgers/point-152389-a1400.npz
REQUEST_ROOT/sample-ledgers/point-152389-a2000.npz
REQUEST_ROOT/sample-ledgers/science-152390-a1100.npz
REQUEST_ROOT/sample-ledgers/science-152390-a1400.npz
REQUEST_ROOT/sample-ledgers/science-152390-a2000.npz
REQUEST_ROOT/sample-ledgers/science-152392-a1100.npz
REQUEST_ROOT/sample-ledgers/science-152392-a1400.npz
REQUEST_ROOT/sample-ledgers/science-152392-a2000.npz
```

Every file must match `sample-ledger-contract.json`, bind the corresponding raw
manifest digest and bundle identity, contain all typed members with exact
dtypes, cover the exact Cartesian scan/detector/sample membership once and in
order, and carry the 64 pinned Boost-MT19937 scan-sign realizations. Missing or
fabricated terms are rejected. Final F010 planes are not an allowed source.

Set the directory containing the nine already-produced ordinary NPZ files.
Do not point this at `REQUEST_ROOT/sample-ledgers`; installation is one-way and
must not overwrite partial or uncertain request state.

```sh
# OWNER ACTION REQUIRED: export the exact Unity producer-output directory.
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
```

When all nine files exist, freeze the reconstruction authority and emit—but do
not execute—the seven-case plan:

```sh
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

Stop if the approved producer, either raw manifest, or any ledger is
unavailable. Return that minimum missing fact as a named evidence gap; do not
change Citlali or invent a value in this campaign.

## 7. STOP — human external-action boundary

Nothing before this point submits a job. The owner must now review all seven
commands, paths, digests, account/QoS/constraint/reservation values, CPU counts,
and the immutable reconstruction manifest.

```sh
# OWNER ACTION: this submits seven repaired-success cases, one at a time.
unity_ssh bash -s -- "$REQUEST_ROOT/plans/submit-seven-cases.sh" <<'REMOTE'
set -euo pipefail
plan=$1
test -f "$plan"
test ! -L "$plan"
bash "$plan"
REMOTE
```

The plan uses `sbatch --wait --parsable`, partition `toltec-cpu`, 64 GiB,
24 hours, and CPUs `1/6/1/16/1/16/1`. Each allocation rechecks the binary,
snapshot, launcher/config/integrity, raw manifests, producer sources, and all
nine ledgers before and after `02_redu.sh`. It records allocation identity,
complete stdout/stderr, GNU-time resources, timestamps, and Slurm accounting.
All seven cases must exit zero. The historical `S-X-SEQ` jobkey is retained
only as identity; its old publication failure must be absent and its
observation/coadd products must complete.

## 8. Freeze collection and emit bounded analysis plan

After all seven cases finish successfully, build the canonical result
collection and emit—but do not execute—the analysis/freeze plan:

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

The driver reconciles each numeric job ID, submission status, partition, CPU
count, Slurm state/exit, submit/start/end timestamps, wrapper exit, immutable
case record, build/source state, and complete log set before it emits analysis.

## 9. Run analysis and freeze the return bundle

Review the second plan, then run it manually:

```sh
# OWNER ACTION: bounded 16-CPU/128-GiB analysis and deterministic bundle freeze.
unity_ssh bash -s -- \
  "$REQUEST_ROOT/plans/analyze-freeze-and-retrieve.sh" <<'REMOTE'
set -euo pipefail
plan=$1
test -f "$plan"
test ! -L "$plan"
bash "$plan"
REMOTE
```

The frozen analyzer rebuilds analysis inputs without reading numerical payloads,
then runs in one `toltec-cpu` allocation. It checks the request-specific product
union, all eight F010 facts and aliases, threshold/provenance identities, all
realizations, full WCS and centered map/coadd relationships, independent
observation/coadd reconstruction, S-C/S-E support-floor recombination,
blank/covariance/false-z/edge characterization, contract-derived
mapdiag/histogram/PSD inventories, and seq/OpenMP comparisons. It retains every
pixel residual in hashed NPZs and invokes the two pinned baseline tools.

Analyzer exit 0 means the frozen checks completed without a recorded
violation. Exit 2 means complete evidence with one or more scientific
nonconformances. Any other exit is an evidence-execution/contract failure and
stops final freeze. None is a conformance or finding-closure decision.

The exact scan-farm gamma bound is not claimed from normalized Unity outputs;
it requires run-produced pre-normalization per-scan accumulator planes and
commit order. The analyzer records that lane as neutral N/A and still performs
the registered external seq/OpenMP topology/WCS/inventory and numerical
comparisons. The exact gamma policy remains covered by the candidate's local
F011 truth suite.

## 10. Retrieve and verify locally

The generated plan prints this retrieval action as a comment. Run it on the
local machine, using only `unity_toltec`:

```sh
UNITY_BUNDLE=$("$LOCAL_PYTHON" -c \
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

Preserve the tarball, outer digest, extracted immutable tree, nonzero/gap
records, exact package commit, and this runbook. Do not push from this
campaign.

## 11. Fresh re-audit handoff

Give a fresh `codex/reaudit-sci-map-001` task:

1. Exact MAP candidate SHA `ed28dafb37f9113c0d3c95297148157129a90886`.
2. Exact campaign-package commit SHA reported by the repair task.
3. Returned tarball and verified outer SHA-256.
4. Extracted request tree and verified inner manifest.
5. Every analysis, baseline, log, Slurm, inventory, raw-authority, and
   processed-ledger record, including any exit-2 nonconformance.
6. The current ALIGN metadata identities and explicit nonclosure boundary.

The re-auditor independently assesses F001–F013, the returned exact-SHA Unity
evidence, and the upstream conditions. This runbook and any successful owner
execution do not declare SCI-MAP-001 conformant and do not close findings.
