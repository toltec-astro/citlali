# Owner runbook: SCI-MAP-001-UNITY-001 / MAP-UNITY-ED2

This is a future human-run procedure, not present authorization. The
coordinator must first accept the implementation handback and name its exact
commit. Do not push, contact Unity, fill operational values, or run any block
merely because this file exists.

The unchanged application is
`ed28dafb37f9113c0d3c95297148157129a90886`, tree
`cf75c36557178f351fb62781108a6f4b41b19225`. The request remains
`SCI-MAP-001-UNITY-001`; the revision is
`repair-sha-ed28dafb-ed1-2026-08-02`. CAP-POINT and CAP-SCIENCE are auxiliary
primitive captures, not replacements for the unchanged seven acceptance
cases. Every command that reaches Unity uses `unity_toltec`.

Stop on the first missing, ambiguous, extra, dirty, changed, unhashable, or
out-of-cap fact. Preserve partial state; do not overwrite, retry broadly,
choose another raw transfer route, build a second binary, edit the candidate,
change a case/gate, infer a missing primitive from FITS, or delete retained
evidence.

## 1. Coordinator-approved push and separate Unity checkout

Only after the coordinator names `HANDOFF_COMMIT`, the owner may run this
local block from the app-supplied worktree. The branch and commit must match
exactly; no force push is allowed.

Local:

```sh
LOCAL_REPO=/Users/gwilson/.codex/worktrees/aa31/citlali-refactor
# Use this exact value only after the coordinator accepts this handback.
HANDOFF_COMMIT=49e21ea90cd663370aa797f1295e8ee65ad4341c

git -C "$LOCAL_REPO" status --short
test "$(git -C "$LOCAL_REPO" branch --show-current)" = codex/map-unity-ed1
test "$(git -C "$LOCAL_REPO" rev-parse HEAD)" = "$HANDOFF_COMMIT"
git -C "$LOCAL_REPO" push origin codex/map-unity-ed1
```

Unity (after the owner has logged in and allocated the intended interactive
environment): create a separate package checkout if one does not already
exist. Never switch or modify the candidate checkout while obtaining the
package.

```sh
# First-time package checkout only. This must be distinct from the candidate
# source checkout used to build Citlali.
HANDOFF_COMMIT=49e21ea90cd663370aa797f1295e8ee65ad4341c
PACKAGE_CHECKOUT="$HOME/c2025t/2026-ENG-citlali-MAP/citlali-refactor-ed2-package"
PACKAGE_ORIGIN=git@github.com:toltec-astro/citlali.git

test ! -e "$PACKAGE_CHECKOUT"
git clone "$PACKAGE_ORIGIN" "$PACKAGE_CHECKOUT"
```

For either that new checkout or a pre-existing separate package checkout, run
the following clean detached-checkout verification. Do not reuse a directory
with uncommitted files.

```sh
HANDOFF_COMMIT=49e21ea90cd663370aa797f1295e8ee65ad4341c
PACKAGE_CHECKOUT="${PACKAGE_CHECKOUT:-$HOME/c2025t/2026-ENG-citlali-MAP/citlali-refactor-ed2-package}"
commit="$HANDOFF_COMMIT"
checkout="$PACKAGE_CHECKOUT"
test -d "$checkout/.git"
test -z "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)"
git -C "$checkout" fetch origin codex/map-unity-ed1
test "$(git -C "$checkout" rev-parse origin/codex/map-unity-ed1)" = "$commit"
git -C "$checkout" checkout --detach "$commit"
test "$(git -C "$checkout" rev-parse HEAD)" = "$commit"
test -z "$(git -C "$checkout" status --porcelain=v1 --untracked-files=all)"
```

The deployed package is exactly:

```text
PACKAGE_CHECKOUT/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1
```

Do not use or copy an old reduction as the deployed package.

## 2. Resolve owner facts once

Copy `owner-values.template.json` outside Git on Unity and fill every `null`
with a verified operational fact. Empty strings are permitted only for the
three optional Slurm fields. In particular:

- `unity_host_alias` remains `unity_toltec`;
- `canonical_raw_root` is the owner-verified `/work/toltec`;
- `point_source_project` and `science_source_project` are exactly the two
  project paths below `unity_test_root` named by the JSON specs;
- capture roots are exactly
  `request_root/captures/CAP-POINT` and
  `request_root/captures/CAP-SCIENCE`;
- the compact root is exactly `request_root/compact-groups`;
- binary/build/version paths are exactly the request-local paths enforced by
  the schema; and
- the four realized-config paths name the fixed and full/all merged YAML
  files that the commands below will create.

The two raw-selection files are explicit ordered basename lists. Point uses
schema `sci-map-001-raw-selection-v1`, capture `CAP-POINT`, and records covering
only 152389. Science uses capture `CAP-SCIENCE` and records covering each of
152389/152390/152391/152392/152393; 152389/152391/152393 remain pointing
support only. Every basename contains its observation as a numeric token.

The single `authority_selection` file may be an
`sci-map-001-authority-selection-set-v1` object with exact `CAP-POINT` and
`CAP-SCIENCE` `records` arrays. Point contains only
`apt_152389_matched.ecsv`. Science contains, in order,
`apt_152390_matched.ecsv`, `apt_152392_matched.ecsv`, then exactly one selected
`ppt_*.ecsv` for 152389, 152391, and 152393. The stager independently rejects
another matching PPT for any support observation.

Unity (in the already-open Unity shell):

```sh
OWNER_VALUES='<owner-selected path outside Git>/SCI-MAP-001-UNITY-001-owner-values.json'
PACKAGE="$PACKAGE_CHECKOUT/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1"
UNITY_PYTHON='<exact value copied into owner-values.json>'

package="$PACKAGE"
values="$OWNER_VALUES"
python_bin="$UNITY_PYTHON"
test -f "$values"; test ! -L "$values"
cd "$package"
sha256sum -c SHA256SUMS
sha256sum -c SHA256SUMS.ed2
source_root=$("$python_bin" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["unity_source_checkout"])' "$values")
SOURCE_ROOT="$source_root" PYTHON_BIN="$python_bin" scripts/verify-package.sh
"$python_bin" SCI-MAP-001-analysis.py validate-owner-values \
  --input "$values" --require-existing
"$python_bin" scripts/unity-campaign.py --campaign campaign.json validate \
  --owner-values "$values" --require-existing --expect-request-root absent
"$python_bin" scripts/unity-campaign.py --campaign campaign.json identity \
  --owner-values "$values" --expect-request-root absent
```

If the candidate or dependency worktree is dirty, a path differs, or an owner
fact is not known, stop. The local reference
`/Users/gwilson/work_toltec/local_data/citlali-validation/v1` is discovery-only.
No `/work/toltec/citlali-validation/v1` counterpart is assumed, and neither
tree may be wholesale-rsynced into this lane.

## 3. Prepare the request and compile exactly once

Unity (as a separately invoked preparation step):

```sh
package="$PACKAGE"; values="$OWNER_VALUES"; python_bin="$UNITY_PYTHON"
driver="$package/scripts/unity-campaign.py"
"$python_bin" "$driver" --campaign "$package/campaign.json" prepare \
  --owner-values "$values"
request_root=$(
  "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["request_root"])' "$values")
frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1"
"$python_bin" "$frozen/scripts/unity-campaign.py" \
  --campaign "$frozen/campaign.json" build --owner-values "$request_root/owner-values.json"
```

This is the only compilation. It records source/dependency/compiler/build
inputs, version output, and the one ordinary executable SHA-256. A second,
instrumented, or rebuilt executable is a hard stop.

For later blocks, start a human Unity shell, set only the exact request root
and frozen Python selected above, and derive every other operational value
from the immutable frozen owner file:

```sh
REQUEST_ROOT='<exact frozen request_root>'
UNITY_PYTHON='<exact frozen unity_python>'
FROZEN="$REQUEST_ROOT/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1"
FROZEN_VALUES="$REQUEST_ROOT/owner-values.json"
CAPTURE_TOOL="$FROZEN/scripts/ed2-capture.py"
COMPACT_TOOL="$FROZEN/scripts/compact-evidence.py"
DRIVER="$FROZEN/scripts/unity-campaign.py"
value() {
  "$UNITY_PYTHON" -c \
    'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' \
    "$FROZEN_VALUES" "$1"
}
test "$(value request_root)" = "$REQUEST_ROOT"
test "$(value unity_python)" = "$UNITY_PYTHON"
TOLPROJ=$(value tolproj_executable)
TOLPROJ_SITE_CONFIG=$(value tolproj_site_config)
UNITY_SOURCE_CHECKOUT=$(value unity_source_checkout)
POINT_SOURCE_PROJECT=$(value point_source_project)
SCIENCE_SOURCE_PROJECT=$(value science_source_project)
CAPTURE_POINT_ROOT=$(value capture_point_root)
CAPTURE_SCIENCE_ROOT=$(value capture_science_root)
COMPACT_EVIDENCE_ROOT=$(value compact_evidence_root)
CANDIDATE_BINARY=$(value candidate_binary)
CANDIDATE_BUILD_MANIFEST=$(value candidate_build_manifest)
CANDIDATE_VERSION_OUTPUT=$(value candidate_version_output)
CAPTURE_POINT_FIXED_REALIZED_CONFIG=$(value capture_point_fixed_realized_config)
CAPTURE_POINT_REALIZED_CONFIG=$(value capture_point_realized_config)
CAPTURE_SCIENCE_FIXED_REALIZED_CONFIG=$(value capture_science_fixed_realized_config)
CAPTURE_SCIENCE_REALIZED_CONFIG=$(value capture_science_realized_config)
RESOURCE_FILESYSTEM_ROOT=$(value resource_filesystem_root)
RESOURCE_RECORDS="$COMPACT_EVIDENCE_ROOT/_campaign/resource-records"

resource_projection() {
  stage=$1; source=$2
  safe=${stage//:/-}
  output="$RESOURCE_RECORDS/$safe.projection.json"
  "$UNITY_PYTHON" "$CAPTURE_TOOL" resource-projection \
    --stage "$stage" --source "$source" --output "$output"
  printf '%s\n' "$output"
}
resource_record() {
  stage=$1; phase=$2; projection=${3-}
  safe=${stage//:/-}
  projection_args=()
  if test "$phase" = pre; then
    test -n "$projection"
    projection_args=(--projection-authority "$projection")
  else
    test -z "$projection"
  fi
  "$UNITY_PYTHON" "$CAPTURE_TOOL" resource-record \
    --stage "$stage" --phase "$phase" \
    "${projection_args[@]}" \
    --filesystem-root "$RESOURCE_FILESYSTEM_ROOT" \
    --governed-root "$POINT_SOURCE_PROJECT" \
    --governed-root "$SCIENCE_SOURCE_PROJECT" \
    --governed-root "$CAPTURE_POINT_ROOT" \
    --governed-root "$CAPTURE_SCIENCE_ROOT" \
    --governed-root "$COMPACT_EVIDENCE_ROOT" \
    --inventory "$RESOURCE_RECORDS/$safe.$phase.inventory.json" \
    --record "$RESOURCE_RECORDS/$safe.$phase.json"
}
```

## 4. Mandatory no-submit preflight before any project, staging, duplication, or configuration write

Unity (record only; the owner must inspect the generated JSON and make the
next-step decision before proceeding to section 5):

```sh
PREPARE_PROJECTION=$(resource_projection PREPARE-STAGING "$FROZEN/resource-report.json")
resource_record PREPARE-STAGING pre "$PREPARE_PROJECTION"
```

## 5. Create the two lightweight projects and stage inputs

Both project paths must be absent before `init-test`.

Unity (as a separately invoked staging step, and only after the
`PREPARE-STAGING` pre-record has been reviewed):

```sh
request_root="$REQUEST_ROOT"
values="$request_root/owner-values.json"
frozen="$request_root/frozen-package-tree/validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb-ed1"
value() { python_bin=$1; key=$2; "$python_bin" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))[sys.argv[2]])' "$values" "$key"; }
python_bin=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["unity_python"])' "$values")
tolproj=$(value "$python_bin" tolproj_executable)
site=$(value "$python_bin" tolproj_site_config)
test_root=$(value "$python_bin" unity_test_root)
operator=$(value "$python_bin" evidence_operator)
point_project=$(value "$python_bin" point_source_project)
science_project=$(value "$python_bin" science_source_project)

test ! -e "$point_project"; test ! -L "$point_project"
test ! -e "$science_project"; test ! -L "$science_project"
"$tolproj" init-test "$frozen/tolproj-point-source.json" \
  --root "$test_root" --user "$operator" --config "$site"
"$tolproj" init-test "$frozen/tolproj-science-source.json" \
  --root "$test_root" --user "$operator" --config "$site"
test -d "$point_project/data"; test ! -L "$point_project/data"
test -d "$science_project/data"; test ! -L "$science_project/data"
test -z "$(find "$point_project/data" -mindepth 1 -maxdepth 1 -print -quit)"
test -z "$(find "$science_project/data" -mindepth 1 -maxdepth 1 -print -quit)"

canonical=$(value "$python_bin" canonical_raw_root)
raw_point=$(value "$python_bin" point_raw_selection)
raw_science=$(value "$python_bin" science_raw_selection)
authorities=$(value "$python_bin" authority_selection)
authority_root=$(value "$python_bin" authority_source_root)
tool="$frozen/scripts/ed2-capture.py"

"$python_bin" "$tool" raw-manifest --capture-id CAP-POINT \
  --canonical-root "$canonical" --selection "$raw_point" \
  --output "$point_project/logs/raw-link-manifest.json"
"$python_bin" "$tool" stage-raw \
  --manifest "$point_project/logs/raw-link-manifest.json" \
  --destination "$point_project/data" \
  --output "$point_project/logs/raw-link-staging.json"

"$python_bin" "$tool" raw-manifest --capture-id CAP-SCIENCE \
  --canonical-root "$canonical" --selection "$raw_science" \
  --output "$science_project/logs/raw-link-manifest.json"
"$python_bin" "$tool" stage-raw \
  --manifest "$science_project/logs/raw-link-manifest.json" \
  --destination "$science_project/data" \
  --output "$science_project/logs/raw-link-staging.json"

mkdir "$science_project/pointings/reduced"
mkdir "$science_project/pointings/reduced/redu00"
"$python_bin" "$tool" stage-authorities --capture-id CAP-POINT \
  --authority-root "$authority_root" --selection "$authorities" \
  --apt-destination "$point_project/apts" \
  --output "$point_project/logs/authority-staging.json"
"$python_bin" "$tool" stage-authorities --capture-id CAP-SCIENCE \
  --authority-root "$authority_root" --selection "$authorities" \
  --apt-destination "$science_project/apts" \
  --ppt-destination "$science_project/pointings/reduced/redu00" \
  --output "$science_project/logs/authority-staging.json"

"$tolproj" duplicate "$point_project" CAP-POINT \
  --destination-root "$request_root/captures" --config "$site" --refactor
"$tolproj" duplicate "$science_project" CAP-SCIENCE \
  --destination-root "$request_root/captures" --config "$site" --refactor
```

Only individual absolute raw-file symlinks and the named copied ECSV files are
allowed. From the first successful link-staging command onward, `tolproj
copy-raw` is absolutely prohibited. A missing canonical file is a stop, not
permission to copy from NESE or an old reduction.

## 6. Configure the two captures and prove the complete diff

Use Point source `1146+399`; use Science source `NGC4449` with the three
support PPTs in `redu00`. Both capture setup commands use the same ordinary
binary and one CPU, matching fixed reference cases P-SEQ and S-E-SEQ.

```sh
# HUMAN on Unity, after reviewing every derived path.
"$TOLPROJ" setup-pointing-reductions "$CAPTURE_POINT_ROOT" \
  --config "$TOLPROJ_SITE_CONFIG" --refactor --source 1146+399 \
  --pointings-dir capture-pointing --apt-dir apts --cpus 1 --time 24:00:00 --mem 64G
"$TOLPROJ" setup-science-reductions "$CAPTURE_SCIENCE_ROOT" \
  --config "$TOLPROJ_SITE_CONFIG" --refactor --user capture \
  --pointing-reduction redu00 --apt-product matched --cpus 1 --time 24:00:00 --mem 64G
```

Resolve exactly one Point and one Science `02_redu.sh`; ambiguity is a stop.
Set `POINT_REDUCTION_DIR` and `SCIENCE_REDUCTION_DIR` to their parent
directories. Then, for each mode:

1. materialize P-SEQ or S-E-SEQ with `SCI-MAP-001-analysis.py materialize-case`;
2. install it as the mode expert override;
3. snapshot all nine numbered YAML files and merge the fixed config;
4. run `ed2-capture.py capture-overlay` to add only enabled/full/all;
5. install that generated overlay, inventory the live nine-file order, and
   merge the capture config; and
6. run `config-proof` against the two inventories and full merged YAMLs.

The exact commands are:

```sh
prepare_capture_config() {
  mode=$1; case_id=$2; redu=$3; fixed_yaml=$4; capture_yaml=$5; record_dir=$6
  expert="99_${mode}_expert_overrides.yaml"
  test "$mode" != point || expert=99_pointing_expert_overrides.yaml
  mkdir "$record_dir"
  "$UNITY_PYTHON" "$FROZEN/SCI-MAP-001-analysis.py" materialize-case \
    --campaign "$FROZEN/campaign.json" --case-id "$case_id" \
    --owner-values "$FROZEN_VALUES" --output "$record_dir/fixed-overlay.yaml"
  chmod u+w "$redu/$expert"
  cp "$record_dir/fixed-overlay.yaml" "$redu/$expert"
  chmod 0444 "$redu/$expert"
  mkdir "$record_dir/fixed-numbered"
  for name in $("$UNITY_PYTHON" -c \
    'import importlib.util,sys; p=sys.argv[1]; s=importlib.util.spec_from_file_location("c",p); m=importlib.util.module_from_spec(s); s.loader.exec_module(m); print(*m.NUMBERED[sys.argv[2]])' \
    "$CAPTURE_TOOL" "$mode"); do
    install -m 0444 "$redu/$name" "$record_dir/fixed-numbered/$name"
  done
  "$UNITY_PYTHON" "$CAPTURE_TOOL" config-inventory --mode "$mode" \
    --numbered-dir "$record_dir/fixed-numbered" \
    --output "$record_dir/fixed-config-inventory.json"
  "$UNITY_PYTHON" "$UNITY_SOURCE_CHECKOUT/tools/config/tolteca_mode_kit.py" merge \
    --mode "$mode" --mode-dir "$record_dir/fixed-numbered" \
    --manifest "$UNITY_SOURCE_CHECKOUT/config/tolteca/v2/manifest.yaml" \
    --leaf-contract "$UNITY_SOURCE_CHECKOUT/tools/config/config_leaf_contract_resolved.json" \
    --yaml-out "$fixed_yaml" --json-out "$record_dir/fixed-merge-report.json"
  "$UNITY_PYTHON" "$CAPTURE_TOOL" capture-overlay \
    --reference-overlay "$record_dir/fixed-overlay.yaml" \
    --candidate-binary "$CANDIDATE_BINARY" \
    --output "$record_dir/capture-overlay.yaml"
  chmod u+w "$redu/$expert"
  cp "$record_dir/capture-overlay.yaml" "$redu/$expert"
  chmod 0444 "$redu/$expert"
  "$UNITY_PYTHON" "$CAPTURE_TOOL" config-inventory --mode "$mode" \
    --numbered-dir "$redu" --output "$record_dir/capture-config-inventory.json"
  "$UNITY_PYTHON" "$UNITY_SOURCE_CHECKOUT/tools/config/tolteca_mode_kit.py" merge \
    --mode "$mode" --mode-dir "$redu" \
    --manifest "$UNITY_SOURCE_CHECKOUT/config/tolteca/v2/manifest.yaml" \
    --leaf-contract "$UNITY_SOURCE_CHECKOUT/tools/config/config_leaf_contract_resolved.json" \
    --yaml-out "$capture_yaml" --json-out "$record_dir/capture-merge-report.json"
  capture_id=CAP-POINT; test "$mode" = point || capture_id=CAP-SCIENCE
  "$UNITY_PYTHON" "$CAPTURE_TOOL" config-proof --capture-id "$capture_id" \
    --fixed-config "$fixed_yaml" --capture-config "$capture_yaml" \
    --fixed-inventory "$record_dir/fixed-config-inventory.json" \
    --capture-inventory "$record_dir/capture-config-inventory.json" \
    --output "$record_dir/config-proof.json"
  "$TOLPROJ" validate-reduction "$redu"
}

prepare_capture_config point P-SEQ "$POINT_REDUCTION_DIR" \
  "$CAPTURE_POINT_FIXED_REALIZED_CONFIG" "$CAPTURE_POINT_REALIZED_CONFIG" \
  "$CAPTURE_POINT_ROOT/capture-authority"
prepare_capture_config science S-E-SEQ "$SCIENCE_REDUCTION_DIR" \
  "$CAPTURE_SCIENCE_FIXED_REALIZED_CONFIG" "$CAPTURE_SCIENCE_REALIZED_CONFIG" \
  "$CAPTURE_SCIENCE_ROOT/capture-authority"
```

If the realized config has any included fragment outside the nine numbered
sources, pass every such file as a repeated `--included-fragment` to both
inventory commands. Leaving that list empty is permitted only after proving
the runtime source manifest declares none. Any other leaf or source-order
difference is a stop.

Finally call TolProj's `freeze_reduction_executable` for each reduction and
prove that both frozen snapshots have the same SHA-256 as
`request_root/bin/citlali`. Do not compile or substitute another executable:

```sh
"$UNITY_PYTHON" - "$CANDIDATE_BINARY" \
  "$POINT_REDUCTION_DIR" "$SCIENCE_REDUCTION_DIR" <<'PY'
import hashlib
import pathlib
import sys
from tolproj.reduction_runtime import freeze_reduction_executable

binary = pathlib.Path(sys.argv[1]).resolve(strict=True)
digest = hashlib.sha256(binary.read_bytes()).hexdigest()
for raw in sys.argv[2:]:
    frozen = freeze_reduction_executable(pathlib.Path(raw))
    if frozen.source.resolve(strict=True) != binary or frozen.sha256 != digest:
        raise SystemExit(f"wrong frozen candidate binary: {raw}")
    if hashlib.sha256(frozen.snapshot.read_bytes()).hexdigest() != digest:
        raise SystemExit(f"snapshot digest differs: {raw}")
print(digest)
PY
```

## 7. Human-operated 200-GiB Unity-root record

The five governed roots, in immutable order, are the Point project, Science
project, CAP-POINT, CAP-SCIENCE, and compact root. The 200-GiB cap applies to
their Unity workspace use, not to this local package. Before and after every
material stage, make a separately invoked record below the governed compact
root. Each pre-record carries the frozen local planning estimate for human
review; it is not a NetCDF/filesystem upper bound or a guarantee of eventual
full/all-PTC use.

A pre-stage record binds a digest-selected planning estimate; a post-stage
record has no estimate and records zero. Both record logical apparent and
allocated use, the selected filesystem, its available capacity, and the cap.
The record command fails if the measured use already exceeds the cap or its
listed planning estimate cannot fit. It does not submit anything. The owner
must inspect each pre-record before separately invoking the next write or
submission command, and must stop—without deletion, cleanup, cache reuse, or
a larger-ceiling request—if measured use approaches or exceeds `214748364800`,
or available capacity is inadequate. A later record includes earlier record
artifacts because they live below the compact governed root.

The mandatory preparation record is in section 4, before any governed write.
After the staging/configuration work in sections 5--6 is complete, take the
matching post-stage record before considering either capture:

```sh
# Unity: record only, after sections 5--6. Review before any capture pre-record.
resource_record PREPARE-STAGING post
```

## 8. Human-only full/all captures

After reviewing the generated Slurm scripts, account, partition, paths,
binary snapshot, config proof, and the CAP-POINT pre-record, the owner may
separately choose whether to submit CAP-POINT. Do not paste these commands as
one compound block and do not proceed automatically to CAP-SCIENCE.

```sh
CAP_POINT_PROJECTION=$(resource_projection CAP-POINT "$FROZEN/resource-report.json")
resource_record CAP-POINT pre "$CAP_POINT_PROJECTION"
```

```sh
# Unity: owner action only after reviewing CAP-POINT.pre.json.
sbatch --wait --parsable "$POINT_REDUCTION_DIR/02_redu.sh" \
  > "$CAPTURE_POINT_ROOT/capture-submit.txt"
```

```sh
# Unity: record only, after CAP-POINT has finished. Review before CAP-SCIENCE.
resource_record CAP-POINT post
```

```sh
# Unity: record only. Review before separately submitting CAP-SCIENCE.
CAP_SCIENCE_PROJECTION=$(resource_projection CAP-SCIENCE "$FROZEN/resource-report.json")
resource_record CAP-SCIENCE pre "$CAP_SCIENCE_PROJECTION"
```

```sh
# Unity: owner action only after reviewing CAP-SCIENCE.pre.json.
sbatch --wait --parsable "$SCIENCE_REDUCTION_DIR/02_redu.sh" \
  > "$CAPTURE_SCIENCE_ROOT/capture-submit.txt"
```

```sh
# Unity: record only, after CAP-SCIENCE has finished.
resource_record CAP-SCIENCE post
```

CAP-POINT must yield one PTC for 152389. CAP-SCIENCE must yield exactly two,
ordered 152390 then 152392. Set exact paths for each PTC,
`raw_timestream_provenance.yaml`, and `mapmaking_provenance.yaml`; each must be
inside its capture root. The helper rejects another full PTC anywhere in that
root.

Create one factual `source-selection.schema.json` document per capture. It
names only regular, digestible authority files and the five roles
`raw_timestream`, `kids_fit_report`, `apt`, `calibration`, and
`pointing_support`, with exact observation/array/network coverage. Use the
resolved canonical raw targets, copied APT/PPT files, and run-produced KIDs and
calibration authorities. Do not guess a network or use final FITS as a source.
The generator automatically adds the PTC/mapmaking/sample-rate/FWHM/target
roles and rejects incomplete or unused records.

```sh
"$UNITY_PYTHON" "$CAPTURE_TOOL" build-raw-input-manifest \
  --capture-id CAP-POINT --capture-root "$CAPTURE_POINT_ROOT" \
  --ptc "152389=$POINT_PTC" \
  --raw-provenance "152389=$POINT_RAW_PROVENANCE" \
  --map-provenance "152389=$POINT_MAP_PROVENANCE" \
  --source-selection "$POINT_SOURCE_SELECTION" \
  --raw-link-manifest "$POINT_SOURCE_PROJECT/logs/raw-link-manifest.json" \
  --raw-link-staging "$POINT_SOURCE_PROJECT/logs/raw-link-staging.json" \
  --authority-manifest "$POINT_SOURCE_PROJECT/logs/authority-staging.json"

"$UNITY_PYTHON" "$CAPTURE_TOOL" build-raw-input-manifest \
  --capture-id CAP-SCIENCE --capture-root "$CAPTURE_SCIENCE_ROOT" \
  --ptc "152390=$SCIENCE_152390_PTC" --ptc "152392=$SCIENCE_152392_PTC" \
  --raw-provenance "152390=$SCIENCE_152390_RAW_PROVENANCE" \
  --raw-provenance "152392=$SCIENCE_152392_RAW_PROVENANCE" \
  --map-provenance "152390=$SCIENCE_152390_MAP_PROVENANCE" \
  --map-provenance "152392=$SCIENCE_152392_MAP_PROVENANCE" \
  --source-selection "$SCIENCE_SOURCE_SELECTION" \
  --raw-link-manifest "$SCIENCE_SOURCE_PROJECT/logs/raw-link-manifest.json" \
  --raw-link-staging "$SCIENCE_SOURCE_PROJECT/logs/raw-link-staging.json" \
  --authority-manifest "$SCIENCE_SOURCE_PROJECT/logs/authority-staging.json"
```

For each of the nine observation/array pairs, run `producer-authority` into an
absent JSON path below its capture root. Then run `capture-record` once per
capture, supplying the one binary/build/version record, duplicated staging
manifests, config proof, exact PTC paths, and raw-provenance paths. These
commands enforce full binary64 primitives, contiguous every-scan/all-detector
coverage, native `SAMPRATE == telescope.fsmp`, separately realized finite
positive `telescope.d_fsmp`, a positive integral downsample factor with
bit-exact `d_fsmp == fsmp / factor`, bit-exact decimal/hex rates,
interval/exposure,
and scan/write cardinality. Missing authority is a stop.

```sh
for array in a1100 a1400 a2000; do
  "$UNITY_PYTHON" "$CAPTURE_TOOL" producer-authority \
    --raw-input-manifest "$CAPTURE_POINT_ROOT/raw-input-manifest.json" \
    --obsnum 152389 --array "$array" \
    --output "$CAPTURE_POINT_ROOT/producer-authority-152389-$array.json"
done
for obs in 152390 152392; do
  for array in a1100 a1400 a2000; do
    "$UNITY_PYTHON" "$CAPTURE_TOOL" producer-authority \
      --raw-input-manifest "$CAPTURE_SCIENCE_ROOT/raw-input-manifest.json" \
      --obsnum "$obs" --array "$array" \
      --output "$CAPTURE_SCIENCE_ROOT/producer-authority-$obs-$array.json"
  done
done

"$UNITY_PYTHON" "$CAPTURE_TOOL" capture-record \
  --capture-id CAP-POINT --capture-root "$CAPTURE_POINT_ROOT" \
  --binary "$CANDIDATE_BINARY" --build-manifest "$CANDIDATE_BUILD_MANIFEST" \
  --version-output "$CANDIDATE_VERSION_OUTPUT" \
  --raw-link-manifest "$POINT_SOURCE_PROJECT/logs/raw-link-manifest.json" \
  --raw-link-staging "$POINT_SOURCE_PROJECT/logs/raw-link-staging.json" \
  --authority-manifest "$POINT_SOURCE_PROJECT/logs/authority-staging.json" \
  --config-proof "$CAPTURE_POINT_ROOT/capture-authority/config-proof.json" \
  --ptc "152389=$POINT_PTC" \
  --raw-provenance "152389=$POINT_RAW_PROVENANCE"

"$UNITY_PYTHON" "$CAPTURE_TOOL" capture-record \
  --capture-id CAP-SCIENCE --capture-root "$CAPTURE_SCIENCE_ROOT" \
  --binary "$CANDIDATE_BINARY" --build-manifest "$CANDIDATE_BUILD_MANIFEST" \
  --version-output "$CANDIDATE_VERSION_OUTPUT" \
  --raw-link-manifest "$SCIENCE_SOURCE_PROJECT/logs/raw-link-manifest.json" \
  --raw-link-staging "$SCIENCE_SOURCE_PROJECT/logs/raw-link-staging.json" \
  --authority-manifest "$SCIENCE_SOURCE_PROJECT/logs/authority-staging.json" \
  --config-proof "$CAPTURE_SCIENCE_ROOT/capture-authority/config-proof.json" \
  --ptc "152390=$SCIENCE_152390_PTC" --ptc "152392=$SCIENCE_152392_PTC" \
  --raw-provenance "152390=$SCIENCE_152390_RAW_PROVENANCE" \
  --raw-provenance "152392=$SCIENCE_152392_RAW_PROVENANCE"

"$UNITY_PYTHON" "$CAPTURE_TOOL" verify-capture-record \
  --capture-record "$CAPTURE_POINT_ROOT/capture-record.json"
"$UNITY_PYTHON" "$CAPTURE_TOOL" verify-capture-record \
  --capture-record "$CAPTURE_SCIENCE_ROOT/capture-record.json"
```

## 9. Produce and verify exactly nine compact groups

For each pair in this exact order:

```text
152389:a1100 152389:a1400 152389:a2000
152390:a1100 152390:a1400 152390:a2000
152392:a1100 152392:a1400 152392:a2000
```

derive a positive compact-output projection. The following helper fixes the
stage identity, pre/post records, governed-root order, output path, and
immediate verification:

```sh
produce_compact_group() {
  obs=$1; array=$2; ptc=$3; authority=$4
  stage="compact-production:$obs:$array"
  safe=${stage//:/-}
  projection=$(resource_projection "$stage" "$authority")
  resource_record "$stage" pre "$projection"
  "$UNITY_PYTHON" "$COMPACT_TOOL" produce \
    --source "$ptc" --authority "$authority" \
    --resource-record "$RESOURCE_RECORDS/$safe.pre.json" \
    --resource-inventory "$RESOURCE_RECORDS/$safe.pre.inventory.json" \
    --governed-root "$POINT_SOURCE_PROJECT" \
    --governed-root "$SCIENCE_SOURCE_PROJECT" \
    --governed-root "$CAPTURE_POINT_ROOT" \
    --governed-root "$CAPTURE_SCIENCE_ROOT" \
    --governed-root "$COMPACT_EVIDENCE_ROOT" \
    --output-dir "$COMPACT_EVIDENCE_ROOT/$obs:$array"
  resource_record "$stage" post
  "$UNITY_PYTHON" "$COMPACT_TOOL" verify \
    --group "$COMPACT_EVIDENCE_ROOT/$obs:$array/group.json"
}

produce_compact_group 152389 a1100 "$POINT_PTC" \
  "$CAPTURE_POINT_ROOT/producer-authority-152389-a1100.json"
produce_compact_group 152389 a1400 "$POINT_PTC" \
  "$CAPTURE_POINT_ROOT/producer-authority-152389-a1400.json"
produce_compact_group 152389 a2000 "$POINT_PTC" \
  "$CAPTURE_POINT_ROOT/producer-authority-152389-a2000.json"
produce_compact_group 152390 a1100 "$SCIENCE_152390_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152390-a1100.json"
produce_compact_group 152390 a1400 "$SCIENCE_152390_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152390-a1400.json"
produce_compact_group 152390 a2000 "$SCIENCE_152390_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152390-a2000.json"
produce_compact_group 152392 a1100 "$SCIENCE_152392_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152392-a1100.json"
produce_compact_group 152392 a1400 "$SCIENCE_152392_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152392-a1400.json"
produce_compact_group 152392 a2000 "$SCIENCE_152392_PTC" \
  "$CAPTURE_SCIENCE_ROOT/producer-authority-152392-a2000.json"
```

After result collection, run `verify-nine --collection
"$COMPACT_EVIDENCE_ROOT/_campaign/analysis/result-collection.json"`.
The compact producer must emit complete
stream digests, per-scan/per-pixel sufficient statistics, pinned 64 signs,
and first/lower-middle/last valid/flagged traces or typed absence for every
active network. It may not emit a full primitive-term axis.

Focused expansion is not automatic. Only a re-auditor-named request conforming
to `discrepancy-request.schema.json` may run `plan-expansion` and then
`emit-expansion`, each behind its own request-derived pre-stage resource
projection and zero-increment post record. The same retained PTC and producer authority must
be used in both digest-identical passes. Otherwise do not invoke expansion.

If, and only if, such a request is later supplied, use these exact two passes
with the PTC and producer authority for the request's one observation/array:

```sh
: "${FOCUSED_REQUEST_NAME:?exact request_id from the approved discrepancy request}"
: "${FOCUSED_REQUEST_JSON:?absolute approved discrepancy-request JSON}"
: "${FOCUSED_SOURCE_PTC:?exact retained PTC for the requested observation}"
: "${FOCUSED_PRODUCER_AUTHORITY:?matching observation-array producer authority}"
FOCUSED_ROOT="$COMPACT_EVIDENCE_ROOT/focused-expansion/$FOCUSED_REQUEST_NAME"
FOCUSED_PLAN="$FOCUSED_ROOT/plan.json"
FOCUSED_OUTPUT="$FOCUSED_ROOT/expansion.npz"

plan_stage="focused-expansion-plan:$FOCUSED_REQUEST_NAME"
plan_safe=${plan_stage//:/-}
plan_projection=$(resource_projection "$plan_stage" "$FOCUSED_REQUEST_JSON")
resource_record "$plan_stage" pre "$plan_projection"
"$UNITY_PYTHON" "$COMPACT_TOOL" plan-expansion \
  --source "$FOCUSED_SOURCE_PTC" --authority "$FOCUSED_PRODUCER_AUTHORITY" \
  --resource-record "$RESOURCE_RECORDS/$plan_safe.pre.json" \
  --resource-inventory "$RESOURCE_RECORDS/$plan_safe.pre.inventory.json" \
  --governed-root "$POINT_SOURCE_PROJECT" \
  --governed-root "$SCIENCE_SOURCE_PROJECT" \
  --governed-root "$CAPTURE_POINT_ROOT" \
  --governed-root "$CAPTURE_SCIENCE_ROOT" \
  --governed-root "$COMPACT_EVIDENCE_ROOT" \
  --request "$FOCUSED_REQUEST_JSON" --output "$FOCUSED_PLAN"
resource_record "$plan_stage" post

emit_stage="focused-expansion:$FOCUSED_REQUEST_NAME"
emit_safe=${emit_stage//:/-}
emit_projection=$(resource_projection "$emit_stage" "$FOCUSED_REQUEST_JSON")
resource_record "$emit_stage" pre "$emit_projection"
"$UNITY_PYTHON" "$COMPACT_TOOL" emit-expansion \
  --source "$FOCUSED_SOURCE_PTC" --authority "$FOCUSED_PRODUCER_AUTHORITY" \
  --resource-record "$RESOURCE_RECORDS/$emit_safe.pre.json" \
  --resource-inventory "$RESOURCE_RECORDS/$emit_safe.pre.inventory.json" \
  --governed-root "$POINT_SOURCE_PROJECT" \
  --governed-root "$SCIENCE_SOURCE_PROJECT" \
  --governed-root "$CAPTURE_POINT_ROOT" \
  --governed-root "$CAPTURE_SCIENCE_ROOT" \
  --governed-root "$COMPACT_EVIDENCE_ROOT" \
  --plan "$FOCUSED_PLAN" --output "$FOCUSED_OUTPUT"
resource_record "$emit_stage" post
```

## 10. Run the unchanged seven-case lane

Install the two automatic manifests byte-for-byte into the seven fixed cases,
then prepare the native cases:

```sh
"$UNITY_PYTHON" "$DRIVER" --campaign "$FROZEN/campaign.json" \
  bind-raw-manifests --owner-values "$FROZEN_VALUES" \
  --point "$CAPTURE_POINT_ROOT/raw-input-manifest.json" \
  --science "$CAPTURE_SCIENCE_ROOT/raw-input-manifest.json"
"$UNITY_PYTHON" "$DRIVER" --campaign "$FROZEN/campaign.json" \
  prepare-cases --owner-values "$FROZEN_VALUES"
"$UNITY_PYTHON" "$DRIVER" --campaign "$FROZEN/campaign.json" \
  emit-submit-plan --owner-values "$FROZEN_VALUES"
```

Review, then manually execute the emitted plan on Unity:

```sh
bash "$REQUEST_ROOT/plans/submit-seven-cases.sh"
```

The order remains P-SEQ, P-OMP, S-C-SEQ, S-C-OMP, S-E-SEQ, S-E-OMP,
S-X-SEQ. All original observations, arrays, products, 64 realizations, CPU
counts, tolerances, WCS/coadd/support/provenance gates, and expected exit-zero
outcomes remain unchanged. The same ordinary binary digest is mandatory.

## 11. Collection, analysis, and bounded return

After all seven cases finish:

```sh
"$UNITY_PYTHON" "$DRIVER" --campaign "$FROZEN/campaign.json" \
  build-result-collection --owner-values "$FROZEN_VALUES"
"$UNITY_PYTHON" "$DRIVER" --campaign "$FROZEN/campaign.json" \
  emit-final-plan --owner-values "$FROZEN_VALUES"
```

Review the emitted analysis plan before manually running it:

```sh
bash "$REQUEST_ROOT/plans/analyze-freeze-and-retrieve.sh"
```

Analysis exit 0 means the frozen checks completed; exit 2 means a complete
nonconformance result. Neither closes a finding. Other exits are evidence
execution failures. The plan records Slurm accounting, final logical and
allocated resource records, full remote inventory, and a deterministic return
bundle. ANALYSIS and FINAL-BUNDLE each have a typed metadata-derived
projection and pre/post gate, and the plan verifies the complete required
stage-pair set before handback. Analysis, evidence, manifests, and return
construction remain below `compact-groups/_campaign` and therefore inside the
governed compact root. The bundle excludes every exact retained full PTC path from the return tar and
verifies the exclusion; capture records retain their paths, sizes, and
digests. Full PTC remains on Unity.

Run the printed retrieval command locally. It uses only `unity_toltec`.
Verify the outer digest, extract into a fresh directory, and verify
`SCI-MAP-001-UNITY-001-RETURN-MANIFEST.sha256`. Do not expect the full remote
manifest to verify after extraction because the deliberately retained PTC
payloads are absent from the bounded return.

## 12. Retention and stop boundary

Retain both capture roots through fresh MAP re-audit and every requested
focused expansion. Do not run cleanup. `FUTURE_CLEANUP_PLAN.md` deliberately
contains no executable deletion command and remains ineligible until the
coordinator records acceptance of the fresh re-audit.

The final-bundle plan also retains its deterministic temporary TAR below the
governed compact return directory. It is included by the final post-stage
resource record and is not removed, reused, or replaced by this package.

This workflow does not integrate the repair, supply evidence before retrieval
and independent review, close F009/F010/F012/F013 or any ALIGN/CAL/AST/PTC/VAL
dependency, launch re-audit by itself, admit production, or expand production.
