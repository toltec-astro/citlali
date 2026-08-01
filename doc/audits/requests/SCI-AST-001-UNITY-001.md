# SCI-AST-001 Unity external-evidence request

This is the exact human-run request for the Tier A pointing, astrometry,
detector-coordinate, and WCS audit. Its status is **UNSUPPLIED**. Codex did not
connect to Unity, inspect Unity state, execute these commands, or claim any
result. Grant (or another named human operator) must return the complete bundle
before any row below can become evidence.

## Request identity

| Field | Required value |
| --- | --- |
| Request ID | `SCI-AST-001-UNITY-001` |
| Package ID | `SCI-AST-001` |
| Audit specification commit | `17d683ada3856ecb5f0a5c42eed744cb219a3586` plus this request's final report-bearing commit, to be recorded before dispatch |
| Governing source SHA | `9aae0e669384c5c0c0dda93debc194d6b8dac787` |
| Source role | Exact audited implementation; no repair or candidate substitution |
| Permitted dirty state | Clean only; any `git status --short` output rejects the run as same-SHA evidence |
| Required binary version | Exact `build/bin/citlali --version` output; it must identify the governing SHA |
| Build | Existing Unity toolchain, `Release`, target `citlali_cli`, `cmake --build build --target citlali_cli -j 8` |
| Direct dependencies | Return `CMakeCache.txt`, generated version header, compiler IDs/versions, Conan lock/graph if present, and exact KIDs/Tula revisions |
| Runtime | Record host, Slurm job/partition, CPUs, affinity, `OMP_*`, memory/NUMA policy, and environment module list |
| Evidence owner | Grant or a named human operator |

The source checkout, build, and every run must be clean and bound to the full
SHA above. If the build cannot identify that SHA, if a base configuration
digest differs, if a required raw/APT/calibration input is unavailable, or if
an output root already exists, stop. Do not repair, overwrite, or substitute.
A repair requires a new request with a new exact source SHA.

## Exact configuration set

The request configuration manifest is
`doc/audits/requests/SCI-AST-001-UNITY-001/config_manifest.yaml`, SHA-256
`b8c5defb5f56b693141f901b918fbe36ec1c5dc5e3214fde5586c50277fed3d9`.
Audit overlays are applied after the retained low-level input, in the order
shown. The operator must return the canonical merged digest emitted by the
governing binary; `UNSUPPLIED_BY_OPERATOR` is deliberately not a passing value.

| Case | Ordered base input (required SHA-256) | Audit overlay (required SHA-256) | Purpose |
| --- | --- | --- | --- |
| `AST-POINT-SEQ` | `.../redu61/citlali_o152389_0_2_c1.yaml` (`b494d8671fb162f47d6eaadc1299755594db5c8c27353d7e8b4ac8ffe8566ed8`) | `point_seq_overlay.yaml` (`d4fcbdc539513acfd230dc48bad970213786082439b12f3d04790a4cf8f25796`) | Constant correction, Point products, sequential reference |
| `AST-POINT-OMP` | same Point input | `point_omp_overlay.yaml` (`cbd5552ba406c577bfc304c1feb55fe234e59f7472e9c4fbbc5e06a1d8220b41`) | Coordinate seq/OMP equivalence |
| `AST-OOF-SEQ` | `.../redu02/citlali_o152385_0_1_c3.yaml` (`aebfc3b764f4abeca1f7f7cfe60723714f000ca849f60b167cd495294f87bdbc`) | `oof_seq_overlay.yaml` (`852c6e3d1655562f3bd5b7c49a104fbae0118fd8490d98623749796acf384581`) | Three-observation replacement, sequential reference |
| `AST-OOF-OMP` | same OOF input | `oof_omp_overlay.yaml` (`63ba786186eb57d8f22560559b58636792c05850f98fb5600eeaa2d59ca637e2`) | Repeated-application seq/OMP equivalence |
| `AST-SCIENCE-SEQ` | `.../redu23/citlali_o152390_0_2_c2.yaml` (`cd860b8b712810f9f8651c325a6f1642e8453aa6148410cf7ff7a7058de286a3`) | `science_seq_overlay.yaml` (`bea974db126d4f4bd82d1c38d27b10888bb775bbd2da1478c6d66f9aeb095c63`) | Positive-MJD two-support interpolation |
| `AST-SCIENCE-OMP` | same Science input | `science_omp_overlay.yaml` (`ae01aa9a6b70aa6bc24c9195924875976fd8ea36c7f51a3982b99089647971c7`) | Science coordinate seq/OMP equivalence |
| `AST-BEAMMAP-SEQ` | `.../redu05/citlali_o148670_0_2_c1.yaml` (`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`) | `beammap_seq_overlay.yaml` (`018ecd63ba62ca4060dcc5eb2adb633b7c15261eb7c9f7e881dc6a33ae02eb3c`) | APT/detector identity and final TOD reference |
| `AST-BEAMMAP-OMP` | same Beammap input | `beammap_omp_overlay.yaml` (`1bdc6dee0cb6ae1fefcc7f3993c4ffec9cadeeb82786d31c8e8d06b14c723046`) | Detector-coordinate seq/OMP equivalence |
| `AST-POINT-NONDEFAULT-WCS` | same Point input | `point_nondefault_wcs_overlay.yaml` (`73ea6ff702c83703ee23bea0cffa420c899f2c29ab2bc4728eebb212e20a4ddf`) | Fail-closed or exactly realized nondefault WCS controls; silent acceptance is failure |

The complete base paths are frozen in the request manifest. Historical merged
digests are compatibility evidence only and are not reused after an overlay is
applied. No undocumented overlay or interactive edit is permitted.

## Observations, inputs, and arrays

| Case family | Observation/selection | Required input identity | Arrays/networks | Scientific purpose |
| --- | --- | --- | --- | --- |
| Point | `152389`, identical scan selection to accepted `redu61` | Return raw-file list/digests, telescope input, APT, calibration, and source header | All available a1100/a1400/a2000 networks; justify any absence | Constant offset, source crossing, centroid, WCS, Point table |
| OOF | `152385`, `152386`, `152387`, identical sequence to accepted `redu02` | Same identities plus observation ordering | All available arrays/networks | State replacement, repeated application, Point/OOF coordinate tables |
| Science | `152390`, `152392`, identical selection to accepted `redu23` | Same identities plus both positive MJD support records | All available arrays/networks | Explicit-MJD interpolation and retained-iteration propagation |
| Beammap | `148670`, identical selection to accepted `redu05` | Same identities plus exact APT row order/UIDs and raw detector row identity | All available arrays/networks | Detector-coordinate composition, final detector TOD, APT linkage |

Return file size and SHA-256 for every raw, telescope, APT, calibration, and
ephemeris input actually opened. A filename, obsnum, or previous accepted run
is not a content identity. A missing required array, network, telescope field,
or APT mapping is an evidence gap, not a pass.

## Determinism, controls, and simulations

AST is a deterministic coordinate transformation; random noise realizations
and blank-sky RMS are not estimators of its correctness and are
`not_applicable`. The following deterministic controls are required instead:

1. exact sequential/OMP equality of `det_lat`, `det_lon`, and, for RA/Dec,
   `det_ra` and `det_dec`, after stable detector/sample matching;
2. exact requested/effective/resolved/realized pointing-support records for
   each observation and each fruit-loop application;
3. high-precision independent forward-TAN/inverse-TAN round trips for valid
   center, small-offset, RA-wrap, high-declination, and near-domain-boundary
   points;
4. explicit invalid results for non-finite input and every direction with
   nonpositive TAN denominator; no invalid input may become finite map center;
5. both positive and negative azimuth-boundary crossings, plus ordinary
   no-wrap controls;
6. constant, observation-span, positive-MJD, mixed-sentinel, nonmonotone,
   duplicate-support, unbracketed, and subsecond-support cases;
7. APT identity, permutation, duplicate UID, length mismatch, non-finite
   offset, flagged row, and network-subset cases;
8. WCS handedness, zero/one-based CRPIX, odd dimensions, CRVAL precision,
   frame/epoch identity, and complete FITS read/write round trips;
9. inverse-TAN Jacobian and full two-coordinate covariance at declinations
   0, 60, and 80 degrees, including a nonzero cross term; and
10. simulation parity for each enabled frame. Galactic simulation is required
    to fail closed until its coordinate construction is complete.

Items 3--10 do not have a compiled governing-SHA fixture in the repository.
Their present evidence status is therefore **UNSUPPLIED**, even if the four
operational reductions complete. Do not replace them with a Python
reimplementation and call that compiled-source evidence. A later repair must
add bounded tests and receive a new exact-SHA request.

Astronomical recovery is applicable only as an operational cross-check:
return Point/OOF centroids, source crossing selection, Beammap detector/source
fits, PSF location, and map WCS against the retained accepted products. This
cannot override an analytic domain, unit, identity, or covariance failure.

## Required products and provenance

| Artifact | Required identity and content | Completeness rule |
| --- | --- | --- |
| Build identity | Full source SHA, clean status, version text, binary digest, compiler/build/options, direct dependency identities | Exactly one clean build; no mixed SHA |
| Config source manifest | Ordered base and overlay bytes/digests plus canonical merged digest | One complete manifest per case |
| Astrometry provenance | Requested, effective, observation-resolved, realized axes, sign/basis/frame/topology, support values/times/precision, algorithm, counts, validity, and source identities | One observation record per physical observation; absence is failure |
| Detector coordinates | Full `det_lat/det_lon`, full `det_ra/det_dec` where applicable, stable sample/detector IDs, units, frame, epoch, validity, response/covariance availability | Complete full TOD; mini TOD does not satisfy this row |
| WCS/FITS | Full-precision CTYPE/CUNIT/CRPIX/CRVAL/CDELT/RADESYS/EQUINOX and all map arrays | Every required HDU, every array, exact image/WCS round trip |
| Point/OOF/source tables | Coordinate semantic names, means, full covariance/Jacobian, units/frame/epoch, validity, fit identity | Every fitted array/map; missing covariance is explicit failure |
| Telescope variables | Per-variable units/topology/frame/validity and values | No generic radians label for time/state/counter fields |
| APT/detector identity | APT bytes/digest, UID/order, raw row mapping, x/y basis/units/frame/validity | One-to-one mapping or explicit fail-closed rejection |
| Logs | Complete stdout/stderr, exit status, zero unexpected error/critical/fatal records, zero silent required-data skips | One log and status per case |
| Inventory | Relative path, size, SHA-256 of every output; durable location for large products | No unlisted product |

Signal flags, map coverage, hits, support, validity, confidence, response, and
variance are different quantities. A signal flag cannot stand in for
coordinate validity, and formal fit errors cannot stand in for astrometric
correction covariance.

## Pre-registered comparisons and tolerances

| ID | Candidate quantity | Reference | Metric and acceptance bound |
| --- | --- | --- | --- |
| `AST-C001` | Source/build/config identity | Request constants and returned bytes | Full SHA/digest exact; dirty or missing identity fails |
| `AST-C002` | Valid TAN and inverse-TAN | Independent long-double or Astropy reference with exact frame declared | Angular round trip `<=1e-11 rad`; no clipping or silent normalization |
| `AST-C003` | Invalid TAN domain | Analytic denominator `D<=0` and non-finite fixtures | Explicit invalid/failure in 100% of cases; zero finite-center aliases |
| `AST-C004` | Azimuth wrap | Analytic shortest signed angular difference | Both wrap directions continuous to `<=1e-12 rad`; ordinary control unchanged to `<=1e-15 rad` |
| `AST-C005` | Support interpolation | Analytic affine weights from returned double support/sample times | Values `<=5e-12 arcsec` absolute; times preserved to `<=1e-6 s`; no extrapolation or ambiguous sentinel fallback |
| `AST-C006` | Detector composition | Frozen rotation/composition equations and exact APT mapping | Coordinates `<=1e-11 rad`; identity/order exact; invalid inputs fail closed |
| `AST-C007` | WCS | Frozen handedness/indexing equations and independent FITS WCS | CRPIX/CRVAL/CDELT/frame/epoch exact at stored precision; sky round trip `<=1e-10 deg`; nondefault control is rejected or exactly realized, never ignored |
| `AST-C008` | Positional response/covariance | Analytic inverse-TAN Jacobian | Mean `<=1e-10 deg`; every covariance element relative error `<=1e-8` or absolute `<=1e-16 deg^2`, whichever is larger |
| `AST-C009` | Sequential/OMP detector coordinates | Stable matched arrays from paired cases | Bitwise exact coordinate values, validity, and IDs; no missing/extra samples |
| `AST-C010` | Point/OOF/Beammap stable products | Retained accepted predecessor for the same mode | Existing exact profile remains exact after the historical volatile allowlist; all deviations reported |
| `AST-C011` | Science stable products | Retained accepted science predecessor | Existing `atol=2e-8`, `rtol=1e-10`; no skipped record and no loosened bound |
| `AST-C012` | Product units/provenance | Frozen AST variable/state contract | Exact semantic strings/cardinality; no false units, missing validity, or omitted required state |

The numerical tolerances are pre-registered validation bounds, not approval of
the present contract. The scientific owner must confirm or supersede the
subsecond time and precision requirements before a repair is authorized. Do
not loosen a retained profile to admit a result.

## Commands Grant should execute

The first block runs from Grant's Mac after this request has been committed.
It copies only audit-request bytes outside the Unity source checkout. It does
not move a branch or modify the protected comparison repository.

```bash
AST_LOCAL_REQUEST=/Users/gwilson/.codex/worktrees/bcb5/citlali-refactor/doc/audits/requests/SCI-AST-001-UNITY-001
AST_REMOTE_PARENT=/work/toltec/commissioning2025-test/2026-scientific-audits

ssh unity_toltec "test ! -e '${AST_REMOTE_PARENT}/SCI-AST-001-UNITY-001' && mkdir -p '${AST_REMOTE_PARENT}'"
scp -r "${AST_LOCAL_REQUEST}" "unity_toltec:${AST_REMOTE_PARENT}/"
```

The next block runs remotely. Before it is run, Grant must place the refactor
repository at the exact governing SHA in a clean, isolated, human-managed
checkout with a valid Unity build. If that precondition is not already true,
stop and stage it separately; do not change an active or protected checkout
inside this request.

```bash
ssh unity_toltec 'bash -s' <<'UNITY_AST_IDENTITY'
set -euo pipefail

AST_REPO=/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor
AST_ROOT=/work/toltec/commissioning2025-test/2026-scientific-audits/SCI-AST-001-UNITY-001
AST_SHA=9aae0e669384c5c0c0dda93debc194d6b8dac787
AST_BIN=${AST_REPO}/build/bin/citlali

test "$(git -C "${AST_REPO}" rev-parse HEAD)" = "${AST_SHA}"
test -z "$(git -C "${AST_REPO}" status --porcelain=v1)"
test -d "${AST_ROOT}"
test -x "${AST_BIN}"

mkdir -p "${AST_ROOT}/identity" "${AST_ROOT}/logs" "${AST_ROOT}/audit"
git -C "${AST_REPO}" rev-parse HEAD > "${AST_ROOT}/identity/source_sha.txt"
git -C "${AST_REPO}" status --short > "${AST_ROOT}/identity/git_status_short.txt"
git -C "${AST_REPO}" submodule status --recursive > "${AST_ROOT}/identity/submodules.txt"
cmake --build "${AST_REPO}/build" --target citlali_cli -j 8 2>&1 | tee "${AST_ROOT}/logs/build.log"
"${AST_BIN}" --version > "${AST_ROOT}/identity/citlali_version.txt" 2>&1
sha256sum "${AST_BIN}" > "${AST_ROOT}/identity/citlali_binary.sha256"
cp "${AST_REPO}/build/CMakeCache.txt" "${AST_ROOT}/identity/CMakeCache.txt"
cmake -LA -N "${AST_REPO}/build" > "${AST_ROOT}/identity/cmake_cache_listing.txt"
command -v c++ > "${AST_ROOT}/identity/cxx_path.txt"
c++ --version > "${AST_ROOT}/identity/cxx_version.txt" 2>&1
env | LC_ALL=C sort > "${AST_ROOT}/identity/environment.txt"
module list > "${AST_ROOT}/identity/modules.txt" 2>&1 || true
lscpu > "${AST_ROOT}/identity/lscpu.txt" 2>&1 || true
numactl --show > "${AST_ROOT}/identity/numactl.txt" 2>&1 || true

find "${AST_REPO}" -maxdepth 3 -type f \
  \( -name 'conan.lock' -o -name 'conanfile.py' -o -name 'conanfile.txt' -o -name 'gitversion.h' \) \
  -print -exec sha256sum {} \; > "${AST_ROOT}/identity/dependency_and_version_files.txt"

sha256sum "${AST_ROOT}/config_manifest.yaml" "${AST_ROOT}"/*_overlay.yaml \
  > "${AST_ROOT}/identity/request_config_digests.txt"
UNITY_AST_IDENTITY
```

Run the eight ordinary cases. Each base digest is checked immediately before
execution. The output roots were preregistered and must not have existed before
the request transfer.

```bash
ssh unity_toltec 'bash -s' <<'UNITY_AST_RUNS'
set -euo pipefail

AST_REPO=/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor
AST_ROOT=/work/toltec/commissioning2025-test/2026-scientific-audits/SCI-AST-001-UNITY-001
AST_BIN=${AST_REPO}/build/bin/citlali

POINT_BASE=/work/toltec/commissioning2025-test/2026-refactor/point/refactor/reduced/redu61/citlali_o152389_0_2_c1.yaml
OOF_BASE=/work/toltec/commissioning2025-test/2026-refactor/oof/refactor/reduced/redu02/citlali_o152385_0_1_c3.yaml
SCIENCE_BASE=/work/toltec/commissioning2025-test/2026-refactor/science/refactor/reduced/redu23/citlali_o152390_0_2_c2.yaml
BEAMMAP_BASE=/work/toltec/commissioning2025-test/2026-refactor/beammap/refactor/reduced/redu05/citlali_o148670_0_2_c1.yaml

test "$(sha256sum "${POINT_BASE}" | awk '{print $1}')" = b494d8671fb162f47d6eaadc1299755594db5c8c27353d7e8b4ac8ffe8566ed8
test "$(sha256sum "${OOF_BASE}" | awk '{print $1}')" = aebfc3b764f4abeca1f7f7cfe60723714f000ca849f60b167cd495294f87bdbc
test "$(sha256sum "${SCIENCE_BASE}" | awk '{print $1}')" = cd860b8b712810f9f8651c325a6f1642e8453aa6148410cf7ff7a7058de286a3
test "$(sha256sum "${BEAMMAP_BASE}" | awk '{print $1}')" = aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d

run_case() {
  AST_CASE=$1
  AST_BASE=$2
  AST_OVERLAY=$3
  set +e
  "${AST_BIN}" -l info "${AST_BASE}" "${AST_ROOT}/${AST_OVERLAY}" \
    > "${AST_ROOT}/logs/${AST_CASE}.log" 2>&1
  AST_STATUS=$?
  set -e
  printf '%s\n' "${AST_STATUS}" > "${AST_ROOT}/logs/${AST_CASE}.status"
  test "${AST_STATUS}" -eq 0
}

run_case AST-POINT-SEQ "${POINT_BASE}" point_seq_overlay.yaml
run_case AST-POINT-OMP "${POINT_BASE}" point_omp_overlay.yaml
run_case AST-OOF-SEQ "${OOF_BASE}" oof_seq_overlay.yaml
run_case AST-OOF-OMP "${OOF_BASE}" oof_omp_overlay.yaml
run_case AST-SCIENCE-SEQ "${SCIENCE_BASE}" science_seq_overlay.yaml
run_case AST-SCIENCE-OMP "${SCIENCE_BASE}" science_omp_overlay.yaml
run_case AST-BEAMMAP-SEQ "${BEAMMAP_BASE}" beammap_seq_overlay.yaml
run_case AST-BEAMMAP-OMP "${BEAMMAP_BASE}" beammap_omp_overlay.yaml

set +e
"${AST_BIN}" -l info "${POINT_BASE}" "${AST_ROOT}/point_nondefault_wcs_overlay.yaml" \
  > "${AST_ROOT}/logs/AST-POINT-NONDEFAULT-WCS.log" 2>&1
AST_WCS_STATUS=$?
set -e
printf '%s\n' "${AST_WCS_STATUS}" > "${AST_ROOT}/logs/AST-POINT-NONDEFAULT-WCS.status"
UNITY_AST_RUNS
```

Audit each completed root and compare paired sequential/OMP products. Any
required provenance failure, skipped record, serious log entry, missing
coordinate product, or comparator error remains a failure in the returned
bundle.

```bash
ssh unity_toltec 'bash -s' <<'UNITY_AST_AUDIT'
set -euo pipefail

AST_REPO=/home/toltec_umass_edu/work_toltec/citlali_dev/citlali_refactor
AST_ROOT=/work/toltec/commissioning2025-test/2026-scientific-audits/SCI-AST-001-UNITY-001
AST_PYTHON=/home/toltec_umass_edu/tolteca/bin/python
if test ! -x "${AST_PYTHON}"; then AST_PYTHON=python3; fi

audit_case() {
  AST_CASE=$1
  AST_MODE=$2
  AST_PATH=$3
  "${AST_PYTHON}" "${AST_REPO}/tools/baseline/audit_reduction_run.py" \
    "${AST_PATH}" --expected-mode "${AST_MODE}" \
    --require-astrometry-provenance --require-config-source-manifest \
    --require-runtime-provenance --require-mapmaking-provenance \
    --json-out "${AST_ROOT}/audit/${AST_CASE}.json" \
    --report-out "${AST_ROOT}/audit/${AST_CASE}.md"
}

audit_case AST-POINT-SEQ point "${AST_ROOT}/point-seq"
audit_case AST-POINT-OMP point "${AST_ROOT}/point-omp"
audit_case AST-OOF-SEQ oof "${AST_ROOT}/oof-seq"
audit_case AST-OOF-OMP oof "${AST_ROOT}/oof-omp"
audit_case AST-SCIENCE-SEQ science "${AST_ROOT}/science-seq"
audit_case AST-SCIENCE-OMP science "${AST_ROOT}/science-omp"
audit_case AST-BEAMMAP-SEQ beammap "${AST_ROOT}/beammap-seq"
audit_case AST-BEAMMAP-OMP beammap "${AST_ROOT}/beammap-omp"

compare_pair() {
  AST_MODE=$1
  AST_SEQ=$2
  AST_OMP=$3
  AST_NAME=$4
  AST_ATOL=$5
  AST_RTOL=$6
  "${AST_PYTHON}" "${AST_REPO}/tools/baseline/compare_reduction_products.py" \
    "${AST_SEQ}" "${AST_OMP}" --mode "${AST_MODE}" --include-timestream \
    --max-array-elements 0 --exclude citlali_profile.ecsv \
    --exclude runtime_provenance.yaml --exclude timestream_output_provenance.yaml \
    --atol "${AST_ATOL}" --rtol "${AST_RTOL}" --strict \
    --json-out "${AST_ROOT}/audit/${AST_NAME}.json" \
    --report-out "${AST_ROOT}/audit/${AST_NAME}.md"
}

compare_pair point "${AST_ROOT}/point-seq" "${AST_ROOT}/point-omp" point-seq-omp 0 0
compare_pair oof "${AST_ROOT}/oof-seq" "${AST_ROOT}/oof-omp" oof-seq-omp 0 0
compare_pair science "${AST_ROOT}/science-seq" "${AST_ROOT}/science-omp" science-seq-omp 2e-8 1e-10
compare_pair beammap "${AST_ROOT}/beammap-seq" "${AST_ROOT}/beammap-omp" beammap-seq-omp 0 0

find "${AST_ROOT}" -type f -print0 | LC_ALL=C sort -z | xargs -0 sha256sum \
  > "${AST_ROOT}/audit/complete_file_manifest.sha256"
find "${AST_ROOT}" -type f -printf '%P\t%s\n' | LC_ALL=C sort \
  > "${AST_ROOT}/audit/complete_file_inventory.tsv"
UNITY_AST_AUDIT
```

If the assumed Unity Python path is absent, the fallback interpreter identity
must be recorded; missing required modules or audit failures remain gaps. The
operator must additionally return the complete products for independent AST
coordinate/WCS/covariance inspection. The generic comparator does not replace
the absent compiled edge fixtures.

Finally, return the immutable evidence bundle to Grant's Mac without deleting
the Unity copy:

```bash
AST_LOCAL_RETURN=/Users/gwilson/work_toltec/local_data/2026-scientific-audits/SCI-AST-001-UNITY-001
mkdir -p "${AST_LOCAL_RETURN}"
rsync -a --checksum "unity_toltec:/work/toltec/commissioning2025-test/2026-scientific-audits/SCI-AST-001-UNITY-001/" "${AST_LOCAL_RETURN}/"
```

## Evidence Grant must return

Return one immutable bundle or manifest containing:

1. this request ID, audit specification identity, full governing source SHA,
   clean status, exact binary version/digest, build/compiler/dependency files,
   node/Slurm/runtime/environment identity, and operator name/time;
2. exact commands and every exit status, including the nondefault-WCS case;
3. every ordered config byte/digest and the binary-emitted canonical merged
   digest for each case;
4. complete raw/telescope/APT/calibration identities and stable sample,
   observation, detector, array, and network mappings;
5. complete logs with every error/critical/fatal/skip categorized;
6. all FITS, NetCDF, ECSV, provenance, config-source, WCS, detector-coordinate,
   source/Point/OOF/Beammap, and retained-iteration products;
7. machine-readable audit/comparison outputs and all missing/extra/changed/
   skipped/invalid/unreadable records;
8. each metric in `AST-C001`--`AST-C012`, its pre-registered bound, measured
   value, and pass/fail, with `UNSUPPLIED` retained where no fixture exists;
9. the complete size/digest inventory and durable locations for large products;
   and
10. deviations, omitted inputs, unavailable arrays/networks, unexpected state,
    and the operator's signed interpretation-free notes.

The audit records supplied external evidence only after checking this identity
and completeness. Until then `SCI-AST-001-UNITY-001` remains unsupplied,
validation remains incomplete, production remains `existing_use_only`, and no
precise astrometry or WCS expansion is authorized.
