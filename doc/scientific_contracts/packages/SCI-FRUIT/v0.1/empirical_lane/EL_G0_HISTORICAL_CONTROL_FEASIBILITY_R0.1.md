# SCI-FRUIT v0.1 — EL-G0 Historical-Control Feasibility r0.1

Status: **exact recurrence source recovered; executable historical control not
yet reproducibly bound**

## Recovered Source Identity

| Object | Exact identity |
| --- | --- |
| historical source commit | `f70701ad488444f3e2528c6bbe3e798863c9e301` |
| historical source tree | `2009a1397bd67d615a1d6e9a8419e18fc794a81e` |
| core recurrence | `SCI-FRUIT-HISTORICAL-RECURRENCE@f70701ad` |
| `CMakeLists.txt` | 4811 bytes; SHA-256 `8a84cef756cd4cdadd3eb827b0e2615596ded05b65fd8a1b96c35cbaf571885b` |
| fruit-loop config schema source | 20399 bytes; SHA-256 `9382e2d000e91e44b8b6aa1b913fb608531b798e6c5896e4faeab524b716d228` |

The accepted Stage A record already establishes the historical operator order,
complete-map carry, original-observation rerun, residual processing, model
restoration, learned-state behavior, and restart requirements. Gate 0 does not
reinterpret that recurrence.

## Why The Source Commit Is Not Yet An Executable Control

At the historical commit, `CMakeLists.txt` fetched `kidscpp` with Git tag
`v1.x`, not an immutable commit. Tula arrived transitively, and Boost, CCfits,
FFTW, ZLIB, HDF5, compiler, C++ library, thread/runtime settings, and other
environment facts were not locked by a repository-local dependency manifest.

Therefore a fresh build of `f70701ad...` is a **reconstruction candidate**, not
automatically the historical executable. The source commit alone is
insufficient for paired-control identity.

The historical executable does expose Citlali and kidscpp version strings, and
the reduction outputs can contain copied ordered config sources, a merged
config snapshot, config hashes, runtime provenance, and product version stamps.
Those mechanisms make artifact-backed recovery possible if their products are
still available.

## Recovered Configuration Candidates, Not Selected Controls

The historical commit contains several mode templates:

| Template | Historical FRUIT setting in the template | Source SHA-256 |
| --- | --- | --- |
| `config/tolteca/science/70_pipeline.yaml` | enabled; `coadd/raw`; four iterations; upper selection; S/N 2.5; learning enabled | `273c1582ecb0e2cd906e90bad58ce8f328cee59b97570734329d5d3aeedfba52` |
| `config/tolteca/oof/70_pipeline.yaml` | disabled; `obsnum/raw`; one iteration; source-dominant thresholds; learning disabled | `bc608f70b662bb256bfa82ad83733ddd5d3cfb456acbb13a51c74e464611e72e` |
| `config/tolteca/beammap/70_pipeline.yaml` | enabled; `obsnum/raw`; two iterations; flux-selected; learning disabled | `8bab688427884051ce608688b12a755e46d7125bf0740c0506824bea103f1732` |

The corresponding v2 defaults have SHA-256 values
`803d5b0203725b40d8abafbb4b21f32a79f54804df696dab0e0fe3c9b41f080f`,
`59833ec6554a4429602a6275a35c61c41455ab5bef5768a983a53790e4086a16`,
and `815c8ac2e15527cea1cd25f05e7f54b04591a3d8629a7a1b6afda23768040f14`
for science, OOF, and beammap respectively.

These are authoring templates containing runtime placeholders and mode
defaults. They do not identify the ordered configuration actually used for one
observation, and OOF FRUIT is disabled by default. None is promoted to the
scientific control.

## Admissible Control-Recovery Paths

### Path A — Preserved Artifact (Preferred)

Bind a preserved executable by SHA-256 and version output, plus its dependency/
runtime environment, ordered input configs and merged config hash, raw inputs,
APT/calibration/ephemeris identities, parent route/grouping, map/filter policy,
threading, stopping, outputs, and required provenance. Re-execution must pass a
small deterministic reproducibility check before development comparison.

### Path B — Pinned Reconstruction

Rebuild `f70701ad...` only after every dependency and environment fact is
pinned. Call the result a reconstructed control until it is compared with an
artifact-backed anchor on common inputs and the acceptance tolerance is frozen
before the comparison.

### Path C — No Exact Control

If neither an artifact-backed control nor a validated reconstruction exists,
baseline-relative Gate-D development is unavailable. A scientifically useful
new-method study may later require an explicit successor architecture, but it
cannot claim comparison with exact historical Citlali under the accepted
framework.

## Required `HISTORICAL_CONTROL_ID`

```text
HISTORICAL_CONTROL_ID = (
  executable_sha256_and_version,
  source_and_dependency_refs,
  build_and_runtime_environment,
  ordered_config_sources_and_merged_hash,
  raw_input_apt_calibration_ephemeris_ids,
  reduction_parent_route_grouping_and_map_policy,
  recurrence_selection_learning_and_stopping_policy,
  output_response_support_and_provenance_contract
)
```

Every field remains required. Unknown is not equal to historical default.
