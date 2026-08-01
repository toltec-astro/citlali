# SCI-CAL-001 atmosphere regeneration specification v2

## Status and identity

This specification separates two scientific identities that must not be merged:

- **legacy generic-q lineage closure**: registered generic q25/q50/q75 bytes are recovered and exactly reproduce the Citlali coefficients; generic q95 and the generic products' exact generator/profile/run provenance remain absent;
- **copied AM 12.2 suite regeneration**: a complete source/input/workflow/output package is local for 25 annual/seasonal profiles and can be rerun as a separately identified AM 12.2 calculation family.

The specification is bound to repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and repair-line evidence head `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`. It does not claim that the copied AM 12.2 profiles generated the registered generic q artifacts. `regeneration_manifest.json` is the machine-readable package state and validates against `atmosphere_regeneration_manifest.schema.json`; follow-up custody and study details are preserved in the copied-AM, native-regeneration, frequency-resolution, and post-hoc H2O-scale hypothesis manifests.

No operator or operational opacity/elevation domain is authorized by this specification.

## Immutable lineage inventory

### Registered generic q artifacts

| Model | Local artifact | SHA-256 | TolTECA registry MD5 | Registry ID | Status |
| --- | --- | --- | --- | ---: | --- |
| q25 | `amLMT25.npz` | `6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b` | `008d7fa69aff187a9edf419f3d961b4c` | 454 | recovered |
| q50 | `amLMT50.npz` | `1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81` | `6ec393672be8af4dfa06a3f4cf9aa32e` | 455 | recovered |
| q75 | `amLMT75.npz` | `adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e` | `d6cf4bb27008179ec491864388deac58` | 456 | recovered |
| q95 | datafile ID 461 | unknown until supplied | `0ca7b331823237767d26016d19bffb3d` | 461 | absent; no seasonal/annual substitution |

The generic-q NPZ schema is:

- `el`: float64 `[31]`, degrees, exactly 20--80 in steps of 2;
- `atmFreq`: float64 `[50001,31]`, GHz, each column exactly 0--500 in steps of 0.01;
- `atmTtx`: float64 `[50001,31]`, dimensionless transmission;
- `atmTRJ`: float64 `[50001,31]`, Rayleigh-Jeans atmosphere temperature in kelvin.

### Copied AM 12.2 suite

The distinct copied-suite identity is:

| Role | Identity |
| --- | --- |
| copied Linux executable | co-staged x86-64 ELF; SHA-256 `3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c`; AM 12.2 build `Aug 26 2022 19:20:13`; GCC 9.4/Ubuntu 20.04/OpenMP; its identity matches copied headers, but co-staging is suite-custody evidence rather than proof that this exact ELF produced the outputs or any generic q artifact |
| source/document payload | AM 12.2; 135 inventoried source/build files, 121,636,394 bytes; aggregate SHA-256 `0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8` under `sha256(relative_path NUL file_sha256_bytes NUL)` |
| AMC inputs | 25 files, 121,065 bytes; sha256sum-record aggregate `d3e4d9e1c095ffafb77b22a7d72a988335f36e476e240aadc27b8c23ef0f3bde`; independent basename/NUL/raw-digest aggregate `b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c`; per-file identities in `copied_am_manifest.json` |
| run script | `01_do_am_runs.sh`; SHA-256 `02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3` |
| command printer | `generateAmModels.py`; SHA-256 `29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6` |
| NPZ packer | `make_npz.py`; SHA-256 `3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5` |
| raw outputs | 900 DAT files, 2,983,517,161 bytes; SHA-256 `b9bcdb36952444f4db33549fa621318c5f757dbe36c4b6a11addceb46ec95053` over sorted `relative_path<TAB>bytes<TAB>sha256<LF>` records; every file has 50,001 numeric rows |
| packed products | 25 NPZ files; 1,440,066,950 total bytes; canonical manifest SHA-256 `18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac` |

The source aggregate is a canonical local-payload identity, not a claim of equality to a separately downloaded release archive. All paths and per-file digests needed for follow-up validation are recorded in machine-readable manifests. Uploader logs are excluded from inputs and must not be read or copied.

## Copied AM 12.2 model inputs

The 25 AMC profiles are the Cartesian product of:

```text
season = annual, DJF, MAM, JJA, SON
H2O percentile = 5, 25, 50, 75, 95
```

Each profile records:

- site comment latitude 18.986 degrees and longitude -97.314 degrees;
- MERRA-2 product `inst3_3d_asm_Np` v5.12.4, 2007--2016 climatology, accessed 2017-02-01;
- the selected H2O percentile independently at each pressure level and median O3;
- explicit pressure, temperature, H2O, and O3 layers, with a 590-mbar bottom layer;
- `Nscale troposphere h2o` as the ninth argv parameter and `T0=2.7 K`.

The realized AM headers state plane-parallel geometry, no refraction, and `g=980.665 cm s^-2`. The AMC files do not encode altitude or a geodetic reference datum. Those omissions do not prevent exact reruns of the copied calculation, because AM consumes the explicit pressure-layer profiles; they do prevent silently relabeling this site convention as a more complete physical LMT metadata contract.

The five annual products have no matching generic TolTECA registry identity. All 20 seasonal products exactly match their explicit seasonal IDs at registry commit `25ccce10bfb50145424c88257a584ab92486ddf1`. This distinction is a mandatory identity check in every regeneration.

Study A compares every annual/DJF/MAM/JJA/SON q25/q50/q75 product against the same-percentile generic raw grid over 50,001 frequencies and 31 common elevations. None is content-identical; the 60-row comparison reports transmission and Rayleigh-Jeans maximum/RMS differences without inventing a cross-quantity "closest family" score. A separate 84-row table preserves every annual q25/q50/q75/q95 degree-six coefficient, its explicit eight-decimal value, and its repair-base comparison.

The frozen preregistration's `v1` identities belong only to the annual-anchor Study C that was stopped after the lineage mismatch. Diagnostic C1 evaluates the pre-existing legacy-anchor `piecewise_linear_los_tau_v0` and `pchip_los_tau_v0` surfaces. `FOLLOWUP_STUDY_DEVIATION_LOG.md`, SHA-256 `a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036`, records this clarification without editing the frozen protocol or changing candidate numerics.

## Exact copied-suite execution contract

For every AMC profile and every integer zenith angle from 10 through 80 degrees in steps of 2, execute the argv array:

```text
[
  "<am-executable>",
  "LMT_am_inputs/<profile>.amc",
  "0", "GHz", "500", "GHz", "10", "MHz",
  "<zenith-angle-deg>", "deg", "1.0"
]
```

The working directory is `Big_Atmosphere`. The original batch wrapper prepends `srun`; a local regeneration omits only that scheduler wrapper. It must record the executable digest, compiler/build identity, argv, working directory, `OMP_NUM_THREADS`, `AM_CACHE_PATH`, locale, host, stdout/stderr, return code, and generated SHA-256 for every case.

An evidentiary checker may accept return code 1 **only** when all 50,001 exact frequency rows are present, the canonical unresolved-narrow-line summary has count 86, 87, or 88, and every warning header is either that summary or its exact per-column unresolved-line record. Cache-mutation warnings, unknown warning classes, and error lines fail closed. This bounded acceptance allows comparison with the copied warning-status calculations; it is not an owner-approved operational warning policy and must never be reported as clean success.

The raw output columns are exactly:

```text
f GHz, tau neper, tx dimensionless, Trj K, Tb K
```

The copied packer reverse-sorts files by their ZA-tagged names, derives `el=90-ZA`, and writes uncompressed NPZ members `el`, `atmFreq`, `atmTRJ`, `atmTtx`, and `atmTaun`. It drops `Tb`. A regeneration comparison must first compare all five raw DAT fields independently using exact binary64 values; NPZ packing is a separate derived-artifact step and must record Python/NumPy and output digest. Direct `atmTaun`, not `-log(atmTtx)`, is the optical-depth authority across the full spectral grid because copied transmission contains exact zeros at opaque line samples.

## Native rebuild and deterministic check

The copied Linux executable is immutable historical suite evidence but is not runnable on the present Darwin arm64 host. A source-only copy was built with:

```text
make -j8 gcc-omp COMPILER_GCC=gcc-15
```

using GCC 15.2. The resulting Mach-O arm64 AM 12.2 executable has SHA-256 `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`. This is a distinct successor build, not a byte replacement for the copied Linux binary.

From the repository root, the complete annual 5/25/50/75/95 by ZA 10--80 exact-value protocol is:

```sh
/Users/gwilson/tolteca/bin/python \
  validation/sci_cal_001_atmosphere_operator_2026-08-01/run_am12_native_regeneration_check.py \
  --am-root /Users/gwilson/work_toltec/local_data/AM \
  --am-executable /private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am \
  --cache-dir /private/tmp/sci_cal_001_am12_2_native_matrix_context_v2_final_20260801_root \
  --jobs 7 --omp-threads 2 \
  --compiler-executable /opt/homebrew/Cellar/gcc/15.2.0_1/bin/gcc-15 \
  --native-build-command "make -j8 gcc-omp COMPILER_GCC=gcc-15"
```

The script first gates on annual q95 at ZA 10 and 70; all frequency/tau/tx/`Trj`/`Tb` values must match exactly before it launches the other 178 cases. The complete canonical 180-case run passes: every numeric data section is byte-identical to the copied reference and all five parsed fields are exactly equal, with zero maximum absolute difference. All 180 return codes remain 1 under the bounded warning contract; 72 have warning count 86 and 108 have count 87. This is not a clean software-success claim. One whole-cache nonblocking POSIX writer lock excludes other processes. Within the nonoverlapping smoke and remaining-matrix phases, one ordered worker queue owns each of seven deterministic matrix-index shards; cache-mutation, unknown-warning, and error-line totals are all zero.

Before AM executes, the runner writes immutable `execution_context.json`, SHA-256 `8ff9af2fa844db88f94ca27585e2f33854dc38fe5422935dc57865a669e60093`. Its complete content is copied into the committed manifest. It binds the runner SHA-256, actual run host, pinned `LANG=C`/`LC_ALL=C`, compiler identity, build command, both executable identities, 135-file AM source inventory, frozen workflow, five annual profiles, all 180 copied reference grids, run scope, argv, job/thread parameters, cache topology, and normalization rule. Every raw-output sidecar binds that context digest. Cache-only replay reconstructs the context from current immutable facts while retaining the recorded execution host, and fails rather than relabeling changed bytes or parameters.

The committed normalized numeric-output aggregate is `18abf7fb57f335637c7cb2e105aea910f491d74dcd485df01c63ef759a28cd5c`. A second aggregate, `fc465133e1cc2ac7458f593209dd8b0adbf320ba79a233fcf852f018883aefaf`, binds warning-bearing combined output after replacing only volatile runtime and dcache-counter header lines. `native_regeneration_metrics.csv` has SHA-256 `1d6f099383880207bca94cc0f0236a379a158a0be17e4a365b62371cb1ebca87`; `native_regeneration_manifest.json` has SHA-256 `128d2b8481d64120be2fac020658f9f6abbe3de620438563572e6d40d8493ac4`.

The first parallel attempt shared one AM cache and is explicitly excluded after 28 cases emitted 31 cache-mutation warnings. Its numerical lines were exact, but it does not satisfy the canonical software-execution contract. A later numerically exact and warning-class-valid sharded attempt is also superseded because its cache did not yet bind the complete immutable execution context or commit normalized warning-bearing output identity.

Cache-only deterministic verification is:

```sh
/Users/gwilson/tolteca/bin/python \
  validation/sci_cal_001_atmosphere_operator_2026-08-01/run_am12_native_regeneration_check.py \
  --am-root /Users/gwilson/work_toltec/local_data/AM \
  --am-executable /private/tmp/sci_cal_001_am12_2_native_build_20260801_root/am \
  --cache-dir /private/tmp/sci_cal_001_am12_2_native_matrix_context_v2_final_20260801_root \
  --jobs 7 --omp-threads 2 \
  --compiler-executable /opt/homebrew/Cellar/gcc/15.2.0_1/bin/gcc-15 \
  --native-build-command "make -j8 gcc-omp COMPILER_GCC=gcc-15" --check
```

The external cache holds raw outputs, execution sidecars, AM cache shards, the whole-cache lock file, and `execution_context.json`; it is not part of the committed package. Raw combined-output SHA-256 values stay there. Committed metrics retain both numeric-text digests and normalized warning-bearing output digests, so volatile runtime/cache counters cannot perturb package bytes while warnings and all nonvolatile output remain cryptographically bound. The committed metrics/manifest/report hold deterministic summaries and the complete execution-context content and digest.

Replacing `--check` with `--regenerate-from-cache` validates the same complete cache and rewrites only the three package artifacts; it never launches AM. This is the deterministic artifact-regeneration path after a completed canonical run.

## Frequency-grid policy evidence

The copied 0--500-GHz, 10-MHz grid is immutable suite lineage evidence. The preregistered diagnostic additionally evaluates 140--280 GHz at 10, 5, 2, and 1 MHz for DJF q5/q95, ZA 10/70, and exact nodes 150.00, 214.29, 225.00, and 272.73 GHz.

Maximum fractional correction differences versus 1 MHz are:

| Step | Maximum difference |
| ---: | ---: |
| 10 MHz | 0.000340% |
| 5 MHz | 0.000360% |
| 2 MHz | 0.000360% |
| 1 MHz | 0% |

The 10-MHz grid passes the preregistered 0.1% bounded numerical-resolution diagnostic. The 140--280-GHz 10-MHz runs are not byte-identical at their center nodes to the copied 0--500-GHz calculations because AM's realized range changes them by at most `3.3999942199932977e-6` in fractional correction. Warning/status 1 persists at every tested resolution, while cache/unknown warnings and error lines are zero. Therefore the copied 10-MHz grid remains immutable lineage evidence and the range-bounded study establishes only center-frequency convergence; neither warning acceptance nor a different production grid is authorized without owner approval.

## Post-hoc P1 H2O-scale hypothesis diagnostic

Diagnostic P1 varies only `Nscale troposphere h2o` in each immutable copied AMC profile. Fixed 48-iteration bisection selects the midpoint of the innermost exact parsed-transmission plateau matching each repair-base target at 225 GHz and elevation 80 degrees. All 100 target/profile anchors match, and all 100 fitted-scale hypotheses were run directly over 0--500 GHz by 10 MHz and elevation 20--80 degrees by 2 degrees. The complete record is `h2o_scale_hypothesis_manifest.json`, 99,719 bytes, SHA-256 `1316b92a06edc7dc1eb7a6752e271a7b80eb409192ad9f7bf2882cc12928d14c`; the generator SHA-256 is `caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`.

The exact frozen rank-one results are:

| Target | Rank | Copied profile | H2O scale | Ranked RMS | Maximum fractional correction error |
| --- | --- | --- | ---: | ---: | ---: |
| q25 | transmission | `LMT_MAM_5` | `1.81225445269332575e+00` | `5.11939193880871224e-03` | `7.79414740836802711e+00` |
| q25 | Rayleigh-Jeans | `LMT_DJF_5` | `3.01439309124786581e+00` | `7.77548133113115214e-01 K` | `8.34500816020430307e+02` |
| q50 | transmission | `LMT_MAM_25` | `9.15696647246186712e-01` | `3.23305754092318917e-03` | `9.98458439029974554e-01` |
| q50 | Rayleigh-Jeans | `LMT_DJF_25` | `2.02963214820032256e+00` | `6.04530350357074253e-01 K` | `7.36370582820754521e+02` |
| q75 | transmission | `LMT_DJF_50` | `1.88602893644962655e+00` | `1.56476455256103768e-03` | `4.83405416660586074e+00` |
| q75 | Rayleigh-Jeans | `LMT_DJF_75` | `1.01048455031671569e+00` | `4.92756783098706019e-01 K` | `1.41911867623572991e+01` |
| q95 | combined 93-point nominal-ratio surface | `LMT_DJF_25` | `6.88363302058917359e+00` | `5.41090729776348960e-03` | `1.19094929017647764e-02` |

No q25/q50/q75 full-grid candidate passes the provisional one-percent correction diagnostic. All 225 direct q25/q50/q75 candidate/band rows at 272.73, 214.29, and 150.00 GHz pass; the worst error is `6.65829283961727556e-03` (0.6658292839617276%). No q95 combined-ratio candidate passes: the smallest maximum error is `1.11745240975796860e-02` (1.1174524097579686%, `LMT_annual_25`, frozen RMS rank 18), and the RMS winner is 1.1909492901764776%. The full-grid provenance comparison and nominal operator-facing diagnostic are separate results.

Correction-tau provenance is asymmetric. For q25/q50/q75 the candidate side uses direct AM `atmTaun`; the generic truth NPZs have no tau member, so their truth tau is reconstructed as `-log(atmTtx)`. For q95, generic raw bytes are absent, so both candidate and repair-literal truth sides use nominal transmission ratios and reconstruct LOS ratio tau as `-log(Tband/T225)`. Consequently the frozen H2O report's unqualified direct-tau sentence and the frozen manifest's `fractional_correction_metrics_reconstruct_tau_from_tx=false` and `tau_authority` wording are overbroad candidate-side-only statements. This package-level specification supersedes that interpretation without changing the frozen P1 runner, report, or manifest bytes.

The canonical execution-context SHA-256 is `05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`. Its 13,667 unique referenced runs comprise 9,792 return-code-0 and 3,875 accepted warning-status-1 runs; other-warning and error-line totals are zero. The normalized numeric-text aggregate is `343acc6878062a433b665b0c80516212dc3b338fc77337bc9b6d1ade8196d1e1`, and the normalized warning-bearing-output aggregate is `3fcfe769fab3490e7067876a55c75a06e6d17e8990f137238399d02ab246728f`. Cache-only replay passed under a shared whole-cache lock and launched no AM process.

P1 is a post-hoc candidate-recipe search, not custody proof or an independent model holdout. It closes none of the 11 unresolved generator, profile, site, grid, q95, independent-profile, spectral, bandpass, or operational-domain facts and does not authorize an operator.

## Exact recovered legacy derivation

For each available generic q model and each raw elevation node, select exactly 225.00 GHz and the legacy band frequency:

| Band | Nominal sample frequency |
| --- | ---: |
| a1100 | 272.73 GHz |
| a1400 | 214.29 GHz |
| a2000 | 150.00 GHz |

Define:

```text
R_q,b(e) = atmTtx_q(nu_b,e) / atmTtx_q(225 GHz,e).
```

Fit `R_q,b(e)` with `numpy.polyfit(elevation_radians, R, 6)` over all 31 generic-q nodes and round each coefficient to eight decimals. This exactly reproduces all 63 repair-base q25/q50/q75 literals. No passband participates.

The selector coordinate is verified for q25--q75 as:

```text
tau225_q = -log(atmTtx_q(225 GHz, 80 deg)) / X(80 deg),
X(80 deg) = 1.01538872688246729.
```

The q25/q50/q75 coordinates are `5.04874104674104401e-02`, `8.83393725904400573e-02`, and `1.58313198574890929e-01`. The repair-base q95 coordinate `3.04868387190534607e-01` is derived from source literal `0.7337698` and remains provisional until generic q95 bytes are supplied.

To reproduce the recovered generic fits without modifying their source repository:

```sh
/Users/gwilson/tolteca/bin/python \
  validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity
/Users/gwilson/tolteca/bin/python \
  validation/sci_cal_001_atmosphere_operator_2026-08-01/recover_legacy_raw_grids.py \
  --source-dir /Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity --check
```

The first command writes normalized evidence tables in this package only. It does not rewrite sibling-repository NPZs.

## Generic-lineage closure protocol

The copied suite does not satisfy the generic-lineage request. To close that lineage:

1. Stage generic q95 datafile ID 461 and verify MD5 `0ca7b331823237767d26016d19bffb3d`, SHA-256, byte count, members, dtypes, shapes, and custody.
2. Supply or identify the exact generic-q generator/profile/run payload. Verify every byte by SHA-256 and retain its raw source without normalization.
3. Execute only its recorded argv and settings. Do not substitute copied annual/seasonal profiles, a native AM build, contemporary site defaults, or inferred H2O scaling for missing historical facts.
4. Compare regenerated q25/q50/q75 values with the registered recovered artifacts before inspecting q95. Differences remain evidence; no value is coerced.
5. Normalize all four generic models to a common long table while preserving immutable raw outputs separately.
6. Freeze any successor operator and independent intermediate-run plan before loading held-out results.

If historical generic provenance cannot be recovered, the owner may instead authorize a newly versioned AM 12.2 model family. That decision must explicitly choose annual, seasonal, season-aware, or another stated profile-construction rule. The existence of complete annual files is not approval to select annual anchors.

## Versioned continuous-operator candidates

For a sample with valid zenith `tau225` and eligible aligned elevation `e`, evaluate each q-anchor band transmission with the top-of-atmosphere pivot:

```text
X(e) = sec(z) * (1 - 0.0012 * (sec(z)^2 - 1))
T_q,b(e) = R_q,b(e) * exp(-X(e) * tau225_q)
L_q,b(e) = -log(T_q,b(e))
C_q,b(e) = exp(L_q,b(e)).
```

The full sample airmass is applied to zenith `tau225`; no Beammap-relative or reference-airmass subtraction is permitted. A continuous candidate interpolates complete line-of-sight optical depth `L`, then returns `T=exp(-L)` and `C=exp(L)`.

All candidates share the owner-approved low-opacity segment:

```text
L_b(tau,e) = (tau/tau_q25) * L_q25,b(e),  0 <= tau <= tau_q25.
```

Above q25, the frozen candidates are:

- `piecewise_linear_los_tau_v0`: adjacent-anchor affine interpolation in zenith `tau225`; baseline because it adds no unmeasured curvature;
- `pchip_los_tau_v0`: shape-preserving PCHIP through q25/q50/q75/q95; challenger;
- `cubic_through_anchors_los_tau_v0`: unconstrained exact-anchor stress test, not recommended for selection.

Requests outside a declared opacity/elevation domain, non-finite inputs, invalid aligned-elevation eligibility, or absent model support fail closed. There is no extrapolation.

## Selection and validation gates

Candidate evaluation must report, by band and across the complete owner-declared domain:

1. exact-anchor LOS-tau error and low-opacity identity error;
2. finite values, `L>=0`, `0<T<=1`, and `C>=1`;
3. continuity at every anchor, including one-sided binary64 `nextafter` probes;
4. nondecreasing `L` with increasing zenith opacity;
5. nonincreasing `L` with increasing elevation, with the known q95/a2000 feature reported rather than hidden;
6. fractional correction error `abs(exp(L_candidate-L_truth)-1)` at raw anchors and preregistered independent intermediate/held-out runs;
7. maximum, 95th percentile, median, RMS, signed range, and error location.

Exact-anchor, finite-positive-transmission, continuity, opacity-monotonicity, and fail-closed-support gates must pass. Elevation monotonicity must either pass or receive an explicit owner scientific disposition supported by recovered raw q95 and independent model evidence. The known `0.839827%` q95/a2000 wrong-way feature is diagnostic rather than automatically release-blocking, but it may not be silently waived.

At most one-percent correction error is a provisional numerical representation-fidelity gate. It is not per-sample physical photometric accuracy. The post-discovery AM 12.2-family stress already fails one percent in a1100 for both piecewise-linear (1.738766%) and PCHIP (1.738068%), while passing in a1400 and a2000. Because the tested profiles are a different lineage and the study was non-blinded, this result neither selects nor rejects a production model; it shows that owner-selected independent validation is still required.

Software correctness, representation fidelity, 5--10% absolute observational accuracy, and approximately 5% observation-to-observation repeatability are separate gates. More samples do not remove shared calibrator, Beammap-extinction, selector, or airmass systematics.

## Owner decisions and remaining input request

Only these items remain open:

- generic q95 datafile ID 461 bytes and custody;
- historical generic-q generator/profile/run identity, **or** explicit approval of a separately versioned AM 12.2 successor family and profile rule;
- monochromatic legacy parity versus a newly versioned band-integrated convention; if band integrated, immutable passbands, weighting, normalization, and quadrature;
- the operational warning/frequency policy for any successor;
- preregistered independent intermediate profiles/runs under the selected physical construction rule;
- operational zenith-tau225 and aligned-elevation limits, endpoint inclusion, and out-of-domain behavior.

No profile family, q95 artifact, bandpass, frequency policy, operator, or domain is chosen silently.

## Aligned-elevation dependency

`SCI-CAL-001-XAUD-001` remains an open, held-for-CAL-re-audit dependency. Atmosphere evaluation may consume aligned elevation only with explicit ordered sample identity, timing gap/interpolation origin, acquired duration, and original-versus-synthesized eligibility. This constraint changes neither the atmosphere equations nor this regeneration scope.
