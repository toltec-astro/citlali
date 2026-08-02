# SCI-CAL-001 local atmosphere provenance inventory

## Search boundary

The search was read-only. Path, filename, type, size, and digest inventory covered the current Citlali worktree, locally available TolTEC-related repositories directly under `/Users/gwilson/GitHub`, and every regular file below `/Users/gwilson/work_toltec/local_data` without following symlinks. Content search covered that boundary except for the explicitly excluded AM Dataverse uploader logs described below. After the initial inventory, the owner staged `/Users/gwilson/work_toltec/local_data/AM`; that tree was inspected locally under the same rule. Unity and the recorded TolTECA HTTP data endpoint were not accessed. No sibling repository or coordination registry was edited.

Repository filename and allowed-content searches used `rg`/`rg --files` for q-model names, atmosphere/opacity/tau225 terms, raw-grid formats, passbands, `am`, MERRA, airmass, nominal frequencies, source literals, and distinctive polynomial coefficients. The AM tree's Dataverse uploader logs remained in the path/filename inventory but were deliberately excluded from content reads because they are not scientific inputs and may contain credentials. No uploader-log content or credential was copied into this package.

The relevant local repository roots included Citlali checkouts plus `tolteca`, `toltec_beammap`, `tolproj`, `tolapt`, `toltec-data-product-utilities`, `toltec-memoranda`, `toltec_observing`, and related TolTEC utility repositories. Positive results came from `toltec_beammap`, TolTECA history, the current Citlali history, one local dissertation, and the newly copied AM 12.2 tree.

## Current coefficient authority and history

At repair base `9aae0e669384c5c0c0dda93debc194d6b8dac787`, the coefficient authority is `include/citlali/core/timestream/rtc/calibrate.h`. Phase-0 evidence parses that file rather than creating a second authority.

The source lineage is:

| Change | Commit |
| --- | --- |
| q25/q50/q75 degree-four polynomials introduced | `598fb51d95d2bbf365d05a818f639528aaf10e70` |
| polynomial elevation coordinate changed to radians | `4ea0e86f2880c2c0d982d0c546cc944049a714f3` |
| q95 added | `3cd114ef6dcb67a8368ffd380ef681357de45426` |
| degree-six coefficients installed | `ea07ed7d0361489e1b0548978bdf2f7d52a71b77` |

The repair-base selector derives zenith q thresholds as `-log(T225 at 80 degrees)/X(80 degrees)`, where its modified-secant `X(80 degrees)` is `1.01538872688246729e+00`. The exact binary64 coordinates are:

| Model | Source/recovered T225 at 80 degrees | zenith tau225 coordinate |
| --- | ---: | ---: |
| q25 | `0.9500275` | `5.04874104674104401e-02` |
| q50 | `0.9142065` | `8.83393725904400573e-02` |
| q75 | `0.8515054` | `1.58313198574890929e-01` |
| q95 | source literal `0.7337698`; registered raw grid absent | `3.04868387190534607e-01` |

## Registered generic q-model recovery

`/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity` contains complete generic q25/q50/q75 NPZ grids. Its observed worktree head was `958a2a15f43189846a24556a63ef908da789c7b8`; the relevant vendored content is tied to commit `cedd5345e8d29546f4103f149527e09a9c68a412`.

| File | SHA-256 | MD5 | TolTECA ID |
| --- | --- | --- | ---: |
| `amLMT25.npz` | `6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b` | `008d7fa69aff187a9edf419f3d961b4c` | 454 |
| `amLMT50.npz` | `1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81` | `6ec393672be8af4dfa06a3f4cf9aa32e` | 455 |
| `amLMT75.npz` | `adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e` | `d6cf4bb27008179ec491864388deac58` | 456 |

The MD5 values exactly match the generic TolTECA static-data records. Every NPZ has members `el`, `atmFreq`, `atmTRJ`, and `atmTtx`; the exact dtype and shape inventory is in `recovered_raw_grid_manifest.json`. The same registry identifies generic q95 as datafile ID 461, expected MD5 `0ca7b331823237767d26016d19bffb3d`. Those bytes remain absent locally.

`LMTAtmosphere.py` (SHA-256 `66f580b85ccbfff9152519ec644df363e4571b9263fe06849dc89aa1858e52d0`) says the grids were generated with Scott Paine's `am` and records historical support under `/data/wilson/am`. The vendored lineage points to Mapping-Speed-Calculator gitlink commit `843cb1f44ef5f9ebaa842b296a7328f9bf64b908`, but no generator, profile, or exact run record for these four **generic** registry products is present in that lineage.

## Copied AM 12.2 source and workflow recovery

The staged AM tree materially changes the local inventory: it is a complete AM 12.2 annual/seasonal calculation suite, not the missing generic q-model package.

The executable co-staged at `am-12.2/bin/am` is an x86-64 Linux ELF with SHA-256 `3fc1f71b3a025ac79f5559bdd2fbf40cf5de2aa7598cabf474f74f9a6c3b290c`. It identifies itself as AM 12.2, build date `Aug 26 2022 19:20:13`, built with GCC 9.4 on Ubuntu 20.04 with OpenMP, and that identity matches the copied output headers. Co-staging is suite-custody evidence that this ELF accompanied the local payload; it is not a producer record proving that this exact ELF generated the copied outputs, and it does not connect the copied family to the generic q products. The ELF cannot execute natively on the present Darwin arm64 host. The copied source, cookbook, and build files identify release 12.2 (June 2022); the deterministic source inventory has 135 files, 121,636,394 bytes, and aggregate SHA-256 `0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8` under `sha256(relative_path NUL file_sha256_bytes NUL)`. This aggregate identifies the local payload; it is not a cryptographic claim that the payload equals an official archive.

`Big_Atmosphere` contains all of the following:

- 25 AMC inputs totaling 121,065 bytes: annual, DJF, MAM, JJA, and SON crossed with H2O percentiles 5, 25, 50, 75, and 95. Their canonical sha256sum-record aggregate is `d3e4d9e1c095ffafb77b22a7d72a988335f36e476e240aadc27b8c23ef0f3bde`; an independent basename/NUL/raw-digest aggregate is `b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c`. The manifest preserves every per-file byte count and SHA-256.
- 900 raw DAT outputs: 25 profiles times zenith angle 10 through 80 degrees in two-degree steps, totaling 2,983,517,161 bytes with canonical manifest SHA-256 `b9bcdb36952444f4db33549fa621318c5f757dbe36c4b6a11addceb46ec95053`.
- 25 packed NPZ products totaling 1,440,066,950 bytes, with canonical manifest SHA-256 `18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac`.
- the frozen run script `01_do_am_runs.sh`, SHA-256 `02d64a26c85f615bb194abd6102206f5cef29267599c78d4318dc327b7ce12a3`;
- the command printer `generateAmModels.py`, SHA-256 `29b5445f18463fee872cfa863e6c7799647980294ca2c85432aceb10ed8262a6`;
- the NPZ packer `make_npz.py`, SHA-256 `3a1c7b5283f03230a0d572620b4eca1a4859d61ca8c2b9786a67f4026e2717b5`.

The exact AM argv body is:

```text
am PROFILE 0 GHz 500 GHz 10 MHz ZA deg 1.0
```

The historical shell adds `srun` and redirects combined AM text output into one DAT file per profile/zenith angle. The AMC files request `f`, line-of-sight `tau`, `tx`, `Trj`, and `Tb`. AM realizes a plane-parallel, no-refraction atmosphere with `g=980.665 cm s^-2`, `T0=2.7 K`, and unit `Nscale troposphere h2o`. Every AMC describes LMT at latitude 18.986 degrees and longitude -97.314 degrees. The profiles cite MERRA-2 `inst3_3d_asm_Np` version 5.12.4, the 2007--2016 climatology, access date 2017-02-01, per-pressure-level H2O percentile statistics, and median O3. They contain explicit pressure, temperature, H2O, and O3 layers down to a 590-mbar base; they do not encode an altitude or geodetic datum.

`make_npz.py` reads each DAT after the 466-line AM configuration header, reverse-sorts the ZA-tagged filenames, converts `el=90-ZA`, and saves uncompressed float64 arrays `el`, `atmFreq`, `atmTRJ`, `atmTtx`, and `atmTaun`; `Tb` is not retained in the NPZ. Direct validation found 50,001 rows and exact DAT-to-NPZ equality for frequency, tau, transmission, and Rayleigh-Jeans temperature in all 900 outputs.

Every copied output is numerically complete but carries AM's unresolved-narrow-line warning; the historical `srun` footer records exit status 1. Warning counts are 86--88. Three outputs also record a Slurm busy-retry message. These facts prohibit describing the copied runs as clean software successes, even though their numerical grids are complete and internally reproducible.

## Registry product-identity distinction

TolTECA registry commit `25ccce10bfb50145424c88257a584ab92486ddf1`, object SHA-256 `5f117c3e5644faf3141ff647ec256f0f0404b9d0ebc1b16218222ee5daed8b72`, resolves the product-alias question, not the historical generator-lineage question:

- all 20 copied DJF/MAM/JJA/SON NPZ files exactly match their explicit seasonal registry IDs and MD5 values;
- the five copied annual NPZ files have no matching generic registry identity;
- generic IDs 454, 455, 456, and 461 are separate products, not aliases for annual or seasonal files.

Accordingly, no copied annual or seasonal q95 file may be substituted for generic q95 ID 461. Annual, DJF, MAM, JJA, and SON q25/q50/q75 were each compared with the same-percentile generic grid over all 50,001 frequencies and 31 common elevations; none is content-identical. The 84 annual degree-six coefficient values, their explicit eight-decimal forms, and repair-base comparisons are preserved separately. No family is promoted as "closest" by an unregistered aggregate metric. At q95, copied DJF `T225(80 deg)=0.7301993` versus legacy source literal `0.7337698`; the resulting modified-secant tau225 coordinates differ by about 1.576%. These numerical and product-identity differences do not establish whether the same AM 12.2 generator lineage produced the generic products.

The frozen preregistration named `v1` only for the annual-anchor Study C that stopped after the registered-product identity mismatch. Post-discovery diagnostic C1 evaluates the already-existing legacy-anchor `piecewise_linear_los_tau_v0` and `pchip_los_tau_v0` surfaces. `FOLLOWUP_STUDY_DEVIATION_LOG.md` records that clarification without editing the frozen protocol or changing C1 numerics. Historical generic-generator association remains not established.

## Native reproduction and frequency-resolution evidence

A source-only AM 12.2 native rebuild was made outside the evidence package using GCC 15.2. Its Darwin arm64 executable has SHA-256 `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`. This distinct build is successor reproduction evidence only; it does not replace or establish custody of the copied Linux executable, and neither executable is tied by evidence to the four generic q products.

Preregistered annual-q95 smoke cases at ZA 10 and 70 reproduced all 50,001 copied values exactly for frequency, tau, transmission, `Trj`, and `Tb`, so the staged check continued over all five annual profiles and all 36 zenith angles. All 180 canonical cases match the copied numeric data lines byte-for-byte and all five parsed fields exactly; every maximum absolute difference is zero. All cases retain status 1, with warning-count distribution 86 in 72 cases and 87 in 108 cases. A whole-cache POSIX writer lock, a smoke-phase completion barrier, and seven deterministic per-phase ordered cache-shard queues yield zero cache-mutation warnings, zero unknown warnings, and zero error lines. The immutable external-cache execution context has SHA-256 `8ff9af2fa844db88f94ca27585e2f33854dc38fe5422935dc57865a669e60093`; it binds the runner, actual host, C locale, compiler/build, executable, source, workflow, profiles, all 180 copied references, and execution parameters, and every sidecar binds that context. The normalized numeric-output aggregate SHA-256 is `18abf7fb57f335637c7cb2e105aea910f491d74dcd485df01c63ef759a28cd5c`; the normalized warning-bearing full-output aggregate is `fc465133e1cc2ac7458f593209dd8b0adbf320ba79a233fcf852f018883aefaf`; the metrics SHA-256 is `1d6f099383880207bca94cc0f0236a379a158a0be17e4a365b62371cb1ebca87`; and the native-regeneration manifest SHA-256 is `128d2b8481d64120be2fac020658f9f6abbe3de620438563572e6d40d8493ac4`. Raw rerun outputs and execution sidecars remain in the external cache.

An earlier shared-cache attempt was excluded from canonical evidence after 28 of 180 cases emitted cache-mutation warnings (22 `insert_as_mru`, 9 `promote_to_mru`, with overlap). Its numeric lines were also exact, but exact numerics do not erase a software-execution warning failure. A later numerically exact sharded attempt with no cache/unknown/error diagnostics was superseded because its external cache did not yet bind the complete execution context or commit normalized warning-bearing output identity. Neither predecessor supplies canonical package results.

The preregistered 140--280 GHz resolution study used DJF q5/q95, elevations 80/20 degrees, and 10/5/2/1-MHz grids. Maximum extinction-correction differences from 1 MHz were 0.000340%, 0.000360%, 0.000360%, and 0%, respectively. The 10-MHz 140--280 results differ from the copied 0--500-GHz centers by at most 0.000340%, so the bounded 0.1% numerical-resolution diagnostic passes. The unresolved-line warning and status 1 persist at finer resolution; this pass neither creates a clean run policy nor authorizes a new grid.

## Post-hoc H2O-scale provenance-hypothesis diagnostic

Frozen diagnostic P1 varied only `Nscale troposphere h2o` across each of the 25 copied profiles and matched each generic target's parsed 225-GHz transmission at elevation 80 degrees. All 100 target/profile anchors match exactly. The fitted-scale 0--500-GHz by 10-MHz, elevation-20--80-by-2-degree grid was then run directly for all 100 hypotheses; the earlier affine-in-scale construction remains ancillary screening only. The complete machine-readable record is `h2o_scale_hypothesis_manifest.json`, 99,719 bytes, SHA-256 `1316b92a06edc7dc1eb7a6752e271a7b80eb409192ad9f7bf2882cc12928d14c`.

The frozen rank rules are separate transmission RMS, Rayleigh-Jeans RMS, and, because generic q95 bytes are absent, q95 combined 93-point nominal-ratio RMS. They produce:

| Target | Rank | Copied-profile hypothesis | H2O scale | Ranked RMS | Maximum fractional correction error |
| --- | --- | --- | ---: | ---: | ---: |
| q25 | full-grid transmission | `LMT_MAM_5` | `1.81225445269332575e+00` | `5.11939193880871224e-03` | `7.79414740836802711e+00` |
| q25 | full-grid Rayleigh-Jeans | `LMT_DJF_5` | `3.01439309124786581e+00` | `7.77548133113115214e-01 K` | `8.34500816020430307e+02` |
| q50 | full-grid transmission | `LMT_MAM_25` | `9.15696647246186712e-01` | `3.23305754092318917e-03` | `9.98458439029974554e-01` |
| q50 | full-grid Rayleigh-Jeans | `LMT_DJF_25` | `2.02963214820032256e+00` | `6.04530350357074253e-01 K` | `7.36370582820754521e+02` |
| q75 | full-grid transmission | `LMT_DJF_50` | `1.88602893644962655e+00` | `1.56476455256103768e-03` | `4.83405416660586074e+00` |
| q75 | full-grid Rayleigh-Jeans | `LMT_DJF_75` | `1.01048455031671569e+00` | `4.92756783098706019e-01 K` | `1.41911867623572991e+01` |
| q95 | combined nominal-ratio surface | `LMT_DJF_25` | `6.88363302058917359e+00` | `5.41090729776348960e-03` | `1.19094929017647764e-02` |

No q25/q50/q75 full-grid hypothesis passes the provisional one-percent fractional-correction diagnostic. At the three legacy monochromatic frequencies, all 225 direct q25/q50/q75 profile/band rows pass; the worst error is `6.65829283961727556e-03` (0.6658292839617276%). No q95 combined-ratio hypothesis passes: the smallest maximum correction error is `1.11745240975796860e-02` (1.1174524097579686%, `LMT_annual_25`, rank 18 by the frozen RMS rule), while the RMS rank-one result is 1.1909492901764776%. These are post-hoc numerical diagnostics, not observational accuracy or operator-selection evidence.

The correction metric has an important asymmetric optical-depth provenance. For q25/q50/q75, the candidate side uses direct AM `atmTaun`, while the generic truth NPZs contain no tau array, so truth LOS tau is reconstructed as `-log(atmTtx)`. For q95, both candidate and repair-literal truth sides are nominal band/reference ratios and both use `-log(Tband/T225)`. The frozen report's unqualified statement that LOS tau is always authoritative and the frozen manifest's `fractional_correction_metrics_reconstruct_tau_from_tx=false`/`tau_authority` wording are therefore overbroad: they are valid only for the candidate side of the q25/q50/q75 comparisons. This package-level clarification supersedes that interpretation; the frozen runner, report, and manifest remain byte-preserved to retain execution-context and cache identity.

The canonical context SHA-256 is `05148050e96e73577ec75be525b026b5bf37bbd2a8753f8e3702fc0b6dfb2bee`; runner SHA-256 is `caa41ca105eec6df99f31d982ca69910ef2d7e1ebcbad86c96faa7d0e4cd3c2c`. Across 13,667 unique referenced runs, return-code counts are 9,792 status 0 and 3,875 accepted warning-status 1. The normalized numeric-text aggregate is `343acc6878062a433b665b0c80516212dc3b338fc77337bc9b6d1ade8196d1e1`; the normalized warning-bearing-output aggregate is `3fcfe769fab3490e7067876a55c75a06e6d17e8990f137238399d02ab246728f`. Diagnostics contain 3,875 warning-bearing runs, 139,655 unresolved-column warning lines, unresolved-line count sum 335,885, and zero other-warning or error lines. A full cache-only replay passed under the shared whole-cache lock without launching AM.

P1 narrows possible copied-profile/H2O-scale recipes after seeing the legacy targets. It closes no custody-backed generator, generic-profile construction, missing-q95, independent-intermediate-profile, spectral-choice, or operational-domain fact. Its all-direct full-grid results and its nominal operator-facing diagnostic must remain distinct.

## Recovered fit and spectral convention

Direct calculation from the generic q25/q50/q75 grids proves the production fit used:

```text
ratio(e) = atmTtx(nu_band,e) / atmTtx(225 GHz,e)
coefficients = round(numpy.polyfit(elevation_radians, ratio, 6), 8)
```

with `nu_band` 272.73, 214.29, and 150.00 GHz. All 63 rounded coefficients and all three 80-degree transmission literals match the repair-base source exactly. This is stronger numerical evidence than ambiguous historical prose about ratio direction or nominal frequency rounding.

No bandpass is integrated in the recovered lineage. Available but unused spectral artifacts are:

- `model_passbands.npz`, SHA-256 `861e6ce7af55b18c14a800defaf0b9a11099a16c307da08e391e1d8f79a39765`, MD5 `c8cae1089964f1a90ecfee36267d1fcd`, modeled passbands attributed in the source to Sean Bryan;
- TolTECA passband v1.0.0 ECSV tables: a1100 SHA-256 `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72`, a1400 `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e`, and a2000 `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff`.

Those arrays can support a newly approved band-integrated convention, but they cannot be silently assigned to the legacy coefficients. A successor needs an explicit immutable spectral convention and provenance.

## Historical narrative and grid discrepancy

`/Users/gwilson/work_toltec/local_data/doc/mmccrackan_dissertation.pdf`, SHA-256 `2aa4373aaa0394f1a79e6668047a7aecd07d4914ce162c931f495d5502a49be0`, attributes atmosphere grids to Scott Paine's July 2022 `am` release (DOI `10.5281/zenodo.6774376`), LMT q25/q50/q75/q95 conditions, and NASA MERRA-constrained layers. It describes a sixth-order elevation fit, nominal band frequencies, modified-secant coefficient 0.0012, and selection from 80-degree 225-GHz transmission.

Its prose states an elevation grid of 10 through 80 degrees in ten-degree steps. The registered generic q25/q50/q75 arrays have 20--80 degrees in two-degree steps. The copied AM 12.2 workflow explicitly generates ZA 10--80 degrees in two-degree steps, yielding elevation 10--80 degrees in two-degree steps. The copied workflow therefore explains its own 36-node products but does not resolve the generic-grid wording discrepancy or establish that it generated the legacy coefficients.

## Bounded unresolved provenance and owner choices

The local search has recovered a reproducible 25-profile AM 12.2 source/input/workflow/output suite and exact custody identities for 20 seasonal products. Post-hoc P1 has also narrowed numerical profile/H2O-scale recipes, but it resolves none of the eight successor study-definition facts or three nonblocking historical-custody facts. The search has **not** recovered the generic q95 bytes or an evidence-backed mapping from the generic q25/q50/q75/q95 products to exact generator payload, profiles, profile-construction recipe, or run record.

The owner has selected evaluation of a separately versioned AM 12.2 successor. The following remain study/adoption dependencies:

- select and version the AM 12.2 model family and profile/H2O-scale rule, spectral convention, warning/frequency policy, and independent intermediate-run design;
- generate or approve independent intermediate-opacity evidence across the q95-excluding successor study range;
- approve the operational zenith-tau225/aligned-elevation domain and fail-closed eligibility rules.

Generic q95 datafile ID 461 and historical generic-q generator/profile/run custody remain optional, nonblocking provenance dependencies for historical closure. The copied annual family is not a default. No atmospheric profile, seasonal family, q95 file, bandpass, warning policy, operator, or domain is substituted or selected by this inventory.
