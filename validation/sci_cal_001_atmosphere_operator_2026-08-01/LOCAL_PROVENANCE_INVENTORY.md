# SCI-CAL-001 local atmosphere provenance inventory

## Search boundary

The search was read-only. It covered the current Citlali worktree, locally available TolTEC-related repositories directly under `/Users/gwilson/GitHub`, and every regular file below `/Users/gwilson/work_toltec/local_data` without following symlinks. Unity and the recorded TolTECA HTTP data endpoint were not accessed. No sibling repository or coordination registry was edited.

Repository filename and content searches used `rg`/`rg --files` for q-model names, atmosphere/opacity/tau225 terms, raw-grid formats, passbands, `am`, MERRA, airmass, nominal frequencies, source literals, and distinctive polynomial coefficients. Local-data archive member names and bounded source/notebook/document/config/data content were also searched. The existing untracked scripts in `toltec_beammap` were left unchanged.

The relevant local repository roots included Citlali checkouts plus `tolteca`, `toltec_beammap`, `tolproj`, `tolapt`, `toltec-data-product-utilities`, `toltec-memoranda`, `toltec_observing`, and related TolTEC utility repositories. Positive results came from `toltec_beammap`, TolTECA history, the current Citlali history, and one local dissertation.

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
| q95 | source literal `0.7337698`; raw grid absent | `3.04868387190534607e-01` |

## Recovered atmosphere calculations

`/Users/gwilson/GitHub/toltec_beammap/src/toltec_sensitivity` contains complete q25/q50/q75 NPZ grids. Its observed worktree head was `958a2a15f43189846a24556a63ef908da789c7b8`; the relevant vendored content is tied to commit `cedd5345e8d29546f4103f149527e09a9c68a412`.

| File | SHA-256 | MD5 | TolTECA ID |
| --- | --- | --- | ---: |
| `amLMT25.npz` | `6ddffcd2c68bbc0f6d8f6470eba0d1aa81457dcc2f348fd2d7e44c9dfe48c87b` | `008d7fa69aff187a9edf419f3d961b4c` | 454 |
| `amLMT50.npz` | `1fe6dd2ab7a4d65f445e20c5a8f438eb42884836e7932d86f80c30e235710f81` | `6ec393672be8af4dfa06a3f4cf9aa32e` | 455 |
| `amLMT75.npz` | `adbb8eb974c4e2744c3efb0f627708565f954c4029d9345e4f434699e8843f8e` | `d6cf4bb27008179ec491864388deac58` | 456 |

The MD5 values exactly match TolTECA's static-data registry at `origin/main` commit `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`. Every NPZ has members `el`, `atmFreq`, `atmTRJ`, and `atmTtx`; the exact dtype and shape inventory is in `recovered_raw_grid_manifest.json`.

The same registry identifies q95 as datafile ID 461, expected MD5 `0ca7b331823237767d26016d19bffb3d`, at recorded endpoint `http://lmtdv0.astro.umass.edu/api/access/datafile/461`. Those bytes are not local and the endpoint was not contacted.

`LMTAtmosphere.py` (SHA-256 `66f580b85ccbfff9152519ec644df363e4571b9263fe06849dc89aa1858e52d0`) says the grids were generated with Scott Paine's `am` and records historical support under `/data/wilson/am`. The vendored lineage points to Mapping-Speed-Calculator gitlink commit `843cb1f44ef5f9ebaa842b296a7328f9bf64b908`, but the original generator/configuration is not present locally.

TolTECA at the same revision has a current LMT site convention in `tolteca/common/lmt/__init__.py`, SHA-256 `56113ab1ab9326c65ea07a24d8374f1c2ad6bd577ad1ba0785c01fa41d36d5fa`: longitude `-97d18m52.6s`, latitude `+18d59m10s`, and height 4640 m. No recovered raw-grid metadata ties those values or TolTECA's geometry to the historical `am` runs, so this package records them as an available comparison and does not promote them to generation inputs.

## Recovered fit and spectral convention

Direct calculation from q25/q50/q75 proves the production fit used:

```text
ratio(e) = atmTtx(nu_band,e) / atmTtx(225 GHz,e)
coefficients = round(numpy.polyfit(elevation_radians, ratio, 6), 8)
```

with `nu_band` 272.73, 214.29, and 150.00 GHz. All 63 rounded coefficients and all three 80-degree transmission literals match the repair-base source exactly. This is stronger numerical evidence than ambiguous historical prose about ratio direction or nominal frequency rounding.

No bandpass is integrated in the recovered lineage. Available but unused spectral artifacts are:

- `model_passbands.npz`, SHA-256 `861e6ce7af55b18c14a800defaf0b9a11099a16c307da08e391e1d8f79a39765`, MD5 `c8cae1089964f1a90ecfee36267d1fcd`, modeled passbands attributed in the source to Sean Bryan;
- TolTECA passband v1.0.0 ECSV tables at commit `2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`: a1100 SHA-256 `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72`, a1400 `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e`, and a2000 `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff`.

Those arrays can support a newly approved band-integrated convention, but they cannot be silently assigned to the legacy coefficients. The passband table metadata also does not preserve its named generator script locally, so a successor needs an explicit immutable convention and provenance.

## Historical narrative and discrepancy

`/Users/gwilson/work_toltec/local_data/doc/mmccrackan_dissertation.pdf`, SHA-256 `2aa4373aaa0394f1a79e6668047a7aecd07d4914ce162c931f495d5502a49be0`, attributes the atmosphere grids to Scott Paine's July 2022 `am` release (DOI `10.5281/zenodo.6774376`), LMT q25/q50/q75/q95 conditions, and NASA MERRA-constrained layers. It describes a sixth-order elevation fit, nominal band frequencies, modified-secant coefficient 0.0012, and selection from 80-degree 225-GHz transmission.

Its prose states an elevation grid of 10 through 80 degrees in ten-degree steps. The recovered NPZ arrays and the plotted node density are 20 through 80 degrees in two-degree steps. This package preserves the discrepancy. It uses the exact arrays for numerical recovery and requests the original generation directives before claiming a full rerun.

## Local-data negative result

No original atmosphere generator, `am` executable configuration, MERRA input/profile, raw q grid, fitting script/notebook, or machine-readable passband used by this coefficient lineage exists among the searched regular files below `/Users/gwilson/work_toltec/local_data`.

The local tree does contain copied TolTECA simulator YAML examples describing static q25/q50/q75 atmospheres, two realized provenance records that name q75 only, and informal analysis code with nominal frequencies. These are usage evidence, not regeneration provenance, and were not promoted to inputs.

## Bounded unresolved provenance

The missing facts are exactly enumerated in `owner_input_request.json`: q95 bytes; exact historical `am` payload and argv/configuration; profile bytes and MERRA percentile construction; site/slant-geometry inputs; generator grid/output directives; independent intermediate-profile calculations; operational domain; and the owner spectral choice. Contemporary defaults or inferred physical profiles are not acceptable substitutes.
