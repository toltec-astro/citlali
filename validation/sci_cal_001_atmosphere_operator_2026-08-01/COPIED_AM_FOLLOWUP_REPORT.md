# SCI-CAL-001 copied AM follow-up report

## Status

This deterministic follow-up inventories the newly copied AM 12.2 AMC inputs and NPZ suite and performs a post-discovery, non-blinded numerical stress comparison. It does **not** establish identity with the historical Citlali atmosphere generator, authorize an operator, or declare an operational opacity/elevation domain.

The analysis never reads the nearby Dataverse uploader logs. Those logs are excluded because they are not scientific model inputs and may contain credentials.

## Copied products

The 25 AMC inputs total `121065` bytes. Their frozen sha256sum-record aggregate is `d3e4d9e1c095ffafb77b22a7d72a988335f36e476e240aadc27b8c23ef0f3bde` and their independent basename/NUL/raw-digest aggregate is `b7dd766852b4f422bdc861337e04d8f0184732045ea1a06a962560e86d2ce87c`. Every staged input exactly matches its AM 12.2 cookbook copy; per-file byte counts and SHA-256 identities are machine-readable in the manifest and product inventory.

The frozen suite contains `25` NPZ products totaling `1440066950` bytes. Its canonical manifest SHA-256 is `18dfd96f4438151197d3b6be5201476f7a71710363d81ec49c801101fa12b3ac`. Every product has elevations 10--80 degrees in 2-degree steps and spectra from 0--500 GHz in 0.01-GHz steps, with direct `atmTaun` retained.

All 20 DJF/JJA/MAM/SON products exactly match their explicit TolTECA seasonal datafile ID and MD5 identities at registry commit `25ccce10bfb50145424c88257a584ab92486ddf1`. The five annual products have no matching generic registry identity. The generic `am_q25`, `am_q50`, `am_q75`, and `am_q95` registry artifacts are separate products and are not aliases for any copied seasonal or annual file.

## Direct AM output validation

All `900` copied raw DAT outputs parse as `50001` five-column numeric rows and identify `am version 12.2 (build date Aug 26 2022 19:20:13)`. Their canonical aggregate SHA-256 is `b9bcdb36952444f4db33549fa621318c5f757dbe36c4b6a11addceb46ec95053`, computed as UTF-8 concatenation in relative-path bytewise sort order of relative_path<TAB>bytes<TAB>sha256<LF>.

For every DAT file, zenith angle is mapped to NPZ elevation by `elevation = 90 deg - ZA`. Frequency, direct tau, transmission, and RJ temperature match the corresponding NPZ column exactly for every row. The fifth brightness-temperature column is present and finite in all files; the NPZ omits Tb, so no Tb equality comparison is claimed or invented.

Every complete grid is followed by an unresolved-line warning and a Slurm exit-code-1 footer. The unresolved-line count distribution is `{"86": 324, "87": 540, "88": 36}`; `3` files also contain a Slurm step-creation retry notice. These historical nonzero return footers are retained as provenance and are not reclassified as clean successful runs.

The modified-secant T225-at-80 coordinate places `16` profiles inside the exact legacy q0--q95 support. `9` profiles are excluded without extrapolation: `LMT_DJF_95`, `LMT_JJA_50`, `LMT_JJA_75`, `LMT_JJA_95`, `LMT_MAM_95`, `LMT_SON_75`, `LMT_SON_95`, `LMT_annual_75`, `LMT_annual_95`.

## Legacy identity comparison

Annual, DJF, MAM, JJA, and SON q25/q50/q75 products were each compared with the recovered same-percentile legacy grid over all 50,001 frequencies and 31 common elevations. None is content-identical. The table reports maximum and RMS transmission and Rayleigh-Jeans differences without assigning a best or closest family. The expected historical q95 MD5 remains `0ca7b331823237767d26016d19bffb3d`; no q95 common-grid comparison is invented and none of the copied products is substituted for it.

`copied_am_annual_fit_coefficients.csv` records all 84 annual q25/q50/q75/q95 degree-six coefficient values (four profiles by three bands by seven descending powers), including unrounded binary64 values, explicit eight-decimal values, and comparison with the repair-base literals. This is a deterministic copied-family fit diagnostic, not generic-q identity evidence.

Across all same-percentile copied families, the largest exact repair-base anchor correction difference is `71.436439%` for `LMT_JJA_75/a1100`. This is an identity diagnostic, not an interpolation result.

Protocol identity is resolved by `FOLLOWUP_STUDY_DEVIATION_LOG.md` (SHA-256 `a3df86366c7869579b3255d9ea8f95cf6827e78018e0a2a83a1640360be1b036`). Annual-anchor Study C `v1` was stopped; diagnostic C1 evaluates only the already defined legacy-anchor `v0` candidates. The clarification does not authorize or reinterpret a successor.

## In-support operator stress

Truth is the copied direct line-of-sight `atmTaun` at monochromatic 272.73, 214.29, and 150 GHz. The interpolation coordinate is zenith tau225 derived from copied T225 at 80 degrees using the repair-base modified secant. Both candidates use the fixed linear q0--q25 LOS-tau segment; above q25 they use either piecewise linear or PCHIP interpolation through the exact repair-base fitted surfaces.

| Candidate | Band | Maximum correction error | P95 correction error | Worst profile/elevation | PL--PCHIP maximum difference | Provisional 1% stress result |
| --- | --- | ---: | ---: | --- | ---: | --- |
| `piecewise_linear_los_tau_v0` | `a1100` | `1.738766%` | `0.731837%` | `LMT_MAM_75` / `20.0 deg` | `0.019585%` | `false` |
| `piecewise_linear_los_tau_v0` | `a1400` | `0.771257%` | `0.261395%` | `LMT_MAM_75` / `20.0 deg` | `0.015098%` | `true` |
| `piecewise_linear_los_tau_v0` | `a2000` | `0.905890%` | `0.532755%` | `LMT_MAM_75` / `80.0 deg` | `0.182168%` | `true` |
| `pchip_los_tau_v0` | `a1100` | `1.738068%` | `0.730044%` | `LMT_MAM_75` / `20.0 deg` | `0.019585%` | `false` |
| `pchip_los_tau_v0` | `a1400` | `0.770085%` | `0.260285%` | `LMT_MAM_75` / `20.0 deg` | `0.015098%` | `true` |
| `pchip_los_tau_v0` | `a2000` | `0.862716%` | `0.485404%` | `LMT_MAM_75` / `80.0 deg` | `0.182168%` | `true` |

These results are useful provisional representation stress evidence only. The profiles and candidates were inspected before this analysis, the convention is monochromatic rather than band integrated, and the copied AM 12.2 suite is not the historical q-model lineage. Passing one percent here is not per-sample physical photometric accuracy and does not address the separate 5--10% absolute or approximately 5% repeatability observational gates.

## Disposition

Retain piecewise-linear LOS tau as the baseline and PCHIP as the challenger for further declared studies. Do not authorize either candidate or an operational domain from this follow-up. Historical q95 provenance, the owner-selected spectral convention, preregistered independent runs, and the SCI-ALIGN-001 sample-identity eligibility dependency remain separate gates.
