# FRUIT EL-F8 result manifest r0.4

Test ID: `SCI-FRUIT-EL-F8-PENALTY-PLACEMENT-DECOMPOSITION-R0.1`

Status: **valid completed development result**

## Controlling and interpretive evidence

| Object | Bytes | SHA-256 |
|---|---:|---|
| `SCIENTIFIC_OWNER_EL_F8_AUTHORIZATION_2026-09-03.md` | 854 | `38fcf2ee4a0835af9accf100f13570c2930b511f3a724cd75b1b2e338cace8c0` |
| `REGISTRATION_MANIFEST_R0.4.md` | 2871 | `ec422c9f2ec716092b3772ca43a8f0913abbe795dac0008eaeb2c61997406e3d` |
| `FROZEN_INPUTS_R0.4.md` | 4028 | `17f1aa1dd1205aa6e9f6c81cb5dcc9b369850c49f109e65ee7d8a37d03ba4ede` |
| `ANALYSIS_MANIFEST_R0.4_ANALYSIS_R0.4.yaml` | 3886 | `7228a6dd68f697b98341a140b2fabe8002b4472168b8ed6c104d25b17805ab67` |
| `ANALYSIS_PROVENANCE_R0.4.yaml` | 5716 | `736a17273d9450d32e3205d8bc2dfb13b2a4491c8224af275c3c132b0517c628` |
| `EXECUTION_RESULT_R0.4.md` | 1746 | `881b0d687fa7a8731d90330d6ce80f7434139bd75277e9d19b4afb89c7098ff8` |
| `SCIENTIFIC_INTERPRETATION_R0.4.md` | 7877 | `3bf39b2cab40c1b4f0b1d27cf11a79baf6bb7f8dbf2e6699d68913dd6228dbf9` |

The authorization file is in
`doc/scientific_contracts/packages/SCI-FRUIT/v0.1/empirical_lane`; all other
objects in the table are in this validation directory.

## Repository numerical products

| Object | Bytes | SHA-256 |
|---|---:|---|
| `COMPONENT_METRICS_R0.4.csv` | 5258 | `922e74507d7b0795542eaa61bc2a814697701a54da713e073a3403f0d0980aea` |
| `CROSS_TERMS_R0.4.csv` | 10058 | `d0971408e99dff471b035ab198bdc57950408576b12e4c45395e1827101b841d` |
| `DECOMPOSITION_RESULT_R0.4.json` | 3427 | `aafe17baa18f902e963f01115b7335f089fe2a5a1d20423830879e3a8e0dc95b` |
| `PRIMARY_EXECUTION_R0.4.csv` | 299 | `9606aa5dda68b659655ebd6d89a9f100143457aa258583e8589b3c50ec6c2be0` |
| `TRIGGER_PIXELS_R0.4.csv` | 637 | `9444425fa2de289b44373d80725d153829bb15642fc97c66b63bb435f1114d25` |
| `PENALTY_PLACEMENT_DECOMPOSITION_R0.4.png` | 821207 | `3ae6c33b8db210292a757d0a7943316ba45763a9a7c6af9f32e1b8c5f2148045` |

## Complete external component products

The complete FITS bundles remain in the frozen external analysis directory:

`/Users/gwilson/work_toltec/local_data/fruit-development/fruit-el-f8-penalty-placement-r0.1/r0.4/analysis-r0.4/component-maps`

| Object | Bytes | SHA-256 |
|---|---:|---|
| `point_123424_a1100_el_f8_components_r0.4.fits` | 10226880 | `06572aeb78ea27b5c3f14270911a10f82c8e06d124897dbaa2618ac0a86c7ee2` |
| `point_123424_a1400_el_f8_components_r0.4.fits` | 10226880 | `0c4d5a3b88086060f09acf2245339a5b4b70cd1d947f1d88f33a2ed2da70f989` |
| `point_123424_a2000_el_f8_components_r0.4.fits` | 10226880 | `8b58e738ec16fc2b73b585d3abeb49c9d4e06d734c85ae91479788b65e8ec7aa` |

Each retains the complete `T_current`, `T_map`, `D_current`, `D_map`, and `Q`
component maps and their registered fixed-kernel residuals on the unchanged
355 by 357 grid. They remain outside Git because FITS is misclassified as text
by the repository's current attributes; their immutable hashes and the
repository plot/tables preserve exact identity and ready review.

## Result and validity

- the mechanism classification is **mixed, direct contribution larger** in
  the registered Neptune and annular regions;
- all nine current-placement signal/kernel/weight planes are bitwise
  compatible with EL-F5 and all scientific checkpoint values match;
- the two map-branch checkpoint transformations changed only the registered
  placement policy field and passed independent machine and analysis audits;
- units, WCS/grid, normalization, support, paired configurations, application
  accounting, and the exact `Q` identity all passed;
- all four trajectories completed without replacement and with zero
  unexpected error or critical messages;
- aggregate wall time was `124.58 s`, peak resident memory was
  `860,274,688` bytes, and retained external storage remained below the
  registered 8 GiB bound; and
- all ten analysis-output hashes in `ANALYSIS_PROVENANCE_R0.4.yaml` match the
  retained repository or external copies above.

R0.1 stopped before execution, R0.2 and R0.3 stopped before a complete
four-trajectory result, and three preliminary analysis attempts stopped
before a complete result. Their immutable abort/partial records remain in
this directory. No partial scientific product was substituted into R0.4.

## Repository verification

- all 624 enabled CTest cases passed; the one pre-existing unrelated case
  remained disabled;
- all 240 baseline and FRUIT-loop Python tests passed;
- the complete required configuration preflight passed;
- the affected Python sources and tests passed Ruff and byte compilation;
- every R0.4 YAML file parsed, the retained JSON and external FITS products
  reopened, and the analysis provenance hashes were rechecked; and
- the repository whitespace check passed.

The result records no production-policy selection, detector judgment,
threshold or soft-factor selection, recurrence qualification, Gate-D launch,
Stage B authoring, or Unity activity. Any follow-on study requires its own
authorization.
