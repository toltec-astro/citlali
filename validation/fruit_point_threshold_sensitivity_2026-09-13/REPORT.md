# POINT peak-error threshold sensitivity

2026-09-13 · SCI-FRUIT v0.1 development · r0.1

## Program adherence and prior-work recovery

Follow the [charter](../../doc/scientific_contracts/README.md), [reviewed prior work](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/r0.4/PRIOR_WORK.md), [reporting scope](SCOPE.md) and [saved readout result](../fruit_point_profiled_readout_2026-09-13/SCIENTIFIC_REPORT.md). This check is JSON arithmetic only. All prior results and the parked POINT candidate remain unchanged.

## How to read the counts

The primary tables use the repaired free-Gaussian readout at 60″. P is the pixelwise reference; C is starlet. Absolute peak error is 100 × (peak/declared peak − 1); H has declared peak 100 and D has 90. Gain-ratio error is 100 × [(H/D)/(100/90) − 1]. Matched and crossed pairings stay separate. The seeds are 20260911 and 20260912; crossed pairings reuse those two realizations.

Raw counts include all finite errors. Usable counts additionally require the unchanged saved peak-use judgment, for both members of a ratio. Both retain all 12 peak or six ratio cases in the denominator. Numerical flags are displayed independently, not used to remove cases. “No” numerical completion does not silently change a saved usability judgment. Counts use unrounded errors; displayed errors have four decimal places. These four cutoffs are descriptive, not selected acceptance requirements. Numerical flags show the displayed domain / other domain; ratio flags identify the H numerator and D denominator separately.

| Quantity | Arm | Readout | Available for peak use | 5% raw ; usable | 7.5% raw ; usable | 10% raw ; usable | 15% raw ; usable |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| Absolute peak | P | Repaired free fit | 7/12 | 6/12 ; 3/12 | 9/12 ; 6/12 | 10/12 ; 6/12 | 11/12 ; 7/12 |
| Matched H/D ratio | P | Repaired free fit | 3/6 | 5/6 ; 3/6 | 6/6 ; 3/6 | 6/6 ; 3/6 | 6/6 ; 3/6 |
| Crossed H/D ratio | P | Repaired free fit | 3/6 | 3/6 ; 2/6 | 3/6 ; 2/6 | 5/6 ; 3/6 | 5/6 ; 3/6 |
| Absolute peak | C | Repaired free fit | 9/12 | 4/12 ; 4/12 | 7/12 ; 7/12 | 8/12 ; 8/12 | 10/12 ; 8/12 |
| Matched H/D ratio | C | Repaired free fit | 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 |
| Crossed H/D ratio | C | Repaired free fit | 4/6 | 3/6 ; 2/6 | 5/6 ; 3/6 | 6/6 ; 4/6 | 6/6 ; 4/6 |

The wider cutoffs expose different limitations: C matched ratios are already all inside 5%, but only four pairs are usable. All C crossed ratios are inside 10%; absolute peaks still show larger errors. One C/a1400/D_20260911 absolute error is +15.0004%, just above the literal 15% boundary. The 52″ value is inside 15%, accounting for its one extra raw success there; that peak is withheld in both tables. This sensitivity is reported rather than used to choose a domain or cutoff.

## Error distributions: every primary result retained

| Quantity | Arm | Median signed % | Median absolute % | Worst signed error % | Worst array and realization | Sorted error magnitudes % |
| --- | --- | ---: | ---: | ---: | --- | --- |
| Absolute peak | P | +1.8234 | 4.8975 | -15.5293 | a2000 D_20260912 | 0.0023, 1.5600, 1.8602, 2.0869, 4.4703, 4.6477, 5.1474, 6.9127, 7.0504, 8.1723, 13.3080, 15.5293 |
| Matched H/D ratio | P | +2.0959 | 2.0959 | +5.9610 | a1400 H20260911/D20260911 | 0.1287, 0.1695, 1.5623, 2.6296, 3.4656, 5.9610 |
| Crossed H/D ratio | P | +0.8241 | 5.2925 | +16.1820 | a2000 H20260911/D20260912 | 0.5162, 2.1645, 2.4102, 8.1749, 8.6035, 16.1820 |
| Absolute peak | C | +6.2745 | 6.3076 | +16.6949 | a1400 H_20260911 | 2.2595, 2.8100, 4.3601, 4.4742, 5.1652, 5.2314, 7.3839, 7.6152, 10.4597, 10.8551, 15.0004, 16.6949 |
| Matched H/D ratio | C | +0.0715 | 0.6505 | +1.4735 | a1400 H20260911/D20260911 | 0.2150, 0.3580, 0.5355, 0.7655, 0.7989, 1.4735 |
| Crossed H/D ratio | C | -0.4575 | 4.6247 | +7.9043 | a2000 H20260911/D20260912 | 2.1097, 3.0247, 3.6046, 5.6448, 7.0851, 7.9043 |

## Absolute peaks: signed errors at 60 arcsec

| Array | State or H/D noise pairing | Arm | Signed error % | Peak usable at 60″ | Numerical complete: fit / other domain | Warnings / limitations |
| --- | --- | --- | ---: | --- | --- | --- |
| a1100 | H_20260911 | P | +6.9127 | yes | yes / yes | none |
| a1100 | H_20260911 | C | +7.3839 | yes | yes / yes | none |
| a1100 | D_20260911 | P | +7.0504 | yes | yes / yes | shape warning |
| a1100 | D_20260911 | C | +7.6152 | yes | yes / yes | none |
| a1100 | H_20260912 | P | +4.4703 | yes | yes / yes | none |
| a1100 | H_20260912 | C | +4.3601 | yes | yes / yes | none |
| a1100 | D_20260912 | P | +4.6477 | yes | yes / yes | none |
| a1100 | D_20260912 | C | +5.1652 | yes | yes / yes | none |
| a1400 | H_20260911 | P | +8.1723 | no | yes / yes | shape warning, low peak score |
| a1400 | H_20260911 | C | +16.6949 | yes | yes / yes | shape warning |
| a1400 | D_20260911 | P | +2.0869 | no | yes / yes | shape warning, low peak score |
| a1400 | D_20260911 | C | +15.0004 | no | yes / yes | shape warning, low peak score |
| a1400 | H_20260912 | P | +1.5600 | no | yes / yes | shape warning, low peak score |
| a1400 | H_20260912 | C | +10.8551 | no | yes / yes | low peak score |
| a1400 | D_20260912 | P | -0.0023 | no | yes / yes | shape warning, low peak score |
| a1400 | D_20260912 | C | +10.4597 | no | yes / yes | shape warning, low peak score |
| a2000 | H_20260911 | P | -1.8602 | yes | yes / yes | none |
| a2000 | H_20260911 | C | +2.2595 | yes | yes / yes | none |
| a2000 | D_20260911 | P | -5.1474 | yes | yes / yes | none |
| a2000 | D_20260911 | C | +2.8100 | yes | yes / yes | none |
| a2000 | H_20260912 | P | -13.3080 | yes | yes / yes | none |
| a2000 | H_20260912 | C | -4.4742 | yes | no / yes | none |
| a2000 | D_20260912 | P | -15.5293 | no | yes / yes | low peak score |
| a2000 | D_20260912 | C | -5.2314 | yes | yes / yes | none |

## Matched-noise H/D ratios: signed errors at 60 arcsec

| Array | State or H/D noise pairing | Arm | Signed error % | Peak usable at 60″ | Numerical complete: fit / other domain | Warnings / limitations |
| --- | --- | --- | ---: | --- | --- | --- |
| a1100 | H20260911/D20260911 | P | -0.1287 | yes | H: yes / yes; D: yes / yes | D_20260911: shape warning |
| a1100 | H20260911/D20260911 | C | -0.2150 | yes | H: yes / yes; D: yes / yes | none |
| a1100 | H20260912/D20260912 | P | -0.1695 | yes | H: yes / yes; D: yes / yes | none |
| a1100 | H20260912/D20260912 | C | -0.7655 | yes | H: yes / yes; D: yes / yes | none |
| a1400 | H20260911/D20260911 | P | +5.9610 | no | H: yes / yes; D: yes / yes | H_20260911: shape warning, low peak score; D_20260911: shape warning, low peak score |
| a1400 | H20260911/D20260911 | C | +1.4735 | no | H: yes / yes; D: yes / yes | H_20260911: shape warning; D_20260911: shape warning, low peak score |
| a1400 | H20260912/D20260912 | P | +1.5623 | no | H: yes / yes; D: yes / yes | H_20260912: shape warning, low peak score; D_20260912: shape warning, low peak score |
| a1400 | H20260912/D20260912 | C | +0.3580 | no | H: yes / yes; D: yes / yes | H_20260912: low peak score; D_20260912: shape warning, low peak score |
| a2000 | H20260911/D20260911 | P | +3.4656 | yes | H: yes / yes; D: yes / yes | none |
| a2000 | H20260911/D20260911 | C | -0.5355 | yes | H: yes / yes; D: yes / yes | none |
| a2000 | H20260912/D20260912 | P | +2.6296 | no | H: yes / yes; D: yes / yes | D_20260912: low peak score |
| a2000 | H20260912/D20260912 | C | +0.7989 | yes | H: no / yes; D: yes / yes | none |

## Crossed-noise H/D ratios: signed errors at 60 arcsec

| Array | State or H/D noise pairing | Arm | Signed error % | Peak usable at 60″ | Numerical complete: fit / other domain | Warnings / limitations |
| --- | --- | --- | ---: | --- | --- | --- |
| a1100 | H20260911/D20260912 | P | +2.1645 | yes | H: yes / yes; D: yes / yes | none |
| a1100 | H20260911/D20260912 | C | +2.1097 | yes | H: yes / yes; D: yes / yes | none |
| a1100 | H20260912/D20260911 | P | -2.4102 | yes | H: yes / yes; D: yes / yes | D_20260911: shape warning |
| a1100 | H20260912/D20260911 | C | -3.0247 | yes | H: yes / yes; D: yes / yes | none |
| a1400 | H20260911/D20260912 | P | +8.1749 | no | H: yes / yes; D: yes / yes | H_20260911: shape warning, low peak score; D_20260912: shape warning, low peak score |
| a1400 | H20260911/D20260912 | C | +5.6448 | no | H: yes / yes; D: yes / yes | H_20260911: shape warning; D_20260912: shape warning, low peak score |
| a1400 | H20260912/D20260911 | P | -0.5162 | no | H: yes / yes; D: yes / yes | H_20260912: shape warning, low peak score; D_20260911: shape warning, low peak score |
| a1400 | H20260912/D20260911 | C | -3.6046 | no | H: yes / yes; D: yes / yes | H_20260912: low peak score; D_20260911: shape warning, low peak score |
| a2000 | H20260911/D20260912 | P | +16.1820 | no | H: yes / yes; D: yes / yes | D_20260912: low peak score |
| a2000 | H20260911/D20260912 | C | +7.9043 | yes | H: yes / yes; D: yes / yes | none |
| a2000 | H20260912/D20260911 | P | -8.6035 | yes | H: yes / yes; D: yes / yes | none |
| a2000 | H20260912/D20260911 | C | -7.0851 | yes | H: no / yes; D: yes / yes | none |

## 52-arcsec sensitivity, kept separate

The next counts use 52″ errors on every corresponding case. The usable column conditions on the same saved 60″ judgment; it is not independent 52″ availability and cannot be substituted to improve the primary result.

| Quantity | Arm | Readout | Available for peak use | 5% raw ; usable | 7.5% raw ; usable | 10% raw ; usable | 15% raw ; usable |
| --- | --- | --- | ---: | --- | --- | --- | --- |
| Absolute peak | P | Repaired free fit | 7/12 | 6/12 ; 3/12 | 9/12 ; 6/12 | 10/12 ; 6/12 | 11/12 ; 7/12 |
| Matched H/D ratio | P | Repaired free fit | 3/6 | 5/6 ; 3/6 | 6/6 ; 3/6 | 6/6 ; 3/6 | 6/6 ; 3/6 |
| Crossed H/D ratio | P | Repaired free fit | 3/6 | 3/6 ; 2/6 | 3/6 ; 2/6 | 5/6 ; 3/6 | 5/6 ; 3/6 |
| Absolute peak | C | Repaired free fit | 9/12 | 4/12 ; 4/12 | 7/12 ; 7/12 | 8/12 ; 8/12 | 11/12 ; 8/12 |
| Matched H/D ratio | C | Repaired free fit | 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 | 6/6 ; 4/6 |
| Crossed H/D ratio | C | Repaired free fit | 4/6 | 3/6 ; 2/6 | 5/6 ; 3/6 | 6/6 ; 4/6 | 6/6 ; 4/6 |

All 52″ signed errors, original free fits, fixed-original-geometry coefficients, truth-template coefficients and safeguards remain in [every error and flag](EVERY_ERROR.md). [Counts and ordered errors](COUNTS_AND_ORDERED_ERRORS.md) retain the complete sorted lists with case identities and per-array summaries. The truth-template coefficient remains a different measurement from the free-profile peak; no diagnostic coefficient is assigned operational usability.

## Limits and preservation

These summaries describe small, dependent saved populations. No new cutoff is recommended. They do not qualify amplitude bias, uncertainty, or production performance. Source presence, centroid use, shape/support warnings, peak-score and domain checks remain exactly as saved in each contributing measurement. Null safeguards retain undefined peak percentage error. The one H/D independent numerical-completion miss (C/H_20260912/a2000 at 60″) remains flagged, including both ratios containing that numerator. The earlier runtime failures and incomplete numerical qualification remain.

No map or timestream was read; no fitter, optimizer, PTC, feedback or new observation was run. [Input bindings](INPUT_BINDINGS.json), [all machine-readable errors](ERROR_RECORDS.json), [summary](SUMMARY.json) and [verification](VERIFICATION.json) preserve the arithmetic and provenance. This reporting packet leaves all previous packets untouched.
