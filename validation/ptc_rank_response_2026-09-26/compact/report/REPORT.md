# PTC ranks 5, 10 and 15: POINT 152430/a2000

Execution: completed_matrix. Original-input RTC/CAL parent generations: 1; trajectories: 12; feedback attempts: 120; noise packages: 3; endpoint analyses: 3; ideal controls: 9.

Rank10 remains the reference. Measurement differences are method sensitivity, not absolute flux truth. Conditional scales are not paired p-values, blank-field sensitivity, or a full adaptive uncertainty law. RTC/CAL transfer was not tested.

| Rank | Branch | Status | Last cycle | Reason |
|---|---|---|---:|---|
| 10 | REF | completed | 10 |  |
| 10 | C-core-plus | completed | 10 |  |
| 10 | C-core-minus | completed | 10 |  |
| 10 | C-shoulder | completed | 10 |  |
| 5 | REF | completed | 10 |  |
| 15 | REF | completed | 10 |  |
| 5 | C-core-plus | completed | 10 |  |
| 5 | C-core-minus | completed | 10 |  |
| 5 | C-shoulder | completed | 10 |  |
| 15 | C-core-plus | completed | 10 |  |
| 15 | C-core-minus | completed | 10 |  |
| 15 | C-shoulder | completed | 10 |  |

## Unmodified reference measurements

Peak/template: mJy/nominal beam. Aperture: mJy/nominal beam arcsec². Widths and positions: arcsec. Fixed apertures/template are common across ranks; fitted source parameters remain free.

| Rank | Stage | Peak | Major / minor FWHM | Template | Aperture 2 |
|---|---|---:|---|---:|---:|
| 5 | M0 | 605 | 9.6991 / 8.70976 | 613.77 | 50542.9 |
| 5 | T1 | 717.61 | 9.9262 / 9.09784 | 751.592 | 70153.5 |
| 5 | T3 | 747.916 | 10.0365 / 9.25408 | 793.663 | 76799.8 |
| 5 | T5 | 749.69 | 10.0434 / 9.26338 | 796.166 | 77195.5 |
| 5 | T10 | 749.802 | 10.0438 / 9.26398 | 796.326 | 77220.8 |
| 10 | M0 | 564.303 | 9.59122 / 8.55217 | 564.303 | 44497.9 |
| 10 | T1 | 701.182 | 9.89984 / 9.02793 | 731.003 | 67679.7 |
| 10 | T3 | 746.389 | 10.1106 / 9.27538 | 795.807 | 77944.7 |
| 10 | T5 | 750.193 | 10.1352 / 9.30628 | 801.989 | 79029.9 |
| 10 | T10 | 750.604 | 10.1376 / 9.30892 | 802.621 | 79135.2 |
| 15 | M0 | 515.716 | 9.38642 / 8.25201 | 500.774 | 36980.4 |
| 15 | T1 | 677.132 | 9.70375 / 8.88256 | 693.782 | 62205.8 |
| 15 | T3 | 745.419 | 9.96674 / 9.22134 | 787.301 | 76250.2 |
| 15 | T5 | 753.581 | 10.0142 / 9.26832 | 799.569 | 78241.1 |
| 15 | T10 | 754.723 | 10.0233 / 9.28213 | 801.668 | 78638.7 |

## Endpoint response and conditional uncertainty

Actual-minus-ideal is the departure from own REF plus the ordinary-MAP injection control. Source percentages use the full own REF measurement, not the artificial increment. Compact plus/minus gains are separate; distorted shoulder response has no scalar uncertainty correction.

| Rank | Signal | Q | Actual − ideal | % full source | / conditional σ | Gain | σ | σ equivalent |
|---|---|---|---:|---:|---:|---:|---:|---:|
| 5 | C-core-plus | peak | 0.11011 | 0.0146851 | 0.0153235 | 1.00147 | 7.18567 | 7.17515 |
| 5 | C-core-plus | fixed_template | -0.35252 | -0.0442683 | -0.0398815 | 0.995609 | 8.83918 | 8.87817 |
| 5 | C-core-plus | aperture_2 | -175.613 | -0.227417 | -0.0578435 | 0.978117 | 3036.01 | 3103.93 |
| 5 | C-core-minus | peak | -0.185147 | -0.0246928 | -0.0257662 | 1.00247 | 7.18567 | 7.168 |
| 5 | C-core-minus | fixed_template | 0.488366 | 0.0613275 | 0.0552502 | 0.993916 | 8.83918 | 8.89329 |
| 5 | C-core-minus | aperture_2 | 234.753 | 0.304002 | 0.0773229 | 0.970747 | 3036.01 | 3127.49 |
| 5 | C-shoulder | peak | 0.661171 | 0.0881794 | 0.0920125 | 1.03336 | 7.18567 | unresolved |
| 5 | C-shoulder | fixed_template | -2.15753 | -0.270935 | -0.244087 | 0.958017 | 8.83918 | unresolved |
| 5 | C-shoulder | aperture_2 | -1669.58 | -2.16209 | -0.549926 | 0.869636 | 3036.01 | unresolved |
| 10 | C-core-plus | peak | 0.258217 | 0.0344013 | 0.0418444 | 1.00344 | 6.17089 | 6.14973 |
| 10 | C-core-plus | fixed_template | -0.521577 | -0.0649842 | -0.0661772 | 0.993503 | 7.88152 | 7.93306 |
| 10 | C-core-plus | aperture_2 | -263.762 | -0.333305 | -0.0965723 | 0.967136 | 2731.24 | 2824.05 |
| 10 | C-core-minus | peak | -0.0835619 | -0.0111326 | -0.0135413 | 1.00111 | 6.17089 | 6.16403 |
| 10 | C-core-minus | fixed_template | 0.0224101 | 0.00279211 | 0.00284337 | 0.999721 | 7.88152 | 7.88372 |
| 10 | C-core-minus | aperture_2 | 67.0937 | 0.0847836 | 0.0245653 | 0.99164 | 2731.24 | 2754.26 |
| 10 | C-shoulder | peak | 0.532313 | 0.0709181 | 0.0862621 | 1.02698 | 6.17089 | unresolved |
| 10 | C-shoulder | fixed_template | -2.73504 | -0.340763 | -0.347019 | 0.946838 | 7.88152 | unresolved |
| 10 | C-shoulder | aperture_2 | -2017.79 | -2.5498 | -0.738783 | 0.842846 | 2731.24 | unresolved |
| 15 | C-core-plus | peak | 0.295845 | 0.0391992 | 0.0542619 | 1.00394 | 5.45218 | 5.43078 |
| 15 | C-core-plus | fixed_template | -0.577795 | -0.0720741 | -0.106049 | 0.992802 | 5.44838 | 5.48788 |
| 15 | C-core-plus | aperture_2 | -290.113 | -0.368918 | -0.179752 | 0.963849 | 1613.96 | 1674.5 |
| 15 | C-core-minus | peak | 0.0462875 | 0.00613305 | 0.00848973 | 0.999384 | 5.45218 | 5.45554 |
| 15 | C-core-minus | fixed_template | 0.0564061 | 0.00703609 | 0.0103528 | 0.999297 | 5.44838 | 5.45221 |
| 15 | C-core-minus | aperture_2 | 36.3742 | 0.0462549 | 0.0225373 | 0.995467 | 1613.96 | 1621.31 |
| 15 | C-shoulder | peak | 0.795065 | 0.105345 | 0.145825 | 1.0407 | 5.45218 | unresolved |
| 15 | C-shoulder | fixed_template | -3.17788 | -0.396408 | -0.58327 | 0.938194 | 5.44838 | unresolved |
| 15 | C-shoulder | aperture_2 | -2356.08 | -2.99608 | -1.45981 | 0.816247 | 1613.96 | unresolved |

## Network populations

| Rank | Network | Stage | Segments | Retained entries | Realized ranks | Converged |
|---|---|---|---:|---:|---|---|
| 5 | 11 | M0 | 6 | 1446812 | [5] | True |
| 5 | 12 | M0 | 6 | 1424121 | [5] | True |
| 5 | 11 | last | 6 | 1446812 | [5] | True |
| 5 | 12 | last | 6 | 1424121 | [5] | True |
| 10 | 11 | M0 | 6 | 1446812 | [10] | True |
| 10 | 12 | M0 | 6 | 1424121 | [10] | True |
| 10 | 11 | last | 6 | 1446812 | [10] | True |
| 10 | 12 | last | 6 | 1424121 | [10] | True |
| 15 | 11 | M0 | 6 | 1446812 | [15] | True |
| 15 | 12 | M0 | 6 | 1424121 | [15] | True |
| 15 | 11 | last | 6 | 1446812 | [15] | True |
| 15 | 12 | last | 6 | 1424121 | [15] | True |

Native per-segment objectives, terminations, detector identities, eligibility and iterations are retained in network-PTC.json and M0/LAST receipts. The route does not publish before/after residual spectra or correlation summaries; no new spectral method was introduced. Lower variance alone is not success.

All-cycle raw apertures, fitted-background contributions, response morphology, admitted pixels and T9–T10 changes remain in compact JSON. Full-support comparisons and any lost/gained pixel locations are in support.json. See the owner assessment for the interpretation and single next recommendation.

Runtime/storage: {"wall_seconds": 596.0447505830089, "maximum_native_process_sampled_RSS": 1556201472, "maximum_stage_storage": 2744225457, "storage_ceiling": 8824001024, "branch_workers_maximum": 3, "numeric_library_threads": 1, "scope": "per-native-process sampled RSS; stage storage snapshots plus allocation reservations, not continuous system or filesystem high water"}

Local AppleClang/Homebrew evidence only; no Unity/Spack qualification.
