# POINT 152430 a2000: replacement comparison

New-run measurements only. High coverage alone is informational. All uncertainty scales are model-based conditional propagation, not an empirically calibrated false-admission probability or paired-difference significance. Apertures are brightness × angular area, not integrated flux.

## Completion

| Policy | Branch | Status | Last cycle | Feedback attempts | Reason |
|---|---|---|---:|---:|---|
| BASE-POINT | REF | complete | 10 | 10 |  |
| BASE-POINT | C-core-plus | complete | 10 | 10 |  |
| BASE-POINT | C-core-minus | complete | 10 | 10 |  |
| BASE-POINT | C-shoulder | complete | 10 | 10 |  |
| BASE-POINT | D-seed-shoulder | complete | 10 | 10 |  |
| CORR | REF | complete | 10 | 10 |  |
| CORR | C-core-plus | complete | 10 | 10 |  |
| CORR | C-core-minus | complete | 10 | 10 |  |
| CORR | C-shoulder | complete | 10 | 10 |  |
| CORR | D-seed-shoulder | complete | 10 | 10 |  |
| DIAG | REF | complete | 10 | 10 |  |
| DIAG | C-core-plus | complete | 10 | 10 |  |
| DIAG | C-core-minus | complete | 10 | 10 |  |
| DIAG | C-shoulder | complete | 10 | 10 |  |
| DIAG | D-seed-shoulder | complete | 10 | 10 |  |

## Leading T10 measurement differences

Source responses are compared to their own policy REF. Actual-minus-ideal isolates response mismatch against the single saved ideal map-addition control; seed-minus-REF is initialization sensitivity.

| Policy | Branch | Comparison | Measurement | Signed effect | % whole REF | Conditional sigma | Effect / sigma |
|---|---|---|---|---:|---:|---:|---:|
| BASE-POINT | C-core-plus | actual_minus_ideal | peak | 0.2582173 | 0.03440129 | 6.170887 | 0.04184444 |
| BASE-POINT | C-core-plus | actual_minus_ideal | fixed_template | -0.521577 | -0.06498425 | 7.881515 | -0.06617725 |
| BASE-POINT | C-core-plus | actual_minus_ideal | aperture_2 | -263.762 | -0.3333055 | 2731.237 | -0.09657234 |
| BASE-POINT | C-core-minus | actual_minus_ideal | peak | -0.08356188 | -0.01113262 | 6.170887 | -0.01354131 |
| BASE-POINT | C-core-minus | actual_minus_ideal | fixed_template | 0.02241006 | 0.002792111 | 7.881515 | 0.00284337 |
| BASE-POINT | C-core-minus | actual_minus_ideal | aperture_2 | 67.09368 | 0.0847836 | 2731.237 | 0.02456531 |
| BASE-POINT | C-shoulder | actual_minus_ideal | peak | 0.5323135 | 0.07091805 | 6.170887 | 0.08626206 |
| BASE-POINT | C-shoulder | actual_minus_ideal | fixed_template | -2.735035 | -0.3407632 | 7.881515 | -0.347019 |
| BASE-POINT | C-shoulder | actual_minus_ideal | aperture_2 | -2017.792 | -2.549803 | 2731.237 | -0.7387831 |
| BASE-POINT | D-seed-shoulder | branch_minus_own_REF | peak | 0.0002380117 | 3.170937e-05 | 6.170887 | 3.857009e-05 |
| BASE-POINT | D-seed-shoulder | branch_minus_own_REF | fixed_template | 0.0003698363 | 4.60786e-05 | 7.881515 | 4.692452e-05 |
| BASE-POINT | D-seed-shoulder | branch_minus_own_REF | aperture_2 | 0.06150037 | 7.771557e-05 | 2731.237 | 2.25174e-05 |
| CORR | C-core-plus | actual_minus_ideal | peak | 0.2582173 | 0.03440129 | 6.170887 | 0.04184444 |
| CORR | C-core-plus | actual_minus_ideal | fixed_template | -0.521577 | -0.06498425 | 7.881515 | -0.06617725 |
| CORR | C-core-plus | actual_minus_ideal | aperture_2 | -263.762 | -0.3333055 | 2731.237 | -0.09657234 |
| CORR | C-core-minus | actual_minus_ideal | peak | -0.08356188 | -0.01113262 | 6.170887 | -0.01354131 |
| CORR | C-core-minus | actual_minus_ideal | fixed_template | 0.02241006 | 0.002792111 | 7.881515 | 0.00284337 |
| CORR | C-core-minus | actual_minus_ideal | aperture_2 | 67.09368 | 0.0847836 | 2731.237 | 0.02456531 |
| CORR | C-shoulder | actual_minus_ideal | peak | 0.5323135 | 0.07091805 | 6.170887 | 0.08626206 |
| CORR | C-shoulder | actual_minus_ideal | fixed_template | -2.735035 | -0.3407632 | 7.881515 | -0.347019 |
| CORR | C-shoulder | actual_minus_ideal | aperture_2 | -2017.792 | -2.549803 | 2731.237 | -0.7387831 |
| CORR | D-seed-shoulder | branch_minus_own_REF | peak | 0.0002380117 | 3.170937e-05 | 6.170887 | 3.857009e-05 |
| CORR | D-seed-shoulder | branch_minus_own_REF | fixed_template | 0.0003698363 | 4.60786e-05 | 7.881515 | 4.692452e-05 |
| CORR | D-seed-shoulder | branch_minus_own_REF | aperture_2 | 0.06150037 | 7.771557e-05 | 2731.237 | 2.25174e-05 |
| DIAG | C-core-plus | actual_minus_ideal | peak | 0.1532004 | 0.02041604 | 6.166691 | 0.02484322 |
| DIAG | C-core-plus | actual_minus_ideal | fixed_template | -0.3097763 | -0.03857789 | 7.880227 | -0.03931057 |
| DIAG | C-core-plus | actual_minus_ideal | aperture_2 | -138.8505 | -0.1750109 | 2730.665 | -0.0508486 |
| DIAG | C-core-minus | actual_minus_ideal | peak | 0.04385838 | 0.005844725 | 6.166691 | 0.00711214 |
| DIAG | C-core-minus | actual_minus_ideal | fixed_template | -0.2921271 | -0.03637996 | 7.880227 | -0.0370709 |
| DIAG | C-core-minus | actual_minus_ideal | aperture_2 | -87.94571 | -0.1108491 | 2730.665 | -0.0322067 |
| DIAG | C-shoulder | actual_minus_ideal | peak | 0.511769 | 0.06820019 | 6.166691 | 0.08298924 |
| DIAG | C-shoulder | actual_minus_ideal | fixed_template | -2.39113 | -0.2977786 | 7.880227 | -0.3034341 |
| DIAG | C-shoulder | actual_minus_ideal | aperture_2 | -1842.174 | -2.321926 | 2730.665 | -0.6746248 |
| DIAG | D-seed-shoulder | branch_minus_own_REF | peak | -0.07534611 | -0.01004089 | 6.166691 | -0.01221824 |
| DIAG | D-seed-shoulder | branch_minus_own_REF | fixed_template | 0.04444268 | 0.005534655 | 7.880227 | 0.005639771 |
| DIAG | D-seed-shoulder | branch_minus_own_REF | aperture_2 | 36.10085 | 0.04550248 | 2730.665 | 0.01322053 |

## Measurement definitions

POINT-EXPECTED-v1: seven-parameter Gaussian plus constant; W=I; one bounded fit per full map. Peak is amplitude above the fitted constant in mJy/nominal beam. Centroid is east/north arcsec. Major/minor FWHM are sorted principal widths; ellipse covariance components avoid angle gauge ambiguity. Fixed template uses the original M0 template and subtracts each map’s fitted constant. Apertures use fixed M0 centroid and radii 1, 1.5, 2 × M0 major FWHM with 4 arcsec² pixels. Raw aperture and fitted-background contributions are separately retained in histories. Endpoint sensitivity operators include refitted constant nuisance and spatial covariance. Angle/width errors are not substituted for sorted or tensor-shape uncertainties. Unsupported uncertainty scales remain unresolved.

M0–T10 compact history is in histories.json; the stage summary table below selects M0/T1/T3/T5/T10. g and eta use the fixed own-REF ordinary-MAP injection controls and unweighted common support. Controls bypass PTC; this experiment tests post-RTC/CAL response and does not establish astronomical flux calibration.

## Selected histories

| Policy | Branch | Cycle | Peak | Major/minor FWHM | Template | Aperture 2 | Selected / added |
|---|---|---:|---:|---|---:|---:|---|
| BASE-POINT | REF | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| BASE-POINT | REF | 1 | 701.182 | 9.899839/9.027931 | 731.0034 | 67679.73 | 65/0 |
| BASE-POINT | REF | 3 | 746.3894 | 10.11064/9.275377 | 795.8068 | 77944.65 | 69/0 |
| BASE-POINT | REF | 5 | 750.193 | 10.13521/9.306276 | 801.9894 | 79029.93 | 69/0 |
| BASE-POINT | REF | 10 | 750.6036 | 10.13761/9.308916 | 802.6206 | 79135.2 | 69/0 |
| BASE-POINT | C-core-plus | 0 | 614.7843 | 9.583672/8.500111 | 612.6599 | 47862.94 | 57/0 |
| BASE-POINT | C-core-plus | 1 | 769.4602 | 9.900066/9.021854 | 801.9438 | 74219.56 | 65/0 |
| BASE-POINT | C-core-plus | 3 | 821.064 | 10.10264/9.272551 | 874.9889 | 85681.92 | 69/0 |
| BASE-POINT | C-core-plus | 5 | 825.4901 | 10.12531/9.297456 | 881.7077 | 86786.72 | 69/0 |
| BASE-POINT | C-core-plus | 10 | 825.9222 | 10.12771/9.299961 | 882.375 | 86897.42 | 69/0 |
| BASE-POINT | C-core-minus | 0 | 511.7171 | 9.620975/8.593938 | 513.7547 | 41035.38 | 52/0 |
| BASE-POINT | C-core-minus | 1 | 632.0118 | 9.91457/9.037957 | 659.7003 | 61186.25 | 65/0 |
| BASE-POINT | C-core-minus | 3 | 671.6407 | 10.11359/9.283811 | 716.4862 | 70189.1 | 66/0 |
| BASE-POINT | C-core-minus | 5 | 675.1038 | 10.13676/9.308682 | 721.8324 | 71089.74 | 66/0 |
| BASE-POINT | C-core-minus | 10 | 675.4597 | 10.13891/9.311036 | 722.367 | 71176.31 | 66/0 |
| BASE-POINT | C-shoulder | 0 | 574.4926 | 9.874312/8.769175 | 589.8722 | 50538.51 | 59/0 |
| BASE-POINT | C-shoulder | 1 | 718.867 | 10.22497/9.307547 | 772.0002 | 76616.64 | 71/0 |
| BASE-POINT | C-shoulder | 3 | 766.5365 | 10.47886/9.577994 | 843.3813 | 88435.57 | 77/0 |
| BASE-POINT | C-shoulder | 5 | 770.4246 | 10.52119/9.612212 | 850.5605 | 89817.11 | 77/0 |
| BASE-POINT | C-shoulder | 10 | 770.8656 | 10.52533/9.61561 | 851.3329 | 89957.02 | 77/0 |
| BASE-POINT | D-seed-shoulder | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| BASE-POINT | D-seed-shoulder | 1 | 707.3214 | 10.07048/9.177733 | 749.2784 | 73053.64 | 66/0 |
| BASE-POINT | D-seed-shoulder | 3 | 747.3087 | 10.12051/9.283068 | 797.4614 | 78238.23 | 69/0 |
| BASE-POINT | D-seed-shoulder | 5 | 750.2941 | 10.13588/9.306926 | 802.1481 | 79056.47 | 69/0 |
| BASE-POINT | D-seed-shoulder | 10 | 750.6038 | 10.13761/9.308917 | 802.6209 | 79135.26 | 69/0 |
| CORR | REF | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| CORR | REF | 1 | 701.182 | 9.899839/9.027931 | 731.0034 | 67679.73 | 65/0 |
| CORR | REF | 3 | 746.3894 | 10.11064/9.275377 | 795.8068 | 77944.65 | 69/0 |
| CORR | REF | 5 | 750.193 | 10.13521/9.306276 | 801.9894 | 79029.93 | 69/0 |
| CORR | REF | 10 | 750.6036 | 10.13761/9.308916 | 802.6206 | 79135.2 | 69/0 |
| CORR | C-core-plus | 0 | 614.7843 | 9.583672/8.500111 | 612.6599 | 47862.94 | 57/0 |
| CORR | C-core-plus | 1 | 769.4602 | 9.900066/9.021854 | 801.9438 | 74219.56 | 65/0 |
| CORR | C-core-plus | 3 | 821.064 | 10.10264/9.272551 | 874.9889 | 85681.92 | 69/0 |
| CORR | C-core-plus | 5 | 825.4901 | 10.12531/9.297456 | 881.7077 | 86786.72 | 69/0 |
| CORR | C-core-plus | 10 | 825.9222 | 10.12771/9.299961 | 882.375 | 86897.42 | 69/0 |
| CORR | C-core-minus | 0 | 511.7171 | 9.620975/8.593938 | 513.7547 | 41035.38 | 52/0 |
| CORR | C-core-minus | 1 | 632.0118 | 9.91457/9.037957 | 659.7003 | 61186.25 | 65/0 |
| CORR | C-core-minus | 3 | 671.6407 | 10.11359/9.283811 | 716.4862 | 70189.1 | 66/0 |
| CORR | C-core-minus | 5 | 675.1038 | 10.13676/9.308682 | 721.8324 | 71089.74 | 66/0 |
| CORR | C-core-minus | 10 | 675.4597 | 10.13891/9.311036 | 722.367 | 71176.31 | 66/0 |
| CORR | C-shoulder | 0 | 574.4926 | 9.874312/8.769175 | 589.8722 | 50538.51 | 59/0 |
| CORR | C-shoulder | 1 | 718.867 | 10.22497/9.307547 | 772.0002 | 76616.64 | 71/0 |
| CORR | C-shoulder | 3 | 766.5365 | 10.47886/9.577994 | 843.3813 | 88435.57 | 77/0 |
| CORR | C-shoulder | 5 | 770.4246 | 10.52119/9.612212 | 850.5605 | 89817.11 | 77/0 |
| CORR | C-shoulder | 10 | 770.8656 | 10.52533/9.61561 | 851.3329 | 89957.02 | 77/0 |
| CORR | D-seed-shoulder | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| CORR | D-seed-shoulder | 1 | 707.3214 | 10.07048/9.177733 | 749.2784 | 73053.64 | 66/0 |
| CORR | D-seed-shoulder | 3 | 747.3087 | 10.12051/9.283068 | 797.4614 | 78238.23 | 69/0 |
| CORR | D-seed-shoulder | 5 | 750.2941 | 10.13588/9.306926 | 802.1481 | 79056.47 | 69/0 |
| CORR | D-seed-shoulder | 10 | 750.6038 | 10.13761/9.308917 | 802.6209 | 79135.26 | 69/0 |
| DIAG | REF | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| DIAG | REF | 1 | 701.182 | 9.899839/9.027931 | 731.0034 | 67679.73 | 65/0 |
| DIAG | REF | 3 | 746.3894 | 10.11064/9.275377 | 795.8068 | 77944.65 | 72/3 |
| DIAG | REF | 5 | 750.0707 | 10.13836/9.311009 | 802.1561 | 79143.37 | 74/5 |
| DIAG | REF | 10 | 750.3924 | 10.14499/9.317368 | 802.9891 | 79338.2 | 76/7 |
| DIAG | C-core-plus | 0 | 614.7843 | 9.583672/8.500111 | 612.6599 | 47862.94 | 57/0 |
| DIAG | C-core-plus | 1 | 769.4602 | 9.900066/9.021854 | 801.9438 | 74219.56 | 65/0 |
| DIAG | C-core-plus | 3 | 820.9826 | 10.10433/9.273901 | 875.0279 | 85724.65 | 75/6 |
| DIAG | C-core-plus | 5 | 825.2359 | 10.13226/9.305005 | 882.0362 | 86989.19 | 75/5 |
| DIAG | C-core-plus | 10 | 825.6029 | 10.13645/9.313392 | 882.9554 | 87225.42 | 77/7 |
| DIAG | C-core-minus | 0 | 511.7171 | 9.620975/8.593938 | 513.7547 | 41035.38 | 52/0 |
| DIAG | C-core-minus | 1 | 632.0118 | 9.91457/9.037957 | 659.7003 | 61186.25 | 65/0 |
| DIAG | C-core-minus | 3 | 671.6407 | 10.11359/9.283811 | 716.4862 | 70189.1 | 66/0 |
| DIAG | C-core-minus | 5 | 675.0557 | 10.1379/9.309841 | 721.8587 | 71114.91 | 69/3 |
| DIAG | C-core-minus | 10 | 675.3796 | 10.14124/9.312841 | 722.4209 | 71224.18 | 69/3 |
| DIAG | C-shoulder | 0 | 574.4926 | 9.874312/8.769175 | 589.8722 | 50538.51 | 59/0 |
| DIAG | C-shoulder | 1 | 718.867 | 10.22497/9.307547 | 772.0002 | 76616.64 | 72/1 |
| DIAG | C-shoulder | 3 | 766.2483 | 10.48816/9.584611 | 843.6635 | 88626.79 | 84/7 |
| DIAG | C-shoulder | 5 | 770.1603 | 10.52907/9.625406 | 851.0735 | 90130.9 | 90/13 |
| DIAG | C-shoulder | 10 | 770.6108 | 10.53367/9.633316 | 852.0503 | 90338.47 | 90/13 |
| DIAG | D-seed-shoulder | 0 | 564.3027 | 9.591217/8.552165 | 564.3027 | 44497.89 | 54/0 |
| DIAG | D-seed-shoulder | 1 | 707.3214 | 10.07048/9.177733 | 749.2784 | 73053.64 | 66/0 |
| DIAG | D-seed-shoulder | 3 | 747.2567 | 10.12163/9.284167 | 797.4893 | 78264.79 | 73/4 |
| DIAG | D-seed-shoulder | 5 | 750.1784 | 10.13934/9.312492 | 802.3648 | 79187.32 | 74/5 |
| DIAG | D-seed-shoulder | 10 | 750.3171 | 10.14571/9.319903 | 803.0336 | 79374.31 | 74/4 |

## Execution and resources

Counts: `{"trajectories_started": 15, "trajectories_complete": 15, "feedback_attempts": 150, "initial_physical_inputs": 4, "ideal_controls": 9, "endpoint_attempts": 3, "native_invocations": 155, "input_transfers": 0, "input_transfer_bytes": 0}`.

Resources: `{"native_wall_seconds_sum": 1499.3508748342574, "upstream_seconds": 31.617487292009173, "phase_wall_seconds": {"PHASE-references": 86.70437337498879, "PHASE-shapes": 26.244842707994394, "PHASE-responses": 160.8016417500039, "PHASE-base": 85.71239487500861}, "phase_peak_family_rss_bytes": {"PHASE-references": 1621819392, "PHASE-shapes": null, "PHASE-responses": 7815839744, "PHASE-base": 888668160}, "native_max_sampled_rss_bytes": 1556480000, "largest_observed_stage_bytes": 2901191288, "ceiling_bytes": 16159001984, "disk_measurement": "stage snapshots plus enforced in-flight allocation reservations; not a continuous filesystem high-water measurement", "numeric_threads": 1, "phase_workers": {"PHASE-references": 2, "PHASE-shapes": null, "PHASE-responses": 12, "PHASE-base": 1}}`.

The native processing guards, subtract/restore checks, control audits, shape definitions, exact inputs and configuration remain in compact products. Numeric libraries use one thread; outer workers are in resources.json. No threshold/covariance retuning, extra cycles, other observations/arrays, or JINC.
