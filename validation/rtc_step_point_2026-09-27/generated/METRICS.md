# Saved POINT step metrics

| Observation | nw | events / admitted | affected D / all D | valid detector-s | RTC retained detector-s | CAL/MAP contributions |
|---|---:|---:|---:|---:|---:|---:|
| 152430 | 11 | 77 / 9 | 7 / 486 | 30689.354 | 26551.312 | 1446812 |
| 152430 | 12 | 32 / 0 | 0 / 491 | 30685.430 | 27136.934 | 1424121 |
| 152420 | 11 | 130 / 64 | 26 / 486 | 30175.314 | 25646.899 | 1414012 |
| 152420 | 12 | 118 / 25 | 25 / 491 | 30179.238 | 26680.934 | 1400208 |

Ratios compare signed increments to own ideal on full REF support; position/width discrepancies are absolute arcsec. T5 sigmas are conditional scales, not paired-difference significance.

| branch | stage | peak | template | ap1 | ap1.5 | ap2 | spatial g | distortion eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C-core-plus | T0 | 0.62377 | 0.50827 | 0.42320 | 0.31687 | 0.28402 | 0.49587 | 0.55220 |
| C-core-plus | T5 | 0.99006 | 0.99349 | 0.99860 | 0.99251 | 0.99292 | 0.99293 | 0.03156 |
| C-core-minus | T0 | 0.62542 | 0.51232 | 0.42585 | 0.30608 | 0.26641 | 0.49889 | 0.55556 |
| C-core-minus | T5 | 0.98943 | 0.99384 | 0.99567 | 0.99643 | 1.00233 | 0.99037 | 0.02617 |
| C-shoulder | T0 | 0.39308 | 0.38373 | 0.40063 | 0.39382 | 0.37287 | 0.38993 | 0.64225 |
| C-shoulder | T5 | 1.00951 | 0.96242 | 0.97805 | 0.90429 | 0.86241 | 0.87338 | 0.20486 |

Window metrics use all finite samples except ±12 native rows around the declared center, independent pre/post medians and adjacent finite-difference MAD. Descriptive contrast only: slopes, source structure and colored noise are not removed. No per-step significance threshold or new fit. Raw x/r and RTC use Δf/f; CAL/PTC use mJy/beam. Excluded stage values are unavailable, never zero.
