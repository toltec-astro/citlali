# Saved POINT step metrics

| Observation | nw | events / admitted | affected D / all D | valid detector-s | RTC retained detector-s | CAL/MAP contributions |
|---|---:|---:|---:|---:|---:|---:|
| 152430 | 1 | 504 / 51 | 48 / 494 | 30749.491 | 24185.430 | 2526957 |
| 152430 | 2 | 344 / 25 | 24 / 510 | 31970.746 | 26340.934 | 2846710 |
| 152430 | 3 | 1083 / 72 | 66 / 533 | 33700.626 | 16757.768 | 1733290 |
| 152430 | 4 | 597 / 11 | 11 / 532 | 32995.860 | 26801.521 | 2699042 |

Ratios compare signed increments to own ideal on full REF support; position/width discrepancies are absolute arcsec. T5 sigmas are conditional scales, not paired-difference significance.

| branch | stage | peak | template | ap1 | ap1.5 | ap2 | spatial g | distortion eta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C-core-plus | T0 | 0.85047 | 0.75947 | 0.68973 | 0.60380 | 0.54347 | 0.75311 | 0.29734 |
| C-core-plus | T5 | 0.99963 | 0.99190 | 0.98488 | 0.97160 | 0.96174 | 0.99090 | 0.08299 |
| C-core-minus | T0 | 0.85285 | 0.75822 | 0.68647 | 0.59855 | 0.54048 | 0.75223 | 0.28647 |
| C-core-minus | T5 | 1.00037 | 0.99016 | 0.98089 | 0.96589 | 0.95906 | 0.98951 | 0.02149 |
| C-shoulder | T0 | 0.79478 | 0.67697 | 0.65963 | 0.62082 | 0.57349 | 0.63232 | 0.41349 |
| C-shoulder | T5 | 1.06742 | 0.93130 | 0.93319 | 0.85625 | 0.81653 | 0.84860 | 0.24002 |

Window metrics use all finite samples except ±12 native rows around the declared center, independent pre/post medians and adjacent finite-difference MAD. Descriptive contrast only: slopes, source structure and colored noise are not removed. No per-step significance threshold or new fit. Raw x/r and RTC use Δf/f; CAL/PTC use mJy/beam. Excluded stage values are unavailable, never zero.
