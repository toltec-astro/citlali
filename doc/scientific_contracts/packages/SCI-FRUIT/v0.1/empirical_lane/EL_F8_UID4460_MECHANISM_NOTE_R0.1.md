# SCI-FRUIT EL-F8 — Why one detector changed the map r0.1

Status: **read-only implementation and development-evidence interpretation;
not scientific authority or a detector-quality judgment**

## Plain answer

UID 4460 did not make the large arcs simply by contributing too much signal
to the final map. The existing learning machinery removed the detector from
scan 6 *before* RTC and PTC cleaning in iteration 5. That changed the shared
processing applied to the other a1400 detectors. The resulting map difference
therefore spreads along several scan paths.

In other words, the detector really was flagged. The large imprint is mainly
the consequence of **where that flag was applied**, not evidence that one
unflagged detector directly painted every arc.

## What the detector looked like before the hard exclusion

The frozen observation-123424 APT and iteration-4 checkpoints show:

| Check | UID 4460 result |
|---|---|
| APT identity | a1400, network 9 |
| APT decision | accepted |
| APT `flag`, `flag2`, and `kids_flag` | all zero |
| APT beam fit | 7.44 by 7.18 arcsec |
| APT signal-to-noise | 62.29 |
| accumulated detector-weight factor, control | 0.75819 |
| accumulated detector-weight factor, injected | 0.75867 |
| position among 445 validated, unflagged a1400 detectors | 36th percentile from the low-weight end |

Thus ordinary weight validation already reduced this detector by about 24
percent, but did not identify it as an extreme weight outlier. The injected
and control factors are nearly identical. The APT also records it as a normal,
usable detector. Four localized RTC despike intervals involving this UID were
learned in earlier scans, so the detector is not asserted to be spotless.

## How the hard record was learned

The configured map diagnostic selected the eight most extreme off-source map
pixels after requiring map-level robust significance above 8. It then assigned
each selected pixel to a leading detector/scan contributor. A factor-zero
record was created when the same detector and scan appeared at least four
times.

At iteration 4:

- UID 4460 appeared in three qualifying pixels in the control and four in the
  injected branch;
- four was exactly the hard-exclusion threshold;
- the four selected map values were about 112--136 mJy/beam; and
- their detector-specific leave-one-out significances were only 1.70--2.12,
  with a maximum of 2.123.

The present rule does not require the leave-one-out significance itself to
exceed a detector-specific threshold. A detector may therefore be the leading
contributor to four globally unusual pixels without being demonstrably bad on
its own. The one-pixel control/injected difference then changes the action
from no hard record to complete scan-local exclusion.

## What happened in the next iteration

At the beginning of iteration 5, the carried factor-zero record:

1. matched UID 4460 in scan 6;
2. newly flagged all 676 of its raw samples before RTC;
3. entered PTC with 305 corresponding processed samples already flagged; and
4. therefore changed detector participation before shared atmospheric and
   other cleaning operations.

EL-F6 shows the causal result of that action. Applying the record changes the
a1400 map by 2.4370 mJy/beam RMS around real Neptune and 2.2739 mJy/beam RMS in
the registered annulus, but only 0.2581 mJy/beam RMS around the injected
source. EL-F7 independently retains the same separation. Those measurements
do not say that UID 4460 directly contains 2.4 mJy/beam of false sky. They say
that removing it before a shared, data-dependent operator changes the output
by that amount.

## Why the next test should change placement, not strength

A fractional weight would mix two questions: whether the trigger is credible
and what numerical weight is best. The cleaner first question is whether the
large amplification comes from applying the hard decision before shared
cleaning.

The proposed EL-F8 test therefore compares:

- the current behavior, which excludes the detector before RTC/PTC; and
- a development-only placement in which the detector remains in shared
  cleaning but is excluded immediately before final map accumulation.

Both placements honor the same learned record. The second prevents a
map-domain diagnostic from changing the earlier shared cleaning operator while
still withholding the detector's direct final-map contribution. This is a
mechanism test, not yet a recommendation for production.

## Evidence status

This explanation is supported by the checked-out implementation and frozen
EL-F5--F7 development evidence. Those are non-authoritative Stage A sources.
The observation does not establish whether UID 4460 is scientifically good or
bad, whether every detector exclusion behaves this way, or whether
mapmaking-only exclusion is the correct general policy.

The implementation paths inspected at source parent `831bdf69b` were
`mapdiag_workspace_outlier_collect.h`,
`mapdiag_workspace_learning_emit.h`,
`learning_detector_exclusion_apply_impl.h`, `pointing_run_impl.h`, and
`pointing_map_population_impl.h`. The exact external evidence identities were:

- EL-F5 injected log: 2,727,885 bytes, SHA-256
  `f2452be6338b814c2e386503898ce7b2163b877502980946745277de425a2f67`;
- EL-F5 injected iteration-4 learning table: 881,918 bytes, SHA-256
  `4ea7986dbea4ea619438092dab0f7c5221b520ee6901529b52787d3fa7b24616`;
- matched observation-123424 APT: 6,207,815 bytes, SHA-256
  `16389e5e58b76d39ef7fcedd3888c662e96db592bc3a8561530379e51c435626`;
- injected iteration-4 checkpoint: SHA-256
  `2d600fde6b642ea053bc49d357bed16c800bb1dd689c0ee5ae084e115970fb7c`;
  and
- control iteration-4 checkpoint: SHA-256
  `a77505ab0637c1f257016ee0d9e801b3bba17ed52ab88d52f417a5c1513b451f`.
