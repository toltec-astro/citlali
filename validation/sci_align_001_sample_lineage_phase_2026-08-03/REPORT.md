# SCI-ALIGN-001 sample-lineage and sample-phase diagnostic

## Disposition

The frozen Beammap 148670 experiment supports a common approximately
1.5-sample **effective** assigned-coordinate displacement, but it does not
support the proposed decomposition as a Citlali off-by-one plus an independent
half-sample convention defect.

Stage A found no row permutation. The actual raw inputs have exact shared
`Data.Toltec.Is`/`Qs`/`Ts` row cardinality, zero packet gaps, zero slot
collisions or reversals, and zero admitted science slots lacking an original
native detector row. The inspected implementation carries the same native row
through timestamp reconstruction, positive-add offset realization, nearest
slot insertion, scan slicing, fixed-row RTC/PTC processing, and retained
detector-TOD sample copy. Therefore `k=+/-1` is a counterfactual association,
not a demonstrated correction.

On the frozen 4809-detector common-interior cohort, the assigned-slot baseline
is **-12.2495 +/- 0.2748 ms**. The `k=+1, phi=+0.5` model evaluates the
telescope 12.288 ms later and gives **-0.0752 +/- 0.2720 ms**, consistent with
zero in the pooled channel. A secondary continuous profile has slope 0.9883
and crosses zero 0.0791 ms beyond that discrete model, corresponding to a
fixture-specific effective displacement of about **12.367 +/- 0.275 ms**.
This is not a physical timestamp correction.

Material structure remains after the pooled null:

- assigned-basis network estimates span **7.351 ms**, from -4.457 to +2.893
  ms, and retain correlation **-0.963** and slope **-0.877** against the
  exactly defined raw-minus-lattice-slot residual;
- array estimates are +1.032 ms (`a1100`), -0.273 ms (`a1400`), and -3.119 ms
  (`a2000`);
- first- and second-half estimates are +0.662 and -1.194 ms, a **1.856 ms**
  difference;
- the raw-timestamp `k=+1, phi=+0.5` model reduces the interface correlation
  to +0.443 with slope +0.122, but leaves **-0.9541 +/- 0.2659 ms**, rather
  than a null.

The preregistered outcome is therefore:

> **common component explained but interface/time-dependent residual remains**

## Frozen discrete fingerprint

Primary results use identical common interior support for every model.

| Time basis | Counterfactual | Timing residual (ms) |
| --- | --- | ---: |
| assigned slot | `k=0, phi=0` | -12.2495 |
| assigned slot | `k=+1, phi=0` | -4.1201 |
| assigned slot | `k=0, phi=+0.5` | -8.2096 |
| assigned slot | `k=+1, phi=+0.5` | -0.0752 |
| raw detector timestamp | `k=0, phi=0` | -13.1459 |
| raw detector timestamp | `k=+1, phi=+0.5` | -0.9541 |

The near-duplicate results for equal total assigned-slot shifts demonstrate
why the numerical grid alone cannot distinguish row reassociation from a
constant phase change. Direct lineage evidence is decisive, and it found no
off-by-one.

The previous LR-BEAMMAP estimate was -12.1380 ms. The -12.2495 ms primary
baseline here uses the methodological tightening requested for this task:
the frozen 4809-detector matched cohort and common interior support. The
0.1115 ms change is 0.41 of the new jackknife standard error.

## Sky and beam controls

For assigned `k=+1, phi=+0.5`, the right-minus-left centroid is -0.00735
arcsec parallel and +0.03171 arcsec perpendicular to the frozen scan axis.
Relative to the common-support baseline, major/minor FWHM and amplitudes remain
stable:

- baseline left/right major FWHM: 6.3873 / 6.3790 arcsec;
- combined-model left/right major FWHM: 6.3895 / 6.3970 arcsec;
- baseline left/right minor FWHM: 5.9068 / 5.9064 arcsec;
- combined-model left/right minor FWHM: 5.9081 / 5.9102 arcsec;
- baseline left/right amplitude: 1.07231 / 1.07340;
- combined-model left/right amplitude: 1.07196 / 1.07145.

Balanced same-direction parallel nulls are +0.0811 arcsec for left-going and
+0.0505 arcsec for right-going data. They remain much smaller than the
uncorrected 1.196 arcsec direction-odd displacement, but are not zero.

## Support accounting

The common-support primary contains 3,350,736 left-going and 3,349,589
right-going fit-selected detector samples. Before signal/flag/radial cuts, the
`k=+1` native population has 48,090 additional boundary samples across both
directions: one row per 4809 detector by 10 retained detector-TOD slots. All
of those rows fail the already frozen source-radius/validity selection, so the
common- and native-support scientific fit inputs and results are exactly equal
for this retained fixture. The boundary rows remain reported rather than
silently discarded.

## Engineering and physical authority

The engineering slot residual is frozen as

`r[i,n] = (t_raw[i,n] + offset[i]) - (phase + round_half_up(((t_raw[i,n] + offset[i]) - phase)/dt) * dt)`

with `round_half_up(x) = floor(x + 0.5)`, positive-add offsets applied exactly
once before slot assignment, and units of seconds. It is bounded by assignment
geometry and is not a sky-placement tolerance.

`Data.Toltec.Ts` producer semantics remain unavailable: this work does not
establish whether the timestamp marks integration start, end, or effective
centroid. Absolute sky-placement correctness and any physical correction
therefore remain unresolved.

## Scope

No Citlali application or configuration source changed. No Citlali reduction
was run, Unity was not contacted, and SCI-MAP-001-UNITY-001 was not changed.
All results come from digest-bound retained Beammap 148670 inputs and the
frozen LR-BEAMMAP registry/cohort. This package does not accept SCI-ALIGN,
authorize a production correction, or replace later human-run exact-SHA
evidence.
