# Baseline before candidate selection

A completed3 fresh unmodified raw-to-T5 reductions and3 pre-RTC response branches,
plus1 counted RTC-only capture. No science replacement. First-representative all
M0–T5 arrays/measurements match the preserved warm T5 prefix exactly. Subsequent
references include exactly matching conditional T5 uncertainties. The nonzero
first comparator stop and sealed-byte recovery remain recorded separately.

Ready71.607–73.488s; analysis71.699–73.575s; exit71.924–73.763s. Network11 RTC spans
11.48s in first run; overlapping network intervals must not be summed as latency.
Detailed per-network intervals and process-CPU envelopes are in BASELINE_TIMING.
Process CPU counters span both worker threads; they are not exclusive worker CPU.
Stage traces have relative clocks, so the RTC wall fraction is reported using
parallel-stage occupancy bounds, not a falsely exact cross-worker union.

All486/491 detectors are counted;417/411 have calibration/support for the matched
spectral audit. 4s Hann windows, original timestamps, four-FWHM source exclusion,
matched complete support and physical duration are used. The11.006Hz feature
survives low-pass unchanged: all-stage median band powers raw2550.88/4319.02,
RTC2550.89/4319.04 mJy²/beam². M0 PTC reduces these to575.98/2537.70; T5 residual
567.14/2566.02. Source restoration is zero on these source-excluded windows and
therefore cannot explain the remaining line. Native residual coherence remains
high after RTC. This is consequential remaining narrow-line work, not a claim
about electronics vs optical origin or a calibrated map sensitivity loss.

RTC excludes69888/93840 transient pair cells. There are77/32 reviewed events,
including9/0 admitted level-shift groups. No donor replacements occur, so this
observation does not establish their efficacy or ringing performance. The307-tap
low-pass protects factor-two decimation; original validity, speed-center exclusion,
physical domains and full measured support are preserved. Full cause/source-crossing
counts and independent descriptive difference-outlier measurements are saved.

Pre-RTC compact +/- projection gain is0.996307 at RTC (spatial error norm0.005973),
shoulder0.999117 (error0.001957), all on own REF map support. RTC input r/validity
and realized state/center support are unchanged in all branches; raw-unit
calibration-only error <=1.43e-14mJy/beam. T5 compact peak increment gains are
0.991949/0.991965; shoulder peak1.045857, aperture2 gain0.839454 and spatial gain
0.841944. Preserve broad morphology/background terms rather than scalar correction.
Final REF conditional peak sigma6.164mJy/beam, position0.0405/0.0385arcsec,
aperture2 sigma2744.34mJy/beam arcsec². These are not paired-difference significance.
All endpoint values are finite. Independent non-BLAS contraction reproduces every
saved operator and uncertainty exactly in all six trajectories; the numerical-library
warnings are retained and this crosscheck is in UNCERTAINTY_WARNING_CHECK.json.

B selection: two byte-preserving serialization changes recorded in
FAMILY-B-SELECTION. Hypothesized repeated transient validation is negligible and
will not be changed. Native numerical methods, filter support, reassessment and
all required diagnostic outputs remain identical. C is not selected yet: the
previous1Hz/3s notch numerical design exists, but its original selection did not
automatically authorize every detector/cadence. Resolve that policy scope before
any C execution; no parameter search or production admission is authorized.
