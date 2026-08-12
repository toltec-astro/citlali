# Real event-model adequacy failure 07

Date: 2026-08-12

## Trigger

The checksum-authenticated ObsNum 150818 Unity event pilot reproduced the
geometric crossing census and the local forward-model point estimates, but its
one-event-per-page review exposed a residual-diagnostic failure. Several real,
compact source passages are displaced from the single common source center and
are therefore fit very poorly by the common-centroid forward model. The two
separate scan-row 0 / UID 1051 passages are the clearest example; the source is
visible in both, but the common model misses it. The deliberately selected
worst-correlation event has correlation -0.556.

This is not dismissed as a renderer artifact. Of 842 events represented by
the retained detector-scan objective, 148 (17.6 percent) have local
data/model correlation below 0.5 and account for 29.3 percent of weighted
residual SSE. Thirteen have nonpositive profiled amplitude. Those low-
correlation events carry only 4.1 percent of the timing-leverage proxy, which
explains why the ensemble lag remains stable, but it does not make the raw-SSE
objective an acceptable corpus estimator.

The review also exposed an accounting defect: the stack title reported all
907 complete geometric events while its numerical arrays represented 842
events in the retained detector-scan groups, only 829 of which had positive
finite amplitude.

## Failed candidate rule

A preliminary +/-50-ms matched-filter quality gate was rejected before it was
frozen. A fixed spatial detector-center error maps to a speed-dependent time
offset; the rule would reject bright slow crossings such as UID 1051 while
retaining the same spatial displacement at high speed. That would manufacture
the speed dependence the study is intended to measure.

## Repair direction

The successor separates three facts:

1. geometric event identity remains tau-zero/PPT-centered and signal-blind;
2. compact-source morphology is measured with a symmetric spatial matched
   filter over the fixed event window, independent of global tau;
3. qualified event centroids are fit with a bounded robust regression in
   angular space, with equal total base weight per detector.

The matched-filter peak location is retained as data, never used silently as
a detector correction. The timing coefficient is obtained from its signed
speed dependence, while the two sign terms test azimuth and elevation
hysteresis. All geometric, qualified, rejected, and numerical counts remain
separate in outputs.

The first rendered replacement review exposed a second, narrower support
failure: two distinct passages from one detector-scan retained overlapping
`+/-1.5`-FWHM geometric windows, allowing both source peaks to enter either
local centroid. The centroid protocol partitions adjacent windows at their
integer-sample midpoint. This preserves the frozen crossing discovery and
event identities while making local event samples mutually exclusive.

This is an anchor-driven algorithm-development repair authorized by the
protocol's residual-diagnostic clause. It is not independent validation. No
non-anchor real observation may be inspected until synthetic recovery,
checksum, and complete anchor visual gates pass and the implementation is
refrozen.
