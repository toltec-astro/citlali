# Real-anchor numerical gate failure 02

The revision-2 implementation rescaled the optimizer coordinate for tau from
seconds to milliseconds, passed all eight synthetic gates, and was checksum
frozen before ObsNum 150818 was re-examined. The anchor coordinate gate again
passed exactly, but scalar-lag and joint fits again returned `tau=0` exactly.

The remaining failure is the finite-difference step, not the physical fitted
value. SciPy's default absolute step of `1e-8` in the optimizer coordinate
became `1e-8` ms, or `1e-11` s. That perturbation is below stable resolution
of the real profiled objective. The already recorded coarse objective profile
demonstrates that the objective itself is sensitive to tau on millisecond
scales.

Revision 3 therefore declares unit-aware absolute finite-difference steps:
0.01 ms for tau, 0.0001 arcsec for position, hysteresis, and beam-width
coordinates, and 0.00001 rad for beam angle. The 0.01-ms tau step is 1/819.2
of the 8.192-ms detector cadence and is small relative to every scientific
interval of interest while remaining numerically resolvable. No input,
sample support, source window, physical coordinate calculation, model,
baseline, beam, search bound, or acceptance criterion changes.

The full synthetic suite and a successor checksum freeze are required before
the anchor can be inspected again.
