# Proposed single experiment: fixed-template versus free-shape peak readout

Status: recommendation only; requires a separate owner decision before execution.
The current saved-product diagnosis has not performed these fits. Starlet stays
parked for POINT, and all existing outcomes and gates remain unchanged.

Question: how much of the crossed-noise peak error survives when the source
shape is held fixed in the measurement, on the identical reconstructed maps?
This isolates dependence on the free source-shape fit without changing FRUIT.

Inputs are exactly the saved pass-6 H and D maps: two seeds, two arms, three
arrays, giving 24 array maps. There is no source/population search, new data,
reserved observation, PTC learning, feedback inference or alternative iteration.
Use each array's existing supported 60-arcsec domain and its existing 52-arcsec
inner probe, for 48 total linear readout fits. The two domains are retained
comparators, not a new radius sweep.

For each map/domain fit four unrestricted linear coefficients in

    total_map = amplitude * unit_peak_truth_template
                + b0 + bx * x/90 + by * y/90 + residual.

Use the saved injected Gaussian shape, centroid and orientation for that case,
dividing its truth map by the declared analytic peak (100 for H, 90 for D),
not by its sampled pixel maximum. This makes amplitude refer to the same
continuous-peak convention as the known signal. Neither amplitude nor plane
is fixed to truth, and amplitude is signed. Use the same finite pixels and
unweighted least-squares objective as the retained free fit. Record rank,
conditioning, coefficients, residual norm and any inability to solve; do not
substitute a result on failure.

This deliberate use of injection truth is allowed only as a diagnostic ruler.
It is not a candidate production recipe or evidence that a deployable selector
knows the source shape. It must not affect any registered usability label.
The primary comparison is with the already saved free Gaussian-plus-plane fits
on those same two domains; do not refit that comparator during this experiment.

Report all terminal absolute-peak errors, same-seed and crossed-seed H/D ratios,
H and D seed changes, domain sensitivity and fitted-plane shifts for both
readouts. Preserve raw accuracy separately from the original availability.
Do not choose the better readout, domain or result using truth and then call
that a qualified method. No new pass/fail gate, confidence interval or production
claim follows from this small diagnostic.

If the fixed-template readout retains large crossed-noise errors, the saved
maps contain a substantial source-aligned noise/response component. This still
does not separate residual nuisance from PTC transfer bias. If the fixed-template
readout is materially more stable, free-shape/background measurement freedom
amplifies the observed variation; this does not yet separate optimizer effects
from morphology mismatch. Report a mixed result as mixed. The point is to narrow
one causal distinction before proposing any further repair.

This is the sole recommended next experiment. No OOF work, broad noise
qualification, new candidate, parameter sweep or further experiment is bundled
with it or automatically follows it.
