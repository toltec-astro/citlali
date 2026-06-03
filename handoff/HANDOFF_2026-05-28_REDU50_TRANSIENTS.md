# Citlali Beammap Handoff - redu50 RTC Transient Notes

## 2026-05-28

### Context

- Dataset: 3C273 beammap products under
  `/Users/gwilson/work_toltec/local_data/beammaps/3c273`.
- Main reduction inspected: `redu50`.
- Dashboard session: `http://127.0.0.1:8053/`, with `redu50` loaded and
  `redu42` as the comparison baseline.
- `redu50` was intended as a control run: fixed fruitloops flux cuts restored,
  adaptive fruitloops support disabled, 30 Hz lowpass, no downsampling, and a
  widened single fixed notch around the 29.5 Hz complex.

### redu50 Control-Run Assessment

- `redu50` used `v4.0.0-373-gb662887c` and completed cleanly in about
  `1h56m17s`.
- Its beam statistics are effectively back to the `redu42` baseline:
  - `a1100`: median FWHM `6.81"` and axis ratio `1.14`.
  - `redu49`, by contrast, had `a1100` median FWHM `7.46"` and axis ratio
    `1.25`.
- This strongly implicates the adaptive fruitloops source-support settings in
  the `redu49` beam broadening. The empirical-template calibration machinery is
  not the leading suspect for that broadening.
- `redu50` still reached `max iteration reached`; it did not globally stop via
  early convergence.

### PTC TOD Metadata Issue

- The attempted `FRUITLOOPS_ITER` metadata fix did not work for `redu50`.
- The log reports:
  `writing processed time chunk for beammap iteration 5`
- But the downloaded PTC TOD NetCDF still reports `FRUITLOOPS_ITER = 0`.
- The log also contains:
  `PTC TOD file ... has no FRUITLOOPS_ITER variable`
- This means the variable name lookup/update path does not match the file as it
  exists at update time. Do not rely on `FRUITLOOPS_ITER` in these TOD products
  until the writer is fixed and verified.

### Fixed-Notch Result

- The fixed and shared notch passes are being applied, including the widened
  fixed notch:
  - `11.0268 Hz`, width `0.75 Hz`
  - `25.1532 Hz`, width `1.00 Hz`
  - `29.55 Hz`, width `0.90 Hz`
  - `33.2889 Hz`, width `0.25 Hz`
- For uid `332`, detector-level PTC line ratios are low, but network-median PTC
  residuals can still show a line near the 29 Hz complex.
- Conclusion: the broad fixed notch did not obviously solve the common-mode
  residual in the PTC diagnostics, though it also did not recreate the `redu49`
  beam broadening.

### Coincident RTC Transient Case

Specific case inspected from the dashboard:

- uid `3010`
- scan `101`
- product: RTC downloaded outer scan
- array/network: `a1100`, `nw5`
- event local sample: `1114 -> 1115`

Timing relative to scan:

- Downloaded RTC outer scan length: `1825` samples.
- Inner mapped scan region: local samples `512` to `1312`.
- Event sample: local sample `1114`.
- The event is inside the mapped scan, about `75%` of the way through the row.
- It is about `198` samples, or `1.62 s`, before the inner scan end.
- Map speed at the event is about `48 arcsec/s`, close to normal scan speed.
- Therefore this is not simply the telescope starting the end-of-row
  turnaround.

Dashboard note:

- The orange dashed marker in the TOD plot marks the largest robust outlier in
  the plotted trace, not necessarily the source-crossing sample.

### Coincidence Across Detectors

Using the exact derivative sample `1114 -> 1115`:

- Same network `nw5`: `274 / 481` detectors have `>5 sigma` negative
  derivative within `+/-3` samples.
- Same network `nw5`: `140 / 481` detectors have `>8 sigma`.
- Same a1100 array: `1682 / 3166` detectors are `>5 sigma` within `+/-3`
  samples.
- All detectors: `2545 / 5491` detectors are `>5 sigma`.
- At the exact event sample, `94%` of same-network detectors and `90%` of all
  detectors have the same negative sign.
- The same-network median timestream has this as the largest derivative event
  in the downloaded outer scan.

This is not a single-detector glitch.

### Detector Characteristics Of Strong Responders

Strong responders were defined as `signed derivative z < -5` at the exact event
sample.

Network fractions:

- `nw0`: `0.5%`
- `nw1`: `0.0%`
- `nw2`: `0.0%`
- `nw3`: `51%`
- `nw4`: `45%`
- `nw5`: `56%`
- `nw7`: `40%`
- `nw8`: `1%`
- `nw9`: `0%`
- `nw11`: `26%`
- `nw12`: `10%`

Spatial trends:

- Strong responders are biased toward positive `x_t` and somewhat positive
  `y_t`.
- In a1100, the `x >= 0, y >= 0` quadrant has about `62%` strong responders,
  while the `x < 0` quadrants are only about `7-10%`.

Tone-frequency trends:

- Strong responders have lower median tone frequency than the full population:
  about `595 MHz` versus `645 MHz`.
- This is a broad trend, not a narrow tone-frequency cluster.
- Tone frequency alone is not a sufficient explanation.

### Spatial Versus Network Modeling

The signed event amplitude was modeled using different predictors:

- Array-only model: `R^2 ~ 0.03`
- Network-only model: `R^2 ~ 0.37`
- Smooth quadratic focal-plane model: `R^2 ~ 0.19`
- Spatial plus tone-frequency model: `R^2 ~ 0.26`
- Spatial plus network model: `R^2 ~ 0.43`

After subtracting a smooth focal-plane model, networks `3`, `4`, and `5` still
have large negative residuals. This is the strongest argument against a purely
optical/sky transient.

### Interpretation

This event is:

- coincident across many detectors;
- visible across both LNA-bias-board groups;
- mostly the same sign;
- inside the mapped scan, not at turnaround;
- spatially asymmetric on the focal plane;
- still strongly network/topology dependent after subtracting a smooth spatial
  model.

Given the LNA-bias-board information, an isolated network-readout glitch is less
likely. The leading interpretation is broad coupled pickup/RFI or another shared
infrastructure transient whose coupling depends on readout/network topology and
focal-plane position. A purely optical/common-mode sky event is possible, but the
network residuals argue that it is unlikely to be the whole story.

### Suggested Next Steps

- Add a diagnostic view or offline script to identify coincident derivative
  events in RTC outer scans and summarize them by network, array, and focal-plane
  position.
- Compare several similar events, not just uid `3010` scan `101`, to see whether
  the same networks and focal-plane sectors recur.
- Check whether these events appear in housekeeping, telescope telemetry,
  timing/sync channels, or any available readout/power monitoring.
- For mapmaking, consider a coincident transient mask in RTC before PTC, based on
  network/all-array derivative coincidence rather than detector-local despiking.
- Treat these events separately from narrow-line contamination; their morphology
  is step/transient-like rather than a persistent sinusoidal line.
