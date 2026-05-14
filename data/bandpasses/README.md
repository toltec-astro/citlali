# TolTEC Bandpass Inputs

Citlali's current `uK` conversion is monochromatic Rayleigh-Jeans brightness
temperature. It uses each array's nominal center frequency and the Gaussian beam
solid angle to convert `mJy/beam` to intensity per steradian, so bandpass curves
are not used for the current conversion.

If finite-band color corrections are added later, place bandpass files in this
directory with one ECSV file per array:

- `toltec_a1100_bandpass.ecsv`
- `toltec_a1400_bandpass.ecsv`
- `toltec_a2000_bandpass.ecsv`

Use columns:

- `frequency_Hz`: observing frequency in Hz.
- `response`: dimensionless relative spectral response. Arbitrary
  normalization is acceptable if the integration code normalizes it.

Include measurement provenance in the ECSV metadata, especially the source of
the bandpass, measurement date or version, and whether atmospheric or optical
efficiency terms are included.
