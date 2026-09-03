# SCI-POINT v0.1 Author Working-Method Constraints

Identity: `SCI-POINT_WORKING_METHOD_CONSTRAINTS v0.1/r0.3`

Status: sanitized author input candidate

## Adopt Without Reinvention

- one known, isolated, bright Pointing source approximately centered in one
  observation-local map for each requested TolTEC array;
- the zero-background six-role elliptical-Gaussian source-model family:
  amplitude, two-coordinate centroid, two width roles, and orientation;
- the existence of a mature bounded search-and-fit procedure, while keeping
  its width convention, exact objective/weights, search/fallback details, and
  solution rule unavailable until `POINT-COMPATIBILITY-METHOD v0.1` is
  recovered and approved;
- authoritative per-array fit roles containing centroid, amplitude, widths,
  angle, and honest state only when the compatibility method is available;
- marginal formal errors only when the distinct
  `POINT-FORMAL-ERROR-METHOD v0.1` is available;
- AltAz tangent-plane source displacements in arcseconds;
- established dynamic-range and formal-standardization diagnostics under
  truthful non-significance labels; and
- shared numerical fitter reuse with Beammap as an engineering possibility
  that does not merge POINT and SCI-BEAM scientific ownership.

## Abstract Scientifically

- Replace hidden configuration/default behavior with explicit requested,
  effective, and realized state.
- Identify exact parent route and complete FRUIT ancestry; never infer either
  from filenames or directory placement.
- State weight/covariance use, support, non-finite treatment, constraint
  realization, degeneracy, uncertainty meaning, and parent response.
- Preserve current scientific behavior while allowing implementation-neutral
  mathematics and terminology.
- Do not infer any missing compatibility or formal-error method from software,
  configuration, familiar practice, optimizer output, or Beammap similarity.

## Explicitly Exclude

- a new source-model family, free-background extension, blind detection, deblending,
  catalog construction, or blank-field faint-source fitting;
- per-detector Beammap fitting, intrinsic/effective detector PSF authority,
  sensitivity, or APT production;
- OOF optical inference;
- automatic MAP/JINC/FLT route selection, substitution, or fallback;
- coadd fitting or intermediate FRUIT iteration fitting;
- POINT-owned cross-array aggregation, correction sign/composition,
  correction-record selection/publication, or correction application;
- assuming missing covariance is zero, diagonal, or independence; and
- universal flux, intrinsic beam, statistical significance, unique QC cause,
  implementation conformity, validation, performance, readiness, or
  production claims.

## Truthful Diagnostic Identities

- legacy `sig2noise`: fitted amplitude divided by full-map RMS; retained as a
  dynamic-range diagnostic, not statistical significance, and unavailable
  until `POINT-FULL-MAP-RMS-METHOD v0.1` is approved;
- `fitted_amplitude_over_full_map_rms`: canonical name for that same
  descriptive ratio; and
- `fit_sig2noise`: fitted amplitude divided by its formal fitted-amplitude
  error; formal standardization only, not empirical S/N or detection
  probability.
