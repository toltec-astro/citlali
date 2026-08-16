# SCI-CAL v0.1 — Sanitized Conventions And Ownership

Status: owner-approved author reference

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

This extract contains only stable scientific conventions and responsibility
boundaries approved for implementation-blind SCI-CAL authorship. It is not an
implementation description, audit, product schema, or production-status
record.

## Scientific Identity And Shape

- TolTEC arrays are `a1100`, `a1400`, and `a2000`. Networks 0--6 belong to
  `a1100`, 7--10 to `a1400`, and 11--12 to `a2000`. Array, network, detector,
  acquisition, table-row, and map identities are distinct.
- The ordinary detector timestream shape is samples by detectors. Container
  or column position is a locator, not a scientific identity.
- An observation-local acquisition key consists of observation/Tune context,
  network or interface, and the network-local channel/tone slot. A verified
  ordered-row binding or explicit keyed mapping may connect this key to a
  sample column.
- A measured Beammap APT row is identified within one immutable artifact by
  that artifact's identity plus its local row key and observation/network
  context. A UID is artifact-local, not a persistent physical detector name.
- Cross-observation target-to-source association is a separate matcher edge.
  It retains both occurrence-scoped endpoints, method/version, disposition,
  and available quality evidence. Aggregate matcher performance is not a
  per-row probability.
- Design identity is a separate TolAPT match result. Ordinary SCI-CAL use of
  measured Beammap calibration quantities does not require a design match.
- Occurrence identity, order-independent semantic-content identity, and exact
  byte-transport identity are separate. Equal local keys, row positions,
  paths, timestamps, or integer spellings do not establish cross-artifact
  correspondence.

## Units, Reference Plane, And Response

- SCI-CAL v0.1 applies only to `xs` and produces only top-of-atmosphere,
  point-source-peak `mJy/beam`.
- `tau225` is dimensionless zenith optical depth. Elevation is an angle;
  persisted or exchanged values must state their unit. Frequencies and
  passband axes carry explicit units.
- Atmospheric extinction uses full eligible-sample airmass with reference
  plane `X_ref=0`. A zenith band opacity is not itself a line-of-sight sample
  correction.
- Weight is inverse signal-unit squared; conditional variance is signal-unit
  squared. These are conditional measurement quantities unless a separate
  total-uncertainty contract says otherwise.
- Per-detector Beammap beam/template identity is distinct from the realized
  response of a mapmaker, convolution kernel, or filter. Elliptical beam
  parameters are retained when available; a circularized beam is a labeled
  approximation.
- `MJy/sr`, `Jy/pixel`, temperature, extended-source calibration, and
  integrated photometry are not authorized by v0.1.

## Validity, Missing Information, And State

- Missing, disabled, automatic, unavailable, invalid, engineering-only, and
  science-qualified are distinct semantic states. They are not undocumented
  zero, unity, negative, or non-finite sentinels.
- Unknown or unavailable uncertainty is never represented as zero.
- Requested, effective, observation-resolved, and realized calibration state
  are distinct and flow one way.
- Negative or non-finite opacity, invalid airmass/elevation, nonpositive or
  non-finite transmission, missing required identity, unsupported units, and
  out-of-domain model requests cannot be silently clamped, extrapolated, or
  relabeled as calibrated science.
- A coherent observation or explicitly declared processing segment receives
  one calibration-quality class. The class does not switch sample by sample.
- Only `0 <= tau225 <= 0.15` may seek science qualification under v0.1. For
  `0.15 < tau225 <= 0.25`, no calibrated SCI-CAL output is authorized until a
  continuous engineering operator is separately adopted. Larger, absent, or
  non-finite opacity is outside supported calibration.

## Scientific Responsibility Boundaries

- **TolProj** owns project/cohort APT seed selection, calibrator
  interpretation, use of matching, pointing-derived science APT flux
  correction, library curation, and binding the selected artifact to an
  observation. SCI-CAL consumes the selected result and prevents omitted or
  duplicate application; it does not redefine the TolProj estimator.
- **TolAPT** owns design-to-measured matching and immutable, provenance-bearing
  outputs from immutable inputs. SCI-CAL consumes an admitted association and
  does not define matcher policy.
- **TolTECA's operational boundary** owns delivery of selected inputs and
  configuration. SCI-CAL defines their scientific admission and meaning, not
  delivery defaults or compatibility conversions.
- **Beammap and downstream calibration analysis** own estimation of measured
  calibration, sensitivity, and beam quantities. SCI-CAL owns their admitted
  use, factor meaning, once-only composition, validity, uncertainty transfer,
  and realized calibration lineage.
- **ALIGN/AST** owns the ordered common sample axis, time, eligibility,
  elevation/airmass input meaning, and pointing/astrometric semantics.
  SCI-CAL consumes approved abstractions without rederiving those estimators.
- **MAP/FLT** owns mapmaking, convolution/filter response, and empirical
  downstream response fidelity. SCI-CAL records the response basis needed to
  interpret calibration but cannot certify that downstream response.
- Downstream consumers may construct full covariance from approved nuisance
  terms. They may not call conditional weight total uncertainty or
  statistical significance when required systematic terms are unavailable.

## Approved Factor Distinctions

- `flxscale` is the absolute detector calibration factor within the selected
  APT lineage.
- A TolProj pointing-derived flux correction may already be embodied in a new
  selected APT. Its ancestry and once-only inclusion must remain explicit.
- `responsivity` is a relative detector-response quantity for its separately
  declared role; it is not absolute flux calibration.
- `sens` is a separately declared sensitivity or approximate-weight quantity;
  it is not atmospheric extinction or a general total uncertainty.
- Any compatibility `fcf` value must state its exact contents, units, and
  exclusions. It is not an authoritative opaque total calibration.
- Atmospheric extinction, absolute detector calibration, relative
  responsivity, sensitivity, and future unit transfer remain separately named
  factors even when a realized signal multiplier composes some of them.

## Source Lineage Of This Sanitized Extract

The manager prepared this extract from owner-approved scientific content in:

- `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20:doc/SCIENTIFIC_CONVENTIONS.md`;
- the stable TolTEC shared-context summaries of scientific conventions,
  software boundaries, and APT/product ownership;
- TolProj's documented workflow and TolAPT's immutable output contract; and
- the identity principles of accepted-but-unactivated APT ADRs 0010 and 0011.

The author receives this extract, not those broader sources. No implementation
behavior, audit finding, repair requirement, test result, validation evidence,
or active ALIGN material is imported by this lineage statement.
