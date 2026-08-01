# SCI-ALIGN-001 coordinator decision brief — 2026-08-01

Status: audit integrated; owner decisions pending

Package: `SCI-ALIGN-001`

Governing source: `9aae0e669384c5c0c0dda93debc194d6b8dac787`

Audit artifact commit: `aeeac7f36e1ab0ab17bfbf3f603364faff02d715`

Audit identity tip: `9e234eada67c88feacddfc8b7e1afb0e1cffd818`

Final report SHA-256:
`6aaed0e6e16e4c37cd24d15b98346f84024ffd7920bd0524e7a170dbc728a393`

## Current disposition

The audit proposes `contract=proposed`, `implementation=nonconformant`,
`validation=in_progress`, `production=existing_use_only`, and `verdict=amend`.
Seven P0 implementation findings and seven P1 contract, evidence, policy, and
dependency findings remain open. No repair, Unity evidence, or re-audit is
authorized yet.

The eight decisions below should be taken in order because later choices use
the identities and validity states established by earlier ones.

## ALIGN-OD1 — Common grid and clock model

Question: What is the authoritative common sample axis, and what clock model
may ALIGN apply?

Recommended decision:

- use the detector stream as the common-grid and reference-clock authority;
- retain detector acquisition support rather than trimming it to telescope or
  HWPR availability;
- use explicit detector cadence and phase, stable observation/slot identity,
  one shared round-half-up slot operator, and a declared tolerance strictly
  below half a detector sample;
- allow only an observation-constant interface offset initially; clock drift
  requires a separately versioned model and evidence; and
- reject nonmonotonic timestamps, duplicates, collisions, incomparable epochs,
  and out-of-tolerance slot assignments.

Owner input still needed: the authoritative cadence/phase source and the
numerical slot-residual tolerance or the evidence rule used to set it.

## ALIGN-OD2 — Offset and header authority

Question: What does each detector/telescope/HWPR offset mean and where does it
come from?

Recommended decision:

- represent every offset in seconds relative to the detector clock;
- define a positive offset as added to the native coordinate to place it later
  on the detector-reference clock;
- apply it exactly once, before ordering, slot assignment, gaps, scans, or
  interpolation, without integer-sample rounding;
- record source, sign, unit, reference interface, application stage,
  resolution state, and uncertainty; and
- fail closed when any required fact is ambiguous. Zero is valid only when it
  is an explicit authoritative value, not a missing-value default.

Owner input still needed: the exact raw header/config authority for every
interface offset and its uncertainty or bound.

## ALIGN-OD3 — Telescope and HWPR field topology

Question: Which alignment operator is valid for every field?

Recommended decision:

- maintain a versioned per-variable registry naming unit, frame, topology,
  validity, maximum interpolation span, and operator;
- classify fields only as continuous linear scalar, circular angle with an
  explicit period/wrap rule, declared step state with half-open hold support,
  or non-interpolable/exact-only;
- never linearly interpolate timestamps, counters, flags, modes, Boolean or
  categorical states; and
- forbid extrapolation and interpolation across invalid rows or unapproved
  source gaps.

Owner input still needed: approval of the generated per-variable registry,
especially telescope state/counter fields and HWPR angle/state fields.

## ALIGN-OD4 — Gap limits and failure action

Question: Which missing runs may be synthesized, and what happens to the rest?

Recommended decision:

- require both a missing-slot count limit and an elapsed-duration limit;
- fill only a complete bounded internal run whose two endpoints are valid;
- never extrapolate leading or trailing gaps and never partially fill a long
  gap;
- preserve source identities, interpolation weights, reason, run extent, and
  zero acquired exposure for every synthesized cell; and
- distinguish marked-unavailable samples, failed scans, and failed
  observations with explicit thresholds, including a maximum unavailable or
  synthesized fraction.

Owner input still needed: numerical count, elapsed-time, and scan/observation
fraction thresholds plus the fail-sample/fail-scan/fail-observation actions.

## ALIGN-OD5 — Scan policy and identity

Question: How are scans constructed and preserved?

Recommended decision:

- default to one half-open scan covering the full observation unless an
  authoritative explicit-window or fixed-duration request exists;
- for fixed duration, use round-half-up to an integer number of detector
  samples and record both requested and effective duration;
- use stable zero-based full-observation scan IDs and preserve them in subsets;
- retain final partial and short scans with explicit status rather than
  dropping, padding, or renumbering them; and
- retain stable empty-window identity without invoking numerical processors;
  reject overlaps in the base contract.

This recommendation can be approved without selecting a universal minimum
science scan length; downstream consumers may reject a recorded short scan.

## ALIGN-OD6 — Synthesized-sample eligibility

Question: May an interpolated detector value count as an acquired scientific
sample?

Recommended decision:

- strict science eligibility is original-and-valid detector data plus all
  required available-and-valid aligned telescope fields;
- synthesized detector values are excluded by default from science
  eligibility, hits, acquired exposure, and independent statistical weight;
- a separate `continuity_available` state may authorize bounded algorithmic
  continuity use; and
- any broader consumer use requires its own approved response/covariance
  contract and must never relabel synthesized data as original.

## ALIGN-OD7 — Mapping, response, and covariance

Question: What practical representation must ALIGN publish?

Recommended decision:

- persist a sparse row mapping rather than a dense matrix: target identity,
  source row identities, interpolation method, weights, residual, origin,
  validity, reason, and gap/scan identity;
- publish the exact conditional mapping/response needed for downstream
  reconstruction;
- propagate formal covariance when the required input covariance exists,
  including shared-endpoint correlation; and
- represent timing, interpolation-model, and policy/selection uncertainty as
  named available/unavailable terms. Missing terms are never zero and a
  conditional covariance is not total uncertainty.

Owner input still needed: which compact conditional covariance products are
required in the first repair, versus explicitly deferred/unavailable.

## ALIGN-OD8 — HWPR separation and interim production

Question: How should optional HWPR timing and current production be treated?

Recommended decision:

- keep HWPR timestamp, angle, state, availability, and alignment identity in
  ALIGN when present;
- keep polarization demodulation and scientific polarization interpretation
  unavailable pending a separate approved contract;
- missing optional HWPR is explicit and nonfatal for nonpolarimetric modes;
  and
- retain `existing_use_only` for the exact previously accepted profiles while
  failing closed for new timing, synthesized-eligibility, exposure,
  covariance, response, HWPR, or polarization claims until repair and re-audit.

The owner may instead select package-wide `fail_closed` if the seven P0
findings are judged incompatible with any continued new reduction.

## After the decisions

Once OD1--OD8 are recorded, the coordinator may prepare a bounded ALIGN
repair/re-audit handoff from an exact selected application SHA. The repair must
begin with contract fixtures T01--T18 and must not broaden CAL, AST, RTC, VAL,
or polarization algorithms. Human-run exact-repair-SHA Unity evidence and a
fresh independent re-audit follow local closure; neither begins now.
