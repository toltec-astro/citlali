# SCI-FLT-MATCHED v0.1 — Sanitized Scientific Boundaries

Status: sanitized author input

## MAP Parent

Admit only one immutable normalized ordinary-MAP observation bundle or one
immutable normalized ordinary-MAP coadd bundle. Preserve exact WCS/frame,
grouping, parent generation, support/validity, signal unit and quantity,
response/calibration provenance, and coadd membership. A parent coefficient
does not become covariance or precision by name, positivity, or units. Exact
numerical parent gates remain gates; this contract cannot create missing
authority.

## NOI

The exact realized FLT state is fixed before compatible realization members
are transformed. Every member receives the identical transformation, support,
normalization, and failure rules. An NOI companion retains its own estimand
and population and is not automatically physical-noise covariance,
significance, or `C_cond`. NOI-informed state updates and per-member relearning
create separate immutable generations and methods.

## CAL And BEAM

Template and parent calibration dependence is joint. Literal point-source
flux meaning requires exact point-source amplitude convention and compatible
CAL/BEAM lineage. Other templates yield shape amplitudes. Parent beam remains
provenance; any matched-filter beam or solid angle is derived from the exact
response under an explicit measure and validity convention. Missing
calibration covariance is unavailable.

## VAL

FLT owns producer facts, named-use policies, actions, and failure semantics.
VAL registers and evaluates exact immutable profiles but does not invent
admission, publication, covariance, response, consumer, or fallback policy.

## FLT To FRUIT

The product contract preserves a first-class interface sufficient for a future
FRUIT method to consume the filtered amplitude product without reconstructing
undocumented FLT implementation state. It identifies at least the filtered
quantity, template, response, valid support, units/calibration, method/state
identity, uncertainty availability, and provenance. Additional authorized
state remains exactly available or reconstructable through lineage.

This interface supplies no FRUIT source model, subtraction/add-back,
recurrence, learning, stopping/restart/selection, response, uncertainty, or
interpretation. Those belong to a separate FRUIT contract tranche.

## Excluded Downstream Interpretation

The filtered map is not detected sources, candidates, peaks with scientific
meaning, fits, deblended objects, catalog rows, significance, completeness,
purity, or morphology. No current source-analysis owner or dependency is
introduced.

