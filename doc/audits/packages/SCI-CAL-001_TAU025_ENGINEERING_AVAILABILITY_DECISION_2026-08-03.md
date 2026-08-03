# SCI-CAL-001 tau225 science-qualification and engineering-availability decision — 2026-08-03

Status: owner approved policy and bounded protocol-preparation authority only

Package: `SCI-CAL-001`

Decision ID: `CAL-ATM-D006`

Authority: project owner

## Decision

The atmospheric-calibration science-qualification boundary is not an
operational stop boundary.  Citlali must not silently extrapolate a
science-qualified atmosphere operator, but an observation used for engineering
above the routine science-opacity range should not fail merely because it is
not eligible for the science calibration claim.

This decision defines three future product-quality classes.  It does not adopt
an operator, alter the assessed application, replace the existing
fail-closed production disposition, or authorize an atmosphere-model run.

| tau225 condition | Future quality class | Permitted claim |
| --- | --- | --- |
| `0 <= tau225 <= 0.15` | `science_qualification_target` | May seek the strict numerical-fidelity and later observational gates for calibrated science. |
| `0.15 < tau225 <= 0.25` | `engineering_availability_target` | Reduction availability and a versioned engineering correction may be sought, but no science-quality calibration claim is permitted. |
| `tau225 > 0.25`, non-finite tau225, or absent required calibration identity | `outside_supported_calibration` | No silent extrapolation and no calibrated-science label. Any future uncalibrated or failed-calibration product handling requires its own explicit product contract. |

The existing EL25 decision remains a separate, confirmation-only elevation
condition; this decision changes no elevation boundary.  The already-frozen
q0--q75/EL20--80 study remains a failed numerical-adoption study, and the
subsequent EL25 confirmation remains separately bounded.  The present
successor evidence ends at the q75 anchor
`tau225 = 0.158313198574890929`; it does not establish a continuous operator
through `tau225 = 0.25`.  Generic q95 lineage remains diagnostic-only and may
not supply that extension by assumption.

## Boundary and provenance rule

The quality transition at `tau225 = 0.15` must not introduce a new
sample-by-sample correction-operator switch.  A future reduction/product plan
must select one declared correction operator and one compact calibration
quality state for each coherent calibrated observation or declared processing
segment, based on the maximum eligible tau225 in that unit.  A map containing
both low- and high-opacity samples is therefore engineering-qualified unless
it is explicitly partitioned before calibration under a later approved
contract.  Per-sample quality tagging is not authorized by this decision.

Until a separately validated continuous engineering extension exists, this is
planning policy only.  It does not authorize an implicit handoff from a future
science operator to the legacy selector at `tau225 = 0.15`, nor any new
fallback behavior in Citlali.

## Bounded next step: extension protocol preparation

CAL may prepare, but not execute, a separate
`SCI-CAL-001-TAU025-ENGINEERING-EXTENSION-001` protocol.  Its purpose is to
define a direct-AM, independently held-out characterization of a continuous
operator across `0.15 < tau225 <= 0.25`, using the already selected
content-bound TolTECA v1 ECSV passbands and the same explicit line-of-sight
optical-depth convention.

The protocol must:

1. identify the exact AM executable/input/profile, tau225 and elevation
   lattice, spectral integration, and immutable provenance required for
   direct truth calculations;
2. distinguish source anchors from independently held-out opacity/elevation
   points, and test continuity at `.15` without changing the correction
   operator at that boundary;
3. require finite positive transmission, monotonicity where physically
   applicable, exact node identities, and an explicit compact
   observation/segment quality/provenance product contract;
4. report the actual engineering-domain correction error separately from the
   science one-percent numerical-representation-fidelity gate; and
5. return a proposed numerical engineering criterion and an exact execution
   request for owner review before any AM run.

The protocol must not claim that engineering availability is science
calibration, infer generic-q95 provenance, or choose an error tolerance on its
own.

## Explicit exclusions

This decision does not authorize AM execution, Citlali or TolTECA source
changes, repair, re-audit, Unity access, operator or operational-domain
adoption, production-status change, or a new output format.  It preserves the
existing separation among software correctness, atmosphere-representation
fidelity, and observational calibration performance.
