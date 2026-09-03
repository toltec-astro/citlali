# SCI-BEAM — Internal Stage A Dossier

Status: internal scope evidence; never an author reference

Date: `2026-08-16`

This implementation-informed dossier explains why the proposed boundary is
coherent and which current artifacts appear to realize or consume it. Nothing
here establishes the scientifically correct beam model, calibrator model,
likelihood, fit support, prior, covariance, convergence rule, QC threshold,
calibration factor, sensitivity, or promotion policy.

## Anti-Repetition Strategy

The historical BEAM audit never launched, so there is no independent core or
audit to reuse. Stage A therefore reused its inventory and incoming dependency
handoffs, then inspected only living authority documents and sibling ownership
records needed to define the package. It did not redo a source audit or test a
reduction. Existing implementation facts are quarantined here; only abstract
questions and stable owner boundaries enter the Scope Brief.

## Apparent Current Transformation

Citlali's Beammap mode appears to begin with detector-resolved conditioned TOD,
per-detector maps, identities, an APT, source identity and per-array flux,
effective mode policy, and optional soft spatial priors. It iterates between
map formation, source-aware conditioning, candidate selection, and per-detector
fitting. It then emits detector maps, fitted parameters and errors, convergence
and QC state, calibration/sensitivity-related columns, an observation-local
APT, optional fit-QC, and optional bounded detector TOD/diagnostic products.

That description is a scope inventory. It is not authorization for the current
operator or product meanings.

## Apparent Product Families

| Family | Apparent current role | Stage A boundary treatment |
| --- | --- | --- |
| Per-detector map cube | Signal, coefficient/weight, and kernel planes in an AltAz tangent plane | MAP supplies the admitted map bundle; BEAM owns only its model-specific use and fit support |
| Observation-local APT | Detector identity plus fitted position, beam shape, amplitude, calibration/sensitivity candidates, flags, and KIDs quantities | Proposed BEAM result bundle; authority and promotion must be decomposed explicitly |
| Fit-QC table | Inputs, bounds, prior/candidate state, fit state, convergence, and diagnostic decisions | Proposed required conformance/provenance companion, not an independent scientific estimate |
| Detector source-crossing TOD | Bounded processed windows, slots, sample counts, distance, signal, and flags | Optional diagnostic/input-evidence product; does not by itself establish the fitted model |
| Split detector maps | Accepted/rejected detector partitions | A view of explicit QC state; partitioning must not create scientific validity by filename |
| Beammap provenance | Requested/effective/observation/realized lifecycle and output cardinality | Engineering evidence; scientific contract must define which state is scientifically material |

## Apparent Policy Families

The current 74-leaf Beammap policy snapshot covers iteration/phase policy,
reference-detector handling, RFI and scan-band masking, detector weighting and
TOD selection, Gaussian-fit support, split output, soft priors, quality flags,
and sensitivity-band policy. `beammap_source.*` photometry is deliberately
adjacent rather than part of that policy surface.

This inventory is useful because it prevents omitted questions. It is not a
scientific schema, and the number 74 has no intrinsic scientific meaning.

## Protected Interfaces

### CAL and photometry

TolProj selects the calibrator and estimates per-array flux; TolTECA supplies
it. SCI-CAL owns the scientific meaning of calibrated samples and any promoted
detector calibration. BEAM may compare a declared source model with an admitted
amplitude estimate and publish a typed candidate factor with uncertainty and
lineage. It may not silently turn that candidate into calibration authority.

### ALIGN/AST

BEAM may consume a declared sample-to-coordinate relation, detector identity
binding, frame, validity, and uncertainty. It may estimate a source centroid
relative to that frame. It does not own physical timing, absolute pointing,
detector-coordinate truth, astrometric correction, or the mapping from a fit
centroid to those quantities. Active ALIGN work remains outside recovery.

### RTC/PTC/VAL/MAP

RTC and PTC own conditioned signal, causal validity, response, and coefficient
meaning. VAL owns upstream eligibility and non-finite policy. MAP owns the
admitted map estimator, WCS, support, response companion, and map validity.
BEAM owns only the subsequent model fit and BEAM-specific validity; it cannot
repair or relabel an incomplete upstream response.

### TolAPT

TolAPT owns matched/reference APT construction and soft-prior production. The
recovered producer contract says the prior is array/network/slot-local soft
initialization and gating information, not exact detector identity or measured
position truth. Weak regions remain broad and blind fallback remains required.
SCI-BEAM must define what a compatible prior may do to a fit and what evidence
would be required to promote a new prior policy.

### `toltec_beammap`

The downstream repository owns analysis, calibration use, APT diagnostics and
updates, planet calibration, and sensitivity utilities. Its current status
does not identify the A/B evidence for the newest priors. SCI-BEAM must give it
unambiguous inputs and states without absorbing its later analysis algorithms
or treating unreviewed local scripts as evidence.

## Historical Dependency Warnings

- CAL handoff: source flux divided by a fitted/template amplitude and later
  sensitivity use need a single explicit factor/uncertainty lifecycle.
- AST handoff: table order and slot labels cannot replace exact detector
  identity, coordinate validity, frame, and unit binding.
- RTC handoff: an enabled signal-conditioning projection that is absent from a
  kernel prevents the kernel from being called the complete realized response.
- PTC/VAL material: no separate BEAM handoff was created on the recovered
  ledger, but consumer restrictions and owner briefs still require explicit
  upstream eligibility, response, and coefficient status.

These are questions and restrictions, not accepted BEAM findings. The future
author does not receive the raw handoffs.

## Boundary With Iteration

The apparent internal loop has locator and measurement phases, repeated map
passes, fitting, optional source-aware reconditioning, and convergence. That
can belong to BEAM if it is an observation-local estimator for one detector
result bundle. It must not acquire FRUIT's general map-to-TOD feedback,
learning, recurrence, restart, or cross-iteration science-product authority.

## Information Firewall

The proposed author-facing packet may contain only:

- the owner-approved sanitized Scope Brief;
- a content-bound stable convention and ownership extract;
- owner-approved primary scientific references; and
- explicit conditional interface extracts from adjacent contracts when the
  owner approves them.

It excludes source paths, class/function names, current parameter values,
product column lists, audit findings, repair instructions, tests, A/B results,
validation evidence, branch state, Unity state, production status, and this
dossier. Questions discovered here are transferred only in abstract form.
