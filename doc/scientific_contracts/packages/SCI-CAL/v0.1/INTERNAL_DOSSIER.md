# SCI-CAL — Internal Stage A Dossier

Status: internal scope evidence; never an author reference

Date: `2026-08-16`

This dossier records why SCI-CAL is a coherent Citlali package and where the
current software appears to realize or consume its facts. It is deliberately
separate from [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md). Nothing here establishes the
scientifically correct estimator, threshold, model, unit conversion, validity
rule, or product representation.

## Apparent Current Transformation

At a high level, the application appears to combine observation-resolved
calibration intent, a selected measured/matched APT, raw sample state,
per-sample elevation and opacity information, and factor/validity state to
produce calibrated timestream values and downstream calibrated product
metadata. Beammap also appears to produce calibration quantities later
consumed through the APT ecosystem.

That apparent flow motivates this package boundary. It is not the scientific
answer and is not reproduced in the author-facing brief.

## Current Citlali Ownership Evidence

The following paths were inspected on
`origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20`
for scope only:

| Area | Representative paths | Scope question exposed |
| --- | --- | --- |
| Timestream factor application | `include/citlali/core/timestream/rtc/calibrate.h`, `include/citlali/core/timestream/rtc/rtcproc.h` | Which sample signal and factor recipients belong to CAL? |
| PTC/downstream weight use | `include/citlali/core/timestream/ptc/ptcproc.h` | Which conditional weights may depend causally on calibration? |
| APT calibration fields | `include/citlali/core/engine/calib.h` | Which measured fields and units cross the selected-APT boundary? |
| Observation resolution | `include/citlali/core/pipeline/observation_calibration*.h`, `include/citlali/core/pipeline/reduction_observation_calibration.h`, `include/citlali/core/pipeline/flux_calibration.h`, `include/citlali/core/pipeline/flxscale_correction*.h` | Which requested, selected, and realized states need separate scientific identities? |
| Extinction selection | `include/citlali/core/timestream/extinction_model_selection.h` and observation setup paths | Which atmosphere inputs, support, and provenance must the contract define independently? |
| Beammap production | Beammap calibration and flux-conversion implementation paths, including `beammap_flux_conversion_impl.h` | Which Beammap quantities are CAL inputs versus BEAM-owned estimators? |
| Configuration | calibration/extinction entries in the resolved config-leaf contract and associated config readers | Which choices are observation-resolved, and which invalid requests should be scientifically distinguishable? |
| Product/provenance consumers | raw-timestream provenance, map primary-header helpers, Beammap setup metadata, and product contracts | What minimum lineage and validity must downstream products resolve? |

The table intentionally omits function-level conclusions. The current path
layout can change without changing the contract.

## Cross-Repository Boundary Evidence

| Owner | Apparent responsibility relevant to SCI-CAL | Must remain outside SCI-CAL authority |
| --- | --- | --- |
| TolProj | Project/cohort APT seed selection, calibrator interpretation, invocation and use of matching, pointing-derived science APT flux correction, and binding the selected artifact to an observation | Catalog/calibrator selection policy, cohort estimator, pointing-derived correction estimator, and library curation |
| TolAPT | Design-to-measured matching and new immutable provenance-bearing matched products from immutable inputs | Physical/design identity inference and matcher science |
| TolTECA v2 operational line | Selected Citlali inputs and configuration delivery | Delivery-time defaults, compatibility conversions, and operational file mechanics as scientific authority |
| Citlali Beammap | Measured APT and Beammap-derived calibration fields | Beam inference, source fitting, realized beam uncertainty, and empirical response validation, which require BEAM authority |
| MAP/FLT | Realized map/filter response and downstream product construction | Mapmaking, kernel response, filtering, and empirical response-fidelity science |
| ALIGN/AST | Aligned sample identity, time, eligibility, elevation, and pointing/astrometric meaning | Active ALIGN B3c work and all independent ALIGN/AST estimator choices |

## Implementation-Informed Risks To Sanitize

The author-facing scope must be strong enough to address these scientific
risks without revealing current implementation answers:

- a factor can be omitted, duplicated, inverted, or applied at the wrong
  reference plane;
- a valid numeric factor can be attached to the wrong target row, source APT
  row, observation, network, or artifact;
- a conditional weight can be mislabeled as total calibrated uncertainty;
- a `mJy/beam` label can survive a response-changing operation without a
  preserved point-source response definition;
- a discrete atmosphere selector can introduce nonphysical discontinuities;
- science-qualified, engineering-only, unsupported, disabled, unavailable,
  and failed calibration states can be collapsed into a boolean;
- a pointing-derived APT correction can be applied twice or lose its parent
  lineage; and
- a non-`xs` measured detector stream can acquire a calibration label without
  an approved physical meaning; and
- downstream products can become irreproducible if they cannot resolve the
  selected APT, applied factor definition, response basis, and validity.

## Sanitization Record

The Scope Brief retains only the abstract scientific boundary, owner-approved
decisions, legitimate inputs and outputs, external ownership rules, and open
questions. It excludes:

- source paths and function/class names;
- present calculations, defaults, thresholds, and control flow unless already
  elevated by an owner scientific decision;
- audit verdicts and repair findings;
- current test and validation behavior;
- file-layout prescriptions; and
- implementation, validation, production, and re-audit status.

No part of this dossier may be copied into an author packet. If an item here
reveals a genuine missing scientific question, only the abstract question may
be added to the Scope Brief.
