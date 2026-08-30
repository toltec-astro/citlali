# SCI-FLT v0.1 Prior-Work Recovery

Date: `2026-08-30`

Status: Stage A recovery record; implementation-informed and excluded from
implementation-blind Stage B authorship

## Program Adherence And Prior-Work Recovery

This recovery follows the library
[program charter](../../../README.md),
[pilot process](../../../PILOT_PROCESS_REVIEW_2026-08-16.md),
[downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md), and
[prior-work registry](../../../PRIOR_WORK_REGISTRY.md). It began before any new
scientific derivation. Recovered material was classified as `adopt`, `cite`,
`abstract`, `supersede`, `defer`, or `exclude` before selection of a proposed
author packet.

The recovery starting state was repository commit
`b2bfacea32c23a8538a7ea6660345c9edbfe7758`. The approved SCI-NOI Stage A
authority was checked against tag `sci-noi-v0.1-stage-a` at
`be25010b4`. Only the unchanged Stage A records through the final owner
approval were used as authority. The later `stage_b/` tree present in the
starting state was treated as draft and excluded.

## Discovery Procedure

Recovery searched:

- the living package index, registry, roadmap, and refactor status;
- frozen or approved SCI-MAP, SCI-JINC, SCI-BEAM, RTC, PTC, CAL, VAL, ALIGN,
  AST, and SCI-NOI records relevant to the filtering boundary;
- current map-filter configuration, execution, product, provenance, source-fit,
  Pointing, OOF, Beammap, NOI, and FRUIT surfaces;
- repository history and topic refs for `Convolve`, `Wiener`, `lowpass`,
  `matched`, `destripe`, `filter`, and `SCI-FLT`;
- the historical Convolve independent/audit document and subsequent owner
  coordination decisions; and
- tests and validation artifacts only to identify implemented or historically
  exercised states.

No Unity system was queried.

## Recovered Scientific And Program Material

| Material | Classification | Stage A disposition |
| --- | --- | --- |
| Scientific Contract Library Program and pilot process | `adopt` | Governs recovery, firewall, and Stage A/Stage B separation. |
| Owner-approved downstream roadmap, tranche 4 | `adopt` | Requires deterministic convolution/low-pass to be separated from Wiener and other inference-bearing filtering. |
| Frozen SCI-MAP v0.1/r0.7.1 | `cite` | Supplies immutable parent-map identity and honest response/covariance availability; does not define filtering. |
| Frozen SCI-JINC v0.1/r0.3 | `cite` | Supplies a distinct signed-estimator parent whose numerical route may be unavailable; does not define filtering. |
| Approved SCI-NOI Stage A through ODQ-110A/B/C and ODQ-111 | `adopt` at the boundary only | FLT defines the exact transformation; NOI applies it to compatible randomizations. Fixed-state, successor-generation, and per-member-relearned routes remain distinct. |
| Current SCI-NOI `stage_b/` material | `exclude` | Draft, not frozen authority, and outside the user-authorized starting authority line. |
| Frozen SCI-BEAM v0.1 | `cite` | Owns Beammap effective-PSF, source-fit, and related product interpretation; possible template/response producer, not FLT authority. |
| Frozen RTC/PTC/CAL/VAL/ALIGN/AST authorities | `cite` | Retain their own temporal filtering, transformed-timestream, calibration, policy-evaluation, frame, and astrometric ownership. |
| Historical Convolve mixed document at `codex/convolve-contract-audit@800e8ae433f87d3fb7521fcb1a7fdf1d32532949` | `abstract` and `exclude` | Abstract only fixed affine/convolution, response, support, and covariance identities. Exclude source inspection, findings, verdicts, repairs, tests, validation, and historical product claims. |
| Historical FLT-D001 fill-boundary decision | `cite` as prior owner direction | Candidate for reaffirmation: numerical fill is outside scientific admission and admitted support is eroded by the operator footprint. It is not silently promoted to current SCI-FLT authority. |
| Historical FLT-D002 aperture-response decision | `cite` as prior owner direction | Candidate for reaffirmation: transformed map amplitude is not automatic photometry; identically transformed unit-source response can support user-applied response correction subject to CAL. |
| Historical FLT-D003 empirical-calibration decision | `defer` | Its placement of empirical calibration inside FLT must be reconciled with approved SCI-NOI ownership before use. It is not in the proposed author packet. |
| Historical CAL-to-FLT handoff `SCI-FLT-001-XAUD-001` | `exclude` from authorship; `cite` internally | Audit-era dependency evidence, not current CAL scientific authority. It reinforces that unit labels alone do not establish response, covariance, or absolute calibration. |
| Internal `citlali_noise_estimation_plan.tex` | `abstract` only through approved NOI Stage A | Useful historical derivation, but old ensemble estimators and product recommendations are superseded wherever they differ from approved NOI Stage A. |
| FRUIT checkpoint/convergence and source/mode planning records | `cite` internally | Establish current ownership and lifecycle boundaries only. They are not FLT scientific inputs. |

## Recovered Implementation And Evidence Material

The following material is classified `exclude` from Stage B authorship and is
quarantined in [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md):

- map-filter enum, template, tail, edge-guard, tolerance, and output controls;
- the sequential and OpenMP Wiener/Convolve implementations and the unbound
  Gaussian helper;
- observation and coadd execution order, noise-realization application,
  product writing, source finding/fitting, and Pointing fit routing;
- requested/effective/realized configuration and provenance structures;
- the current product registry and historical product-contract documents;
- unit tests, audit records, repair records, baseline comparisons, and Unity
  validation artifacts.

These records establish only what code paths, labels, and evidence states were
found. They cannot select the scientific estimand, transformation, response,
uncertainty meaning, validity rule, or acceptable behavior.

## Package-Specific Recovery Results

1. `Convolve` and the current `lowpass_only` route share a fixed-kernel
   convolution mechanism when all operator state is fixed, but their purposes
   and method identities are not yet scientifically specified.
2. Full Wiener construction depends on a noise spectrum, map weights,
   template, denominator rules, and regularization/truncation choices. It is
   inference-bearing even when the resulting realized operator is later
   frozen.
3. No dedicated current matched-filter estimator was recovered. Template-
   sensitive Gaussian, Airy, and kernel paths exist, while source fitting and
   response interpretation are downstream. A future contract must distinguish
   smoothing by a template from estimating a template amplitude.
4. A data-thresholded map-domain `destripe` routine and configuration value
   were recovered, but the execution call is disabled. Its available
   scientific purpose, output identity, response, and uncertainty are
   unavailable.
5. The standalone Gaussian helper is not bound into the current execution
   lane. It is evidence of prior method work, not an authorized method.
6. No dedicated approved `SCI-FLT-002`, Wiener, low-pass, matched-filter, or
   destripe scientific core was found in the searched repository history.

## Sanitization Outcome

Only the fixed-transformation mathematics in
[`AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md`](AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md)
was abstracted from the mixed historical Convolve record. Current authority
boundaries were restated without implementation details in
[`AUTHOR_BOUNDARY_INPUTS.md`](AUTHOR_BOUNDARY_INPUTS.md). All other recovered
implementation, audit, repair, validation, historical verdict, and draft NOI
Stage B material is excluded from the proposed future author packet.
