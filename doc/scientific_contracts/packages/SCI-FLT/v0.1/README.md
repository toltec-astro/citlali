# SCI-FLT — Map-Domain Filtering Tranche

Status: recovery-first Stage A packet complete; scientific-owner walkthrough
required; implementation-blind Stage B not authorized

Version: `v0.1`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It begins from the current frozen and approved upstream authority line without
modifying SCI-PTC, SCI-MAP, SCI-JINC, SCI-VAL, RTC, CAL, ALIGN, AST, or the
approved SCI-NOI Stage A records.

Recovery began with the package-specific [`PRIOR_WORK.md`](PRIOR_WORK.md) and
the implementation-informed, author-quarantined
[`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md). It separately classified:

- fixed deterministic map-domain convolution and low-pass transformations;
- Wiener transformations whose realized operator depends on a noise model,
  weights, regularization, or other learned state;
- matched, template-sensitive, source-sensitive, and data-thresholded methods
  whose estimand or operator identity can differ from convolution;
- requested, effective, observation-resolved, applied, and realized lifecycle
  facts;
- transformed signal, response/kernel, support, validity, uncertainty, and
  lineage products; and
- historical audit, repair, test, and validation records as evidence only.

No dedicated approved Wiener or general low-pass scientific contract was
recovered. The earlier Convolve document is mixed scientific and
implementation/audit material. Its reusable fixed-transformation mathematics
has therefore been abstracted into a sanitized candidate rather than passed
through to a future author.

## Stage A Result

The initial taxonomy recommends that `SCI-FLT` remain the tranche name but not
the name of one monolithic Stage B contract. Before Stage B, the owner should
split at least:

1. a fixed deterministic transformation package, provisionally
   `SCI-FLT-DET`; and
2. an inference-bearing tranche, provisionally `SCI-FLT-INF`, whose Wiener,
   matched-estimator, source-learned, and data-thresholded methods remain
   distinct until their estimands, priors, learned state, response, and
   uncertainty meanings show that any can share one contract.

That recommendation is not yet scientific authority. It is the first owner
decision in [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).

The sanitized scientist-readable Stage A set is:

- [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md);
- [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md);
- [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md);
- [`AUTHOR_BOUNDARY_INPUTS.md`](AUTHOR_BOUNDARY_INPUTS.md);
- [`AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md`](AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md);
- [`AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md`](AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md); and
- the presently unavailable, owner-content-bound
  [`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md).

The proposed exact future author packet and its information firewall are in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md). It is intentionally
not releasable while owner decisions remain open.

## Controlling NOI Boundary

SCI-FLT owns the exact transformation: purpose, method identity, parameters,
fixed or learned state, order, domain, support, edge and missing-data behavior,
normalization, units, response, lifecycle, and failure policy. Under approved
SCI-NOI Stage A authority, NOI may apply that exact transformation to every
compatible admitted randomization to estimate uncertainty for the exact
transformed scientific product. NOI neither chooses nor defines the filter.
Fixed-state and per-member-relearned routes are different methods and may not
be mixed. The current SCI-NOI Stage B material is draft and was excluded from
this recovery.

## Current Gate And Nonclaims

- Prior-work and implementation recovery: complete for Stage A owner review.
- Ownership, boundary, and initial typed taxonomy: proposed.
- Package split and other scientific choices: owner decisions open.
- Future implementation-blind author inputs: enumerated but not released.
- Stage B authorship: not begun and not authorized.

This package makes no implementation-conformity, validation, calibration,
performance, readiness, production, or freeze claim. It changes no algorithm
and authorizes no algorithm change, evidence campaign, reduction, Unity work,
or downstream implementation action.
