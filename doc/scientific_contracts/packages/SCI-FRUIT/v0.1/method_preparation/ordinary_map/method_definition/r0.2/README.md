# SCI-FRUIT — ordinary-MAP method definition r0.2

Date: 2026-09-08. Scientific owner: Grant Wilson.
Identity: `SCI-FRUIT-ORDINARY-MAP-PAPER-METHOD-R0.2`.
Status: residual-relearning direction approved; targeted paper successor ready
for review. Complete method and exact upstream amendments remain pending.

## Program adherence and prior-work recovery

This work follows the [charter](inputs/program/README.md),
[pilot workflow](inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md),
[frozen FRUIT core](inputs/authority/fruit/SCI_FRUIT_STAGE_B_R0.4_OWNER_CONDITIONAL_FREEZE_2026-09-08.md)
and exact frozen MAP v0.1/r0.7.1. The [recovery delta](PRIOR_WORK.md) retains the
previous packet and authoritative science. [Scope D001–D005](OWNER_SCOPE_APPROVAL_2026-09-08.md)
and the [residual-relearning direction](RELEARNING_DIRECTION_AND_REVIEW.md) are
resolved; no new scope cycle or generic core authorship is requested.

## What changed

Each feedback iteration now constructs a fresh model-subtracted residual from
the original calibrated parent, learns PTC centering and correlated subspace
from that residual, resolves the state, and applies it to the same residual.
The declared PTC recipe stays fixed. Fitted state is held fixed during its own
application and conditional response queries, then learned anew next iteration.
The model affects learning through the residual even though its value path
bypasses direct PTC cleaning. A fresh fit may return the same subspace.

This replaces the fixed-bootstrap recommendation with the distinct proposed
method `FRUIT-FEEDBACK-METHOD/ordinary-map-residual-relearning@r0.2`.
The reason is to address astronomical influence on correlated-subspace
estimation. No recovery improvement or convergence has been demonstrated.

The [method rules and twenty-one classes](METHOD_DEFINITION.md) retain the
unseeded bootstrap, complete-map replacement, proposed support-only/both-sign
selector, matched projection, uniform retained gridding coefficients, fixed
required populations, finite extent and failure rules. Those other method
choices remain proposals. In particular, the selector protects noise structure
as well as sources; relearning does not validate it or authorize a new threshold.

The [boundary table](PTC_MAP_BOUNDARIES.md) now requests PTC residual **learning
and application** admission. [U01–U08 and the question record](DISPOSITIONS_AND_OWNER_QUESTIONS.md)
separate decided direction, remaining method choices, exact amendments,
configuration bindings and execution. The [historical comparison](HISTORICAL_COMPARISON.md)
preserves the JINC control and separates mapmaker effects from feedback effects.

## What remains

The approved contracts do not yet admit the FRUIT residual as this PTC pass's
parent or the rejoined FRUIT child as ordinary MAP input. Those exact amendments
and reference/support compatibility remain pending. Applicable PTC settings,
gridding family/QC and numerical MAP support policy also need exact bindings.
This is what `unavailable_under_current_frozen_parent_permissions` means;
the missing items are not precomputed eigenvectors or a requirement to freeze
them across iterations. No implementation or execution readiness is inferred.

The next review concerns the remaining method choices and exact boundary work.
Residual relearning and the approved scope are not asked again. No experiment
design, execution, author dispatch or upstream successor adoption occurs here.

## Delivery and preservation

[Reference control](REFERENCE_CONTROL.md), [source identities](SOURCE_IDENTITIES.json)
and the unchanged [24-entry proposed-reference inventory](AUTHOR_REFERENCE_INVENTORY.json)
retain the three corrected process-only flags. The supplied discussion remains
manager-only. Any future sanitized author packet needs separate exact approval.
[Verification](VERIFICATION.md) and [manifest](PACKET_MANIFEST.md) bind this
paper packet and its tar.gz. The prior packet/archive, frozen core and upstream
contracts, historical control and existing products are unchanged.
