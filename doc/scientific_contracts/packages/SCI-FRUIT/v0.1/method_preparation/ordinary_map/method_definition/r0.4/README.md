# SCI-FRUIT ordinary-MAP method definition r0.4

Date: 2026-09-09. Scientific owner: Grant Wilson.
Identity: `SCI-FRUIT-ORDINARY-MAP-PAPER-METHOD-R0.4`.
Status: scientific definition accepted and frozen after the three requested
clarifications. Q02 upstream amendments remain proposed and not adopted;
numerical policy instantiation and execution remain separately gated.

## Program adherence and prior-work recovery

This revision follows the [charter](inputs/program/README.md),
[accepted pilot workflow](inputs/program/PILOT_PROCESS_REVIEW_2026-08-16.md),
[roadmap](inputs/program/DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md) and
[frozen FRUIT core](inputs/authority/fruit/SCI_FRUIT_STAGE_B_R0.4_OWNER_CONDITIONAL_FREEZE_2026-09-08.md).
[Prior-work recovery](PRIOR_WORK.md) preserves the approved ordinary-MAP scope,
accepted residual-relearning direction, exact frozen upstream sources and
historical controls. The [owner review and correction crosswalk](OWNER_DIRECTION_AND_CROSSWALK.md)
records the three targeted clarifications and the [scoped freeze](SCIENTIFIC_OWNER_FREEZE_R0.4.md).
The prior mode-aware direction is otherwise unchanged.
There is no new scope cycle, generic contract/PDF set or author dispatch.

## Accepted scientific definition and reference scope

Each iteration relearns PTC contamination from the current sky-subtracted
original-parent data. FRUIT reconstructs the total map, then reconsiders a
replacement astronomical feedback model using an explicit observing-mode
policy. Previously admitted structure can change or be revoked; new structure
can enter. Repeated acceptance is not independent evidence of sky.

The leading reference model is an inference mask times the unfiltered total
map's calibrated flux, with unity application. Admission, flux estimation and
application are separate decisions. POINT, OOF, BEAM, compact blank-field and
extended/mixed SCIENCE policies have explicit required inputs and limits now.
Numerical scores/priors/thresholds remain visible choices. No universal sign
clip, extra flux floor, shrinkage or damping is assumed.

Binary admission × reconstructed flux is the leading ordinary-MAP reference
estimator, not a universal FRUIT restriction. Future fitted or continuous
astronomical components require their own declared estimator and response/
uncertainty treatment; they are not automatically confidence-weighted flux.

For the controlled reference, fixed MAP science support is the available domain
D_a and selected model support A_k can change within it. General FRUIT methods
may declare support changes, but must attribute and account for them so support
loss cannot masquerade as inference rejection or convergence. This does not
relax the controlled reference's fixed-support rules. The unthresholded/both-sign policy remains
a comparison baseline. Historical PCA plus empirical S/N and any actual flux
cut remains a practical baseline and fallback, with its JINC control preserved.
Complexity must earn useful recovery or contamination benefit for its regime.

## Four retained components

1. [Scientific definition and rationale](METHOD_DEFINITION.md): iteration,
   mode policies, model construction, PTC ownership, convergence, stability,
   recovery, response/uncertainty and the twenty-one method classes.
2. [Concise owner-direction crosswalk](OWNER_DIRECTION_AND_CROSSWALK.md):
   retained decisions, changes, precedence and exact unresolved conflicts.
3. [Engineering-conformance requirements](ENGINEERING_CONFORMANCE.md):
   observable runtime duties and targeted prospective tests, with no claim
   that tests have been executed.
4. [Minimal experiment proposal and precise open decisions](EXPERIMENTS_AND_OPEN_DECISIONS.md):
   exact historical/naive references first; POINT is the recommended first
   bounded screen, subject to owner authorization. OOF is a separately
   selectable proposal, not a mandatory second campaign. Independent-pointing
   replication remains required before a pointing-policy recommendation.

The supporting [bounded amendment annex](BOUNDARY_AMENDMENTS.md) supplies the
earlier directive's actual BC-IN clauses (including PTC REQ-099), proposed
BC-OUT/MAP profile successor, correction convention, gridding family/current
QC/retained-use records and frozen-clause crosswalks. It proposes a narrow
residual pass and rejoined-product input, while preserving ordinary routes.
The annex is byte-identical to r0.3 and remains outside Q01 approval.

## Current gates

`FRUIT-FEEDBACK-METHOD = unavailable_pending_separate_owner_approval`.
The route remains `unavailable_under_current_frozen_parent_permissions`.
The concrete amendments are proposed, not adopted. Exact MAP arithmetic remains
v0.1/r0.7.1; current PTC/CAL sources and the generic FRUIT core remain frozen.

Q01's scientific definition is locked; its required numerical policy slots
remain unbound. The next separate substantive review is Q02's exact boundary
amendments. Recovered centering/scaling, one-fit rules and learning direction
are not open again. Missing effective PTC and numerical
MAP support settings are Q03/Q04 bindings. Required inference evidence,
experimental populations, actual case/settings and execution are Q05/Q06.
Optional coherence, detector borrowing, detailed science priors, floors,
non-unity gain and alternative PTC estimators are not mandatory new gates.

This scientific-definition freeze authorizes no amendment adoption, implementation, replay,
injection, Unity work, qualification, production or policy recommendation.
If a bounded comparison is inconclusive or worse, preserve the evidence and
return to the simpler supported contract instead of expanding the search.

## Source control and archive

[Reference control](REFERENCE_CONTROL.md) and [source identities](SOURCE_IDENTITIES.json)
bind 44 exact copied inputs. The [24-entry proposed-reference inventory](AUTHOR_REFERENCE_INVENTORY.json)
is unchanged; its three program inputs are process-only. Additional owner and
manager sources do not become independent-author references.

[Verification](VERIFICATION.md), [manifest](PACKET_MANIFEST.md), and the
[frozen-definition archive](SCI-FRUIT-ordinary-map-method-definition-r0.4-frozen.tar.gz)
bind the exact delivery bytes. Archive inclusion does not approve the Q02 annex
or experiment proposals. Prior packets, frozen authorities, historical
products and both opaque untracked review archives are preserved.
