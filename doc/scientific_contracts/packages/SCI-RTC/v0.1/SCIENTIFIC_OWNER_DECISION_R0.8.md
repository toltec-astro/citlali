# SCI-RTC v0.1 scientific-owner Decision 9

Status: binding owner decision, supplied `2026-08-20`

## Resolved scientific boundary

1. A v0.1 level shift is an additive detector-baseline change on each stable
   plateau: `x(t) = s(t) + b_i`. It does not change gain or responsivity.
   Gain-changing or more general response-changing models remain successor
   scope unless observational evidence requires the owner to reopen this
   assumption.
2. An accepted event contains a stable pre-event plateau, a finite transition
   interval, and a stable post-event plateau. RTC determines or conservatively
   bounds the transition in physical time. A fixed sample count, scan fraction,
   scan leg, or other scan-dependent width is not an admissible definition;
   sample support is derived from the applicable timing vector.
3. Transition cells are unmodeled physical support. They are explicitly
   flagged, excluded from both plateau estimators, assigned to neither plateau,
   and are not made scientifically valid by offset correction. Downstream
   operators may propagate influence according to their support, but the
   physical transition interval remains distinct from that propagated
   influence.
4. A resolved RTC plan may estimate the additive difference between
   sufficiently stable pre/post plateaus and translate one plateau to the
   declared reference. The estimator, stable support, quality and uncertainty
   criteria, reference plateau, application direction, and failure behavior
   remain plan decisions. If support is inadequate, RTC invents no offset; the
   event remains a boundary and the plan explicitly retains or rejects each
   usable plateau.
5. Normal production output retains compact event/treatment state and useful
   population summaries. Full event-by-event fits and detailed diagnostics may
   remain verbose or diagnostic products.

## Required operation sequence

Candidate detection precedes physical transition localization or bounding,
stable-plateau identification, optional valid additive-offset estimation and
application, continued transition flagging, and downstream influence
propagation. Donor replacement remains post-segmentation and cannot cross the
boundary.

This decision establishes architecture, not a numerical estimator, threshold,
validation result, implementation-conformity claim, or production-readiness
claim.
