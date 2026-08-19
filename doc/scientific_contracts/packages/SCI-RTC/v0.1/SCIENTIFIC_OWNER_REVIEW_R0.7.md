# SCI-RTC v0.1 scientific-owner review r0.7

Status: binding surgical-correction request, supplied `2026-08-18`

Source SHA-256:
`01ec886e6d1dad89835463a1cee39dd0da067cf7532608698f90262cb41a9937`

## Review disposition

The supplied review accepts r0.6 at the science-team narrative and scientific-
architecture level.  It rejects another broad rewrite and identifies two
formal blockers plus five bounded clarifications.  It recommends one surgical
r0.7 pass followed by scientific-owner freeze disposition.

## Binding r0.7 corrections

1. RTC begins at the admitted aligned pair.  The native-to-aligned relation is
   upstream ALIGN lineage; the RTC-local operator and response are defined
   relative to the aligned parent.  An optional end-to-end response may compose
   admitted ALIGN response explicitly.
2. At fixed realized state, the conditioned-$x$ numerical Jacobian is
   `[Lx 0]`.  Cross-coordinate covariance enters joint selection,
   learned-parameter, model, diagnostic, and cross-term uncertainty rather than
   an unauthorized numerical $r$ branch.
3. Leakage ratios and angles retain coordinate units or scales, mapping
   revision, normalization, metric, Tune/validity domain, dimensionality, and
   comparison compatibility.  Cross-revision comparison is unavailable absent
   compatible conventions or an explicit transform.
4. A network level-shift event has one event time plus detector-specific timing
   offsets that are zero or constrained by a declared coherence tolerance.
5. Reset is ordinary at an accepted shift.  Carry is a separately authorized
   continuity exception with explicit response and residual-contamination
   criteria.
6. The selected-policy and scientific-purpose inventories include level-shift
   learning, segmentation, plateau assessment, and leakage diagnostics.
7. The science rationale explains why output-time target-atmosphere correction
   does not automatically invert attenuation mixed across temporal-filter
   support.

## Output and stopping rule

The final release candidates render the twelve-section science-team rationale
and the complete formal contract as separate PDFs generated from the same
authority.  The current hybrid is an authorship artifact, not the final
science-team output.  R0.7 shall not change section order, create a new
operation class, or expand the validation program.

This review authorizes no implementation inspection, numerical default,
conformity claim, validation result, readiness claim, or production claim.
Scientific authority is not frozen until the owner reviews the corrected
candidate and explicitly gives the freeze disposition.
