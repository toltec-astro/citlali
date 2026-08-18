# SCI-RTC v0.1 change log: r0.3 -> r0.4

Date: `2026-08-18`

## Formal corrections

- `SCI-RTC-EQ-029` now distinguishes finite attempt index `a` from accepted-plan
  index `k`, defines `x_eval^(0)=Apply(Pi_0,x^(0))`, and advances the accepted
  plan only for an accepted proposal.
- `SCI-RTC-EQ-011` now selects from the final pre-decimation stream
  `v_preD`, not the pre-filter stream `u`.
- Definitions, requirements, and predictions retain their stable IDs while
  adopting the corrected attempt/proposal/disposition semantics.

## Owner-selected scientific boundaries

- Every RTC role now retains raw, uncalibrated detector `Delta f/f`.
- Compatible `flxscale_q/flxscale_d` remains raw donor convention transfer;
  absolute `flxscale` and target atmosphere are applied downstream by SCI-CAL
  after the complete RTC bundle.
- Directly selected synthesized/replaced occurrences are universally excluded;
  noncenter transitive influence is preserved for each consumer's declared
  eligibility policy.

## Presentation and routing

- The rationale title now says Learn--Resolve--Apply.
- Section 1 adds the Beammap, Pointing, OOF, Science, and diagnostic-only
  RTC-plan matrix without inventing unresolved plan values.
- Crosswalk, owner ledger, decision log, verifier, engineering guidance, and
  package records are synchronized to r0.4.

No identifier was renumbered, no production number was selected, and no
implementation or validation claim was added.
