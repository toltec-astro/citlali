# SCI-NOI v0.1 — Scientific-Owner ODQ-102B Approval

Decision identity: `SCI-NOI-ODQ-102B`

Scientific owner: Grant Wilson

Decision date: `2026-08-29`

Status: approved ordinary-method coherence unit; sign law and finite design
remain open

## Exact Owner Decision

> For the ordinary fixed-state NOI method, once a detector receives its
> realization assignment, that assignment applies to all of that detector's
> admitted samples throughout the observation.

## Sanitized Disposition

For `NOI-GEN/PTC-TO-FROZEN-MAP-CONDITIONAL-SIGN@1`, one coherence unit is one
stable realized detector/channel within one exact observation. For realization
member `b`, the NOI assignment modifier for that unit is constant over every
PTC occurrence that:

- belongs to the same exact observation;
- carries that stable realized detector/channel identity; and
- is admitted to the ordinary route at the PTC-to-MAP boundary.

Changing sample number, scan, subscan, chunk, time, traversal order, container
position, worker, or MAP accumulation order does not change the assignment for
that detector within the observation. The same detector identity in a
different observation belongs to a different coherence unit and receives its
assignment under that observation's exact design domain.

The stable unit identity and ordering use canonical observation UID followed by
stable realized detector/channel UID. Numerical index, array position, column,
or encounter order cannot establish or order the scientific identity.

## Non-Implications

This decision selects the coherence partition only. It does not select:

- assignment probabilities or deterministic design weights;
- independence or cross-observation coupling;
- balance, complement pairing, replacement, or duplicate treatment;
- seed/key algorithm or requested/resolved member count;
- design rank, source cancellation, UNC estimator, or covariance meaning; or
- any implementation representation.

Those remain under ODQ-102C and later decisions. The ordinary route remains
numerically unavailable at its frozen PTC coefficient and numerical
`coverage_cut` gates and until an exact sign law/finite design is approved.
