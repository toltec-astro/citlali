# SCI-POINT ODQ-008 Scientific-Owner Approval

Record identity: `SCI-POINT-ODQ-008-APPROVAL-2026-09-02`

Scientific owner: Grant Wilson

Decision date: `2026-09-02`

Status: approved Stage A scientific direction

## Approved Decision

Fitted amplitude, the two fitted widths, and fitted orientation angle are
required components of every numerical
`POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1` result. They are not incidental
implementation diagnostics. Together with the fitted centroid and honest fit
state, they are authorized quality-control metrics for telescope performance
and observing conditions.

The fitted centroid displacement remains the primary SCI-POINT pointing
measurement. The additional fitted quantities retain these bounded meanings:

- amplitude is the fitted source amplitude in the exact parent product's
  declared unit, calibration, normalization, and effective response; it is not
  automatically a universal source flux;
- fitted widths and angle describe effective source shape under the exact
  parent map, filtering, support, and fit model; they are not automatically an
  intrinsic telescope beam, detector PSF, or SCI-BEAM result; and
- any constraint-limited, support-limited, uncertainty-limited, or otherwise
  qualified value carries that state into quality-control use.

CAL or TolProj may consume the amplitude under their own exact authorization
and provenance. Quality-control use may compare or trend these quantities, but
the metrics alone do not establish a unique physical cause for a deviation or
transfer SCI-BEAM authority into POINT. Exact quality-control admission,
threshold, reference, aggregation, and causal-interpretation policy must be
owned and identified by the named use.

## Non-Effects

This approval does not define the ODQ-009 VAL profile mechanics, a particular
quality-control threshold, a telescope/atmosphere causal model, absolute flux
accuracy, or intrinsic beam inference. It does not approve the complete Stage
A packet, authorize Stage B, change an algorithm, or establish implementation
conformity, validation, achieved performance, readiness, or production state.
