# SCI-AST v0.1 Recovered Scope-Control Note

Status: retrospective consolidation record; not the missing original Stage A
author-input file and not new scientific authority

## Program Adherence And Prior-Work Recovery

This record is governed by the
[Citlali Scientific Contract Library Program](../../../README.md) and must be
read with [`PRIOR_WORK.md`](PRIOR_WORK.md). It points to the scope already
approved and preserved in the frozen r0.3 packet without attempting to
reconstruct missing Stage A prose.

## Frozen Scope Boundary

SCI-AST consumes the exact ALIGN occurrence/time/mapping bundle plus named
geometry, pointing-correction, frame, center, WCS, and product-plan parents. It
owns the geometric relation to detector sky direction, tangent coordinate,
continuous FITS pixel coordinate, optional nominal containing pixel, and the
stable RTC-grid coordinate role
`SCI-AST:rtc_output_grid_coordinates@1`. It does not reconstruct upstream
clocks, filter angular coordinates as detector signal, or independently select
MAP deposition `G_pi`.

The exact binding decisions, non-goals, open questions, and affected claims are
the entries in [`OWNER_DECISION_REGISTER.md`](OWNER_DECISION_REGISTER.md).
The exact shared upstream surface is
[`SCI-ALIGN_TO_SCI-AST_BOUNDARY.md`](SCI-ALIGN_TO_SCI-AST_BOUNDARY.md).

No implementation-derived scientific answer is admitted by this recovered
record.
