# SCI-CAL v0.2 Science/Engineering Consistency Report

Status: manager consistency check of the revised scientist-facing source
against the unchanged v0.1 engineering contract; not the required fresh blind
review and not a freeze disposition

Date: `2026-08-16`

## Result

No scientific conflict was introduced. The rewrite changes audience, order,
and explanatory emphasis. The engineering contract remains normative for the
exact assumptions, requirements, machine states, identity mechanics, and edge
predictions.

| Topic | Consistency result |
| --- | --- |
| Central equation | `d_cal = F_APT C_a d_in` is the scientist-facing form of engineering `z = x_xs - b`, `M = F_sel C_a Q`, `y = M z`, with the v0.1 unit factor `Q = 1`. The affine convention is retained in the formal appendix and explicitly assigned upstream. |
| Reference plane | Both views place `flxscale` and the target-atmosphere correction on the top-of-atmosphere reference plane. |
| Factor contents | Both apply selected `flxscale` and target atmosphere exactly once and exclude independent `responsivity`, `sens`, parent `flxscale`, embodied pointing correction, and opaque `fcf`. |
| Units | Both retain the point-source-peak `mJy/beam` boundary. V0.2 makes the already recorded passband/reference-spectrum limitations more prominent and converts them into owner decision Q05; it does not authorize another unit. |
| Atmosphere | Both use `ell = tau225 X`, interpolate the declared ordinate before any reciprocal, require exact content and support, forbid extrapolation/fallback, and withhold numeric output while the record is missing. |
| Opacity policy | Both retain `0 <= tau225 <= 0.15` as the structural science-policy range and no calibrated v0.1 output for `0.15 < tau225 <= 0.25`. V0.2 labels the unexplained rationale and segment policy as Q07 without changing them. |
| Uncertainty | `C_cal = A C_in A^T`, variance scaling by `M^2`, weight scaling by `M^-2`, unavailable-not-zero behavior, and correlated nuisance scopes are unchanged. |
| Response | Both distinguish originating Beammap response from realized map/kernel/filter response and require unit peak or explicit renormalization for literal point-source peak meaning. |
| Claim layers | Structural correctness, atmosphere representation fidelity, relative repeatability, and absolute recovery remain independent. No validation result is claimed. |

## Unresolved Authority Exposed by the Rewrite

The engineering contract legitimately treats some upstream facts as typed
inputs or assumptions. The science-team rationale needs their physical
meaning. Q01--Q09 record those gaps rather than presenting interface
completeness as physical explanation.

## Network 10 Check

The owner assessment asked whether `nw10` is nonexistent, reserved, or an
operating network. The current durable project sources available to this work
say that network IDs 7--10 map to `a1400`, including:

- `doc/SCIENTIFIC_CONVENTIONS.md`; and
- the owner-approved `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` author reference.

No stronger durable source was found that classifies `nw10` as nonexistent or
reserved. The v0.2 science narrative therefore omits a network roster because
it is not needed to explain calibration; the engineering convention remains
unchanged. If the current instrument roster differs, the owner should amend
the governing scientific-conventions authority rather than changing this
package alone.

## Review Boundary

This check was performed by the package manager while revising the document.
It does not replace the required fresh implementation-blind consistency review
after the owner approves the scientific substance.
