# SCI-MAP-002 JINC subpixel-response owner decision — 2026-08-03

Status: owner approved contract clarification; no implementation work
authorized

Package: `SCI-MAP-002`

Decision ID: `SCI-MAP-002-D003-SUBPIXEL-001`

Authority: project owner

## Decision

The realized JINC subpixel response is the approved scientific convention.
`subpixel_n` selects one of the precomputed phase-quantized, point-evaluated
kernel matrices after the sample center is rounded and its residual phase is
binned. Increasing `subpixel_n` refines this phase representation toward the
realized point-sampling response; it is not a pixel-area integration scheme.

This decision preserves existing response and avoids an unapproved change to
the mapmaking operator or a more expensive pixel-integration calculation. A
future implementation/provenance repair must state the point-phase target,
rounding and bin-edge convention, `subpixel_n >= 1` effective behavior, and a
bounded phase/convergence validation. It must not substitute a pixel-area
average, tune JINC parameters, or begin a numerical-optimization campaign
under this authority.

The remaining SCI-MAP-002-D003 decisions—unit-invariant conditioning,
parameter/coefficient admission, coverage/mask/kernel identity, and realized
provenance—remain open. No code change, Unity evidence, repair, re-audit, or
production-status change is authorized.
