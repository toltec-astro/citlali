# SCI-FLT v0.1 Final Stage A Scope Direction

Scientific owner: Grant Wilson

Decision date: `2026-08-30`

Status: owner scope direction complete; exact repaired Stage A bytes still
require owner approval before Stage B

## Source Binding

This record durably transcribes two owner-supplied scope messages:

| Owner input | SHA-256 | Disposition |
| --- | --- | --- |
| `SCI-FLT v0.1 — FINAL STAGE A SCOPE REPAIR BEFORE IMPLEMENTATION-BLIND STAGE B AUTHORSHIP` | `92e0b01bd68cd18afe45ece9ed48949c22453f0aae140f43c10b2993814f5c03` | Accepted the recovery architecture and supplied the final repair requirements. |
| `My recommendation` resolving the three remaining questions | `59d85fa04e3c5f73732c656378c1d1fa0b38a183c9b64141ebd3cd2d95ddcfbd` | Approved strict linearity, the qualified fixed-low-pass-convolution subtype, and full-footprint-only convolution. |

The external attachment paths are not future author inputs. Their scientific
content is sanitized into
[`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md).

## Exact Scope Direction

1. `SCI-FLT` remains the recovery tranche. The first package is named
   `SCI-FLT-FIXED`; `SCI-FLT-DET` is rejected because it collides with the
   detector namespace.
2. `SCI-FLT-INF` is a non-authoritative holding tranche only. No combined
   inference-bearing Stage B contract is launched.
3. Base `SCI-FLT-FIXED v0.1` admits only one fixed linear same-grid map-domain
   transformation, `y = L_Theta m`, with fixed convolution as the concrete
   family. Affine transformation is deferred to a versioned successor.
4. A named fixed-low-pass-convolution subtype is admitted only when its exact
   transfer specification is complete. Low-pass is a qualified claim, not a
   second generic family.
5. Full-footprint-only convolution is the sole v0.1 edge/missing/non-finite
   method. Fixed extension, truncated-unrenormalized convolution, and support-
   renormalized convolution are deferred to separately named successors.
6. MAP observation, MAP coadd, and JINC observation are distinct parent roles.
   SCI-FLT does not coadd and assumes no filter/coadd commutation.
7. The complete finite operator, exact parent/output grids, units, response,
   support, validity, covariance state, NOI parity, lifecycle, causes,
   provenance, and failure behavior are typed explicitly.
8. FLT owns the exact transformation and its local facts. NOI owns ensemble
   generation and empirical uncertainty. Parent producers and downstream
   consumers retain their own facts and use policy; VAL binds and evaluates
   owner policy without authoring it.

## Required Return And Gate

The coordinator must return the revised Stage A artifacts, exact author-packet
manifest, and hashes for scientific-owner review. Stage B remains prohibited
until the owner approves those exact bytes and explicitly launches only
`SCI-FLT-FIXED` authorship.

This direction authorizes no implementation, algorithm change, conformity
claim, response/covariance fidelity claim, validation, calibration evidence,
performance finding, readiness, freeze, production action, Unity work, or
`SCI-FLT-INF` authorship.
