# Sanitized Composition Notes

Status: readability view of exact approved composition semantics; no new
scientific authority

The parent artifacts and their exact digests are bound in `SOURCE_MANIFEST.md`.
This view removes administrative review history and retains only the approved
semantic content needed to compose the frozen sources.

## Reference-First Handoffs

Pass the data needed by the next stage and reference the history rather than
retelling it. Each stage records only the new scientific facts it owns. A
downstream product keeps resolvable parent references and does not copy the
full contents of upstream products, sidecars, manifests, APTs, telescope
records, or response histories.

| Boundary | Data passed directly | References retained |
| --- | --- | --- |
| ALIGN to RTC | Aligned paired detector `x/r` occurrences and required validity | Native Tune/readout and ALIGN parent |
| RTC to AST/CAL | Conditioned RTC product, detector/sample identity, and required validity | RTC product/sidecar and ALIGN parent |
| Matched APT to AST/CAL | Exact detector-row binding and the fields each consumer uses | APT artifact identity, digest, and manifest |
| CAL to PTC | Calibrated ordinary signal, detector/sample identity, and required flags/validity | CAL and RTC products or sidecars |

CAL produces no calibrated `r` quantity. The paired RTC `r` remains reachable
through RTC parentage; PTC owns any auxiliary diagnostic or learning use. CAL
failure does not authorize an uncalibrated RTC-`x` fallback on the ordinary
PTC route.

Unrecoverable RTC-grid pointing is an observation-level hard stop: no CAL,
PTC, ordinary science, or external-ML handoff is produced. Missing or invalid
matched-APT integrity, detector-row binding, or finite nonzero selected
calibration factor blocks calibration at the affected scope. Typed unavailable
response or uncertainty remains claim-local unless an exact requested
operation requires it; it is never replaced with zero or an inferred default.

Original physical-acquisition and valid-original status remains attached to
original occurrences and is never recreated or increased by RTC, CAL, or
PTC. Downstream accounting operates on those occurrences through exact
support/lineage, not on RTC cadence or filter width. A later use-qualified
exposure quantity is owned by that use.

## Approved CAL And RTC Successor Authority

The readable WP-7 authority addendum governs the exact WVR interpolation,
unavailable-opacity, observation-wide quality-classifier, and RTC
logical-stream rules. The recovered atmosphere machine contract, 1,368-row
node table, and TolTECA-v1 passbands are admitted at their exact frozen
digests; no regenerated table or similarly named substitute is permitted.

Sample-local WVR opacity requires a producer-valid exact-time record or a
valid same-observation bracket. Unsupported opacity excludes only the affected
sample from calibrated support and never permits zero, unity, hold, header,
climatology, default, or another-observation substitution. The observation
quality classifier is a separate complete-window operational class and cannot
restore missing numerical support.

RTC terminal completion is logical-stream completion over the declared domain
plus final observation-level facts. Incremental output and consumption are
normal. Arbitrary chunk boundaries are engineering partitions, not scientific
support, and chunked execution must preserve the declared-domain result within
the operator's stated tolerance. Temporary RTC state is not thereby a
persistent product; materialization is optional and explicit. No unnamed
consumer or external-consumer acceptance is required.

## RTC Notation Parity

Use `N_filt` for the number of ordered temporal-filter stages:

```text
F_1, ..., F_N_filt
L^x_Omega = D_M,0 F_N_filt ... F_1 B^x_(omega,R)
```

Use `k_fin` for the final accepted-plan index:

```text
k_fin = k_(A+1) <= A
u_final = A_(Pi_k_fin)(u_0)
```

Existing `K` forms carrying response-kernel or response-status roles remain
response objects and are not counts. The role-sensitive old-to-new map is:

| Frozen form | Canonical composed form | Meaning |
| --- | --- | --- |
| `F_1, ..., F_K` | `F_1, ..., F_N_filt` | Ordered temporal-filter stages |
| `F_K ... F_1` | `F_N_filt ... F_1` | Ordered temporal-filter composition |
| accepted-plan `k in {0, ..., K}` | `k in {0, ..., k_fin}` | Accepted-plan index domain |
| accepted-plan `Pi_K` | `Pi_k_fin` | Final accepted plan |
| accepted-plan `K = k_(A+1)` | `k_fin = k_(A+1)` | Final accepted-plan identity |
| response-kernel/status `K` forms | unchanged | Response kernel, dependence, or typed status |

No substitution is authorized between `N_filt`, `k_fin`, and a response
kernel/status object.

## AST Notation Parity

The frozen generic sample index denotes one exact typed coordinate occurrence
`iota`, either an ALIGN-grid `(A,d,s,role)` occurrence or an RTC-grid
`(RTC,d,n,role)` occurrence.

| Frozen form | Canonical role |
| --- | --- |
| `B_i^AST` | Base pre-mapping facts for exact typed occurrence `iota` |
| `Pi_i^AST` | Exact AST direction, tangent, or pixel parent applicable to `iota` |
| `Pi_i^RTC` | RTC-grid parent `Pi_dn^RTC`, only for an RTC-grid occurrence |
| `G_pi` | `G_p,iota` notation only; no projection authorization |

Inside the admitted small-angle linear representation and its preregistered
adequacy domain, the representation-specific focal-plane expression and the
geometry-operator expression meet at the same realized detector displacement:

```text
B_ds f_d == xi_ds == G_gamma(g_d; t_s, E_s, state)
eta_ds^prod ~= c_s + xi_ds
```

The focal-plane vector `f_d` is not the complete selected geometry datum
`g_d`; the two descriptions are not independent physical quantities.

Role-factored atomic coordinate records, role-specific validity/cause,
uncertainty/Jacobian availability, layered parentage, and requested/effective/
resolved/realized provenance retain their frozen meanings. AST may materialize
an exact downstream-owned request, but this grants no projection, deposition,
weighting, or downstream policy authority.
