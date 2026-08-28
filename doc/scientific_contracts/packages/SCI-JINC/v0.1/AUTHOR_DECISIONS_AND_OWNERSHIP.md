# SCI-JINC v0.1 — Sanitized Inherited Decisions And Ownership

Status: final Stage A repair candidate; awaiting scientific-owner approval

Scientific owner: Grant Wilson

Prepared: `2026-08-28`

This author-facing table carries the exact scientific consequences of the
eight approved historical JINC decisions without admitting their raw recovery
conversation, implementation findings, repair records, tests, or validation.
The exact source objects and SHA-256 values are recorded below. A compatible
successor preserves every listed consequence; a change requires explicit
scientific-owner authority and a versioned successor.

`SCI-JINC-ODQ-107` is the controlling base-v0.1 product-scope disposition.
The inherited rows preserve scientific semantics, but they do not create a
general availability/provenance framework or require every historically
described companion as a base product. Base v0.1 publishes only distinct
`N_p`, `C_p`, `Q_p`, derived `m_p` with local support/validity state, and
`jinc_coefficient_squared_time`. Response, covariance/formal-weight,
standalone support/availability, diagnostics and generalized provenance
products are outside or deferred pending a concrete scientific use.

## Inherited Decision Table

| Stable ID | Exact approved rule | Mathematical and product consequence | Affected objects | Unavailable behavior | Exact predecessor clause superseded | Compatibility / supersession rule |
| --- | --- | --- | --- | --- | --- | --- |
| `SCI-MAP-002-D003-SUPPORT-001` | `r_max` fixes both the first zero of the second JINC factor and the half-width of a fully populated square deposition cache. Square corners beyond radial `r_max` remain part of the response. | Every admitted square-cache pixel receives an analytic point evaluation, subject only to finite-map cropping; no radial membership predicate is applied. | Support, response, covariance, parameter and provenance identities. | A strict circular cutoff, corner suppression, or description of `r_max` as a radial support maximum is unavailable. | Independent core Equations `eq:continuous-support` and `eq:support`, including the `<`/`<=` radial branch and radial geometric predicate. | Compatible only with square support and the dual-use `r_max` identity. A distinct radial support parameter or changed footprint is a successor. |
| `SCI-MAP-002-D003-SUBPIXEL-001` | `subpixel_n` selects a phase-quantized, point-evaluated kernel matrix after center rounding and residual-phase binning. Increasing it refines the realized point-sampling response. | The coefficient is evaluated at the selected point phase; it is not a pixel-area average. Exact rounding, bin edges, representatives, and effective `subpixel_n >= 1` remain required contract facts. | Coefficient, response, square-cache placement, covariance, parameter and provenance identities. | Pixel-area integration or quadrature toward the independent core's pixel-average target is unavailable. | Independent core Equation `eq:pixel-average` and its pixel-integrated convergence target. | Compatible only with point-phase evaluation. Changed phase semantics or an area-integrated operator is a successor. |
| `SCI-MAP-002-D003-CONDITIONING-001` | Require finite contributors, finite `C_p` and `Q_p`, `Q_p>0`, `C_p!=0`, and the dimensionless ratio `rho_p=|C_p|/sum_i|omega_i kappa_ip|`. Reject only when `rho_p` is below a documented floating-point error bound derived from the realized summation method and contributor count. | Finite negative `C_p` is admissible. Exact or numerically unresolved cancellation is typed unavailable; resolved cancellation remains admissible with its small conditional formal weight. Stable scaling may not change the estimator. | Normalization, formal support, conditional weight/covariance, numerical policy and provenance. | Unit-bearing `C` or `Q` floors, a positive-denominator assumption, zero-sky substitution, and silent cancellation clipping are unavailable. | Independent core's unspecified configured coverage/conditioning cut and any absolute `|C|` or `Q` floor. | Compatible only when invariant under signal-unit and common coefficient rescaling and when the realized bound/policy are identified. A changed statistic or unbound threshold is a successor or unavailable. |
| `SCI-MAP-002-D003-ADMISSION-001` | Stable array identity and finite strictly positive kernel shape parameters, `r_max`, pixel size, and array scale are required. Finite negative analytic coefficients are valid. Any non-finite evaluated coefficient fails the selected required JINC product. | Parameter resolution is fail-closed by array; signed lobes retain sign in `N_p` and `C_p` and enter `Q_p` quadratically. | Parameter, coefficient, array/group, support, accumulation, failure and provenance objects. | Positional array fallback, clipping or zeroing negative lobes, silent omission of non-finite coefficients, and detector-local recovery are unavailable. | Independent core's open cardinality/order/domain mapping and any permissive non-finite handling left to later inspection. | Compatible only with exact stable-array selection and fail-closed parameter/coefficient admission. Changed parameter domain or recovery behavior is a successor. |
| `SCI-MAP-002-D003-MASK-001` | Formal JINC support requires finalized finite signal, finite strictly positive conditional formal weight, and passage of every required JINC admission and cancellation check. Empirical policy may narrow but never promote it. | Formal support, empirical downstream eligibility, temporal accounting, and upstream admission remain distinct typed propositions. | Formal-support state, signal, conditional weight, response/covariance availability and downstream policy joins. | Treating time, hits, coefficient availability, finiteness alone, or an empirical mask as the formal-support authority is unavailable. | Independent core's `coverage_bool_I` representation and four-level support wording where it permits a representation name to control abstract identity. | Compatible when the abstract formal-support meaning and no-promotion rule are preserved; representation names may change without changing science. |
| `SCI-MAP-002-D003-COVERAGE-001` | The normative accounting product is `jinc_coefficient_squared_time`: `T^(kappa^2)_p=sum_i I_ip kappa_ip^2/f_s,i`, in seconds. | It measures method-specific coefficient-squared temporal accounting. Analytic zeros contribute zero. It is joined to coefficient/phase and sample-frequency provenance. | Time-accounting, coefficient, support, parameter/phase and provenance objects. | Physical acquired exposure, valid-original exposure, complete temporal support, hits, normalized influence, white-noise-equivalent time, precision, significance, or validity interpretations are unavailable. | Independent core `T_p=sum Delta t_s` geometric-exposure accumulator and its zero-coefficient exposure convention. | Compatible only with the squared dimensionless analytic kernel coefficient `kappa_ip`. Any physical-exposure product remains separately owned and traced. |
| `SCI-MAP-002-D003-KERNEL-001` | The JINC response companion is the realized processing-filtered source-template response transformed through JINC exactly once and normalized by `C_p`: `K_p/C_p=(sum_i I_ip omega_i kappa_ip h_i^processed)/C_p`. | The response uses the exact signal membership, coefficient, point phase, square support, edge crop, normalization and processing/operator parents. | Fixed-state response, signal membership, support, conditioning, WCS, parameter and provenance objects. | A bare analytic JINC, measured PSF, generic beam, hidden subset, double application of upstream response, or downstream renormalization by inference is unavailable. | Independent core Equation `eq:realized-response` only to the extent it left source-template and processing identity generic; the signed normalized structure is retained. | Compatible only with exact response family, parent, domain and single-application identity. Re-resolved or whole-chain response is separately typed. |
| `SCI-MAP-002-D003-PROVENANCE-001` | Every coherent observation bundle has one compact, atomic, one-way requested/effective/resolved/realized record. Required publication or join failure prevents realized success. | The record binds exact array, plan, WCS, parameter, support/phase/conditioning, admission summaries, response/covariance/time/support roles, parents, destinations and digests without per-sample or per-pixel payload. | Lifecycle, every product-role identity, joins, failure and publication state. | Backward mutation, typed/legacy bidirectional synchronization, stale partial reuse, incomplete success, and per-sample/pixel provenance payloads are unavailable. | Independent core's requested/effective/observation-resolved/realized table where it did not yet require the approved atomic joins and failure suppression. | Compatible only with one-way atomic lifecycle and complete required joins. Storage representation may vary; a changed scientific lifecycle requires a successor. |

## Additional Retained Scientific Rules

These rules are not additional D003 decisions, but they constrain the author
task:

- The owner-accepted estimator keeps distinct `N_p`, `C_p`, and `Q_p`, uses
  `m_p=N_p/C_p`, and permits `C_p^2/Q_p` as a conditional formal weight only
  under the exact upstream coefficient and covariance assumptions.
- ODQ-107 requires `N_p`, `C_p`, `Q_p`, `m_p` with its local support/validity
  state, and `jinc_coefficient_squared_time` as the fixed closed bundle. The
  conditional formal-weight interpretation does not create a base-v0.1
  formal-weight product.
- SCI-JINC is a sibling alternative observation mapmaker. It starts after the
  SCI-MAP freeze, consumes exact PTC and AST parents, consumes no SCI-MAP
  product, and inherits no ordinary MAP rule by analogy.
- One complete destination identity is resolved before mutation; ambiguity or
  failure prevents atomic publication. Workers, threads, processes,
  containers, and filenames are not scientific product identity.

## Exact Source Bindings

The reusable independent core is the Git object
`fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`,
SHA-256
`2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24`.

The eight decisions are the exact files at
`8c581bfb26f01b187f4f1e0565f4457bcc25f099:doc/audits/packages/` with these
digests:

| Decision suffix | SHA-256 |
| --- | --- |
| `SCI-MAP-002_SUPPORT_OWNER_DECISION_2026-08-03.md` | `8c6b53476e70c40f5ff98fde6e279c44d58406cc81ffaa86d34d6cf6cd93645a` |
| `SCI-MAP-002_SUBPIXEL_OWNER_DECISION_2026-08-03.md` | `053cedd2a644850ea65db616661cc95a1d872f60945074d40e81d07397716aaa` |
| `SCI-MAP-002_CONDITIONING_OWNER_DECISION_2026-08-03.md` | `8f74aacdd9df5a9119c08d1e627ae7859ff1d87d7a8ecdbfac0d76acaa0eddcc` |
| `SCI-MAP-002_PARAMETER_ADMISSION_OWNER_DECISION_2026-08-03.md` | `811232b2e13644a1b8794a87ef6e62b95de0653d08aa26802097823bd55f8cab` |
| `SCI-MAP-002_VALIDITY_MASK_OWNER_DECISION_2026-08-03.md` | `4d6d236e3b630f6608eb6dc36f8ee9041f7b623ea5f4adbc0f5e0c946a0b9f08` |
| `SCI-MAP-002_COVERAGE_OWNER_DECISION_2026-08-03.md` | `5413039d5deef53647187ae226b06c259ee750120f4444ad3bd6870fe8d8cb58` |
| `SCI-MAP-002_KERNEL_IDENTITY_OWNER_DECISION_2026-08-03.md` | `2ff6b03259e75d2fa96f4fe8c888c7db82ec7cae0c969617f17233aa15be2347` |
| `SCI-MAP-002_PROVENANCE_OWNER_DECISION_2026-08-03.md` | `bb3084e3905ae283266f066dd8406d5d48a787e5af6825b4a9ddf60edac5dd1f` |

Raw decision files remain excluded from the implementation-blind author
packet. This sanitized table is the proposed author input.
