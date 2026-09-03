# SCI-JINC v0.1 — Notation, Equation, Requirement, And Prediction Change Map r0.2

Status: implementation-blind Stage B repair map; stable identifiers preserved,
new identifiers appended; final byte digests belong in the source manifest

Prepared: `2026-08-29`

Scientific owner: Grant Wilson

## Change-Control Rule

The targeted repair does not renumber any stable `r0.1` requirement or
prediction. Requirements `SCI-JINC-REQ-001`--`042` and predictions
`SCI-JINC-PRED-001`--`032` retain their identifiers. Clarified text under a
stable identifier preserves that proposition while making its previously
implicit source, discrete operator, WCS, numerical-unavailability, or scope
boundary exact.

Two new requirements and four new predictions are appended:

```text
SCI-JINC-REQ-043--044
SCI-JINC-PRED-033--036
```

No identifier is deleted, reused for an unrelated proposition, aliased, or
reordered. A later owner review may require a successor map, but numerical or
textual similarity alone cannot authorize renumbering.

## Notation Change Map

| Symbol or identity | `r0.1` state | `r0.2` disposition | Governing repair item |
| --- | --- | --- | --- |
| `j_{1,1}` | Second JINC zero represented by rounded decimal `3.831706`. | Defined exactly as `min{x>0:J_1(x)=0}`; the decimal is nonnormative only. | Exact Bessel root |
| `u_i=(x_i,y_i)` | Continuous AST pixel coordinate described generically. | Exact one-based continuous FITS axis-1/axis-2 coordinate in the target JINC WCS for the same processed sample `n`. | AST source closure; discrete operator |
| `c_i` | Single-valued rounded center required but formula open. | `floor(u_i+1/2)` componentwise; exact half ties go toward the positive coordinate axis. | Discrete operator |
| `phi_i` | Residual phase described generically. | `u_i-c_i` with exact domain `[-1/2,1/2)^2`. | Discrete operator |
| `n_sub` | Positive integer phase resolution. | Retained for every positive integer; no parity restriction or hidden default. | Discrete operator |
| `q_i` | Phase-bin identity not exact. | `floor[n_sub(phi_i+1/2)]` componentwise in `{0,...,n_sub-1}` under left-closed/right-open bins. | Discrete operator |
| `phi_hat_i` | Representative phase open. | Exact componentwise midpoint `-1/2+(q_i+1/2)/n_sub`; even-`n_sub` zero-phase asymmetry disclosed. | Discrete operator |
| `(q_x,q_y,d_x,d_y)` | Cache key not exact. | Normative logical cache key in FITS axis order; optional phase linearization `beta=q_x+n_sub q_y`, axis 1 fastest. | Phase-to-cache mapping |
| `M` | Target WCS named but scalar-size interpretation ambiguous. | Exact affine tangent-plane pixel matrix, including axis order, signs, rotation, and handedness. | WCS metric |
| `Delta` | Scalar pixel size present without full metric restriction. | Positive scale satisfying `M^T M=Delta^2 I`; skewed/anisotropic routes are incompatible. | WCS metric |
| `D` | Finite map domain implicit. | Exact one-based finite pixel-center rectangle `{1,...,N_x} x {1,...,N_y}` with no wrapping topology. | WCS/edge membership |
| `R_a` | Dual use of `r_max` described qualitatively. | Exact angular zero radius `s_a(r_max)_a`. | Cache extent |
| `h_a` | Integer square half-width implementation-resolved. | Exact `ceil[R_a/Delta]`. | Cache extent |
| `d` | Square offset generic. | Exact integer offset in `{-h_a,...,+h_a}^2`; no radial predicate. | Square support |
| `p_i,d` | Destination evaluation point implicit. | Exact destination pixel center `c_i+d`, retained only when center and destination pass their ordered finite-domain gates. | Point evaluation; edge membership |
| `P_NA` | Approximate adequacy principle only. | Exact owner-approved, immutable numerical-adequacy profile identity; currently unavailable. | Numerical-adequacy predicate |
| `E_p` | No complete certificate identity. | Exact realization-bound numerical-adequacy certificate; currently unavailable. | Numerical-adequacy predicate |
| `G_a` | Provenance wording could be read as prohibiting replay metadata. | Compact bundle-level generative record; not a numerical role or generalized provenance product. | Compact replayable provenance |

All pre-existing estimator, unit, array, parent, coefficient, time, and bundle
symbols retain their `r0.1` scientific meanings unless this table states an
exact clarification.

## Equation Change Map

| Canonical equation or group | Stable identity treatment | `r0.2` change | Scientific consequence |
| --- | --- | --- | --- |
| Peak-normalized JINC primitive | Existing equation retained. | No algebraic change. | Removable limit remains exactly one. |
| Analytic kernel | Existing equation retained. | Replace the rounded defining constant with exact `j_{1,1}`. | The second factor's zero at `r'=r_max` is exact. |
| `eq:rounded-center` and `eq:residual-phase` | New canonical equations. | Define `c_i=floor(u_i+1/2)` and `phi_i=u_i-c_i`. | Center, tie direction, and residual domain are reproducible. |
| `eq:phase-bin` and `eq:phase-index` | New canonical equations. | Define half-open phase-bin membership and the exact index mapping. | Bin edges and indices are exact for every positive `n_sub`. |
| `eq:phase-representative` | New canonical equation. | Defines midpoint representatives, including even-`n_sub` asymmetry. | Cache-phase selection is exact and testable. |
| `eq:square-wcs-metric` | New canonical equation. | Defines `M^T M=Delta^2 I`, equivalently `M=Delta R` with `R in O(2)`. | Scalar pixel scale cannot hide skew or anisotropy. |
| `eq:cache-half-width` and `eq:square-offsets` | New canonical equations. | Define `h_a=ceil[s_a(r_max)_a/Delta]` and the full square. | Every square corner is evaluated; no circular cutoff survives. |
| `eq:discrete-radius` | New canonical equation. | Defines `r=||M(d-phi_hat)||_2` at destination pixel center `c_i+d`. | Phase, WCS metric, and evaluation point determine one exact coefficient. |
| Signed estimator equations | Existing equations retained. | No change to `N_p`, `C_p`, `Q_p`, `m_p`, or `A_pi`. | Signed estimator core and denominator remain stable. |
| Conditioning equations | Existing `D_p` and `rho_p` equations retained. | Remove the approximate `10^-3` phrase as a pass/fail rule; separate exact algebraic support from unavailable numerical certification. | `rho_p` remains descriptive and no universal cutoff is invented. |
| Conditional response/covariance equations | Existing conditional relations retained. | Explicitly classified as future mathematics under ODQ-107, not base output roles. | No response/covariance/formal-weight product is created. |
| Compact replay record | New exact field contract, not a numerical equation. | Binds generative inputs and lifecycle at bundle granularity. | Replay is possible without dense provenance payloads. |

## Requirement Change Map

| Requirement IDs | Identifier status | `r0.2` clarification or addition | Targeted directive coverage |
| --- | --- | --- | --- |
| `001`--`006` | Stable, retained | Source/ownership and PTC occurrence/coefficient semantics retained; current exact PTC r0.3 and VAL bindings are made explicit where referenced. | PTC/VAL source closure |
| `007` | Stable, clarified | Parameter route now requires the exact compatible square-pixel WCS metric and explicit positive integer `n_sub`; no numerical values/defaults are introduced. | WCS metric; typed unavailability |
| `008`--`009` | Stable, clarified | Exact `j_{1,1}` replaces the rounded decimal as defining authority. | Bessel root |
| `010`--`011` | Stable, clarified | The exact center, tie, phase interval, bin, representative, key, and destination-point equations replace arbitrary discrete choices. Bessel evaluation and accumulation remain engineering choices subject to future certification. | Discrete operator |
| `012` | Stable, clarified | `h_a=ceil[s_a(r_max)_a/Delta]` and the full integer square are exact; no radial cutoff. | Cache extent and membership |
| `013`--`016` | Stable, clarified | Ordered center-domain gate, destination crop, point coefficient, and common membership identity use the exact discrete/WCS definitions. | Edge membership |
| `017`--`020` | Stable, retained/clarified | Signed accumulator algebra and finite-negative/exact-zero denominator semantics are unchanged; algebraic support is separated from numerical certification. | Algebraic support |
| `021`--`023` | Stable, repaired | The approximate `10^-3` language is removed as a validity predicate. Exact profile plus compatible certificate is required; both are unavailable, so numerical and near-cancellation support are unavailable. | Numerical adequacy |
| `024`--`027` | Stable, retained | Unit/scaling, conditional formal mathematics, and coefficient-squared time retain their meanings. | No scope expansion |
| `028`--`034` | Stable, retained | Exact five-role bundle, atomic failure, per-array observation grouping, destination, and no-coadd semantics remain unchanged. | ODQ-107 scope parity |
| `035`--`037` | Stable, clarified | Registered profile means upstream sample admission only; exact PTC r0.3, AST r0.2, same-`n` join, and SCI-VAL source/profile records are named. | Boundary/profile closure |
| `038`--`040` | Stable, clarified | Response/covariance products remain deferred and no generalized provenance product is added; producer facts and compact bundle replay metadata are explicitly preserved. | Output scope; compact replay |
| `041` | Stable, retained | TolTEC numerical parameter values/defaults remain unauthorized. | Numerical-route unavailability |
| `042` | Stable, clarified | Future assessment must bind exact numerical profile/certificate and keep every claim layer separate. | Numerical certification and nonclaims |
| `043` | New, appended | Requires exact source/version/digest closure for PTC r0.3, AST r0.2, the registered JINC profile, paired SCI-VAL registers, compatibility, and no inferred alias/current-version substitution. | Source-close PTC/AST/VAL interfaces |
| `044` | New, appended | Requires one compact atomic generative record sufficient to replay input population, parents, admission, coefficient/parameter/WCS/operator state, membership, conditioning/certification, destination, lifecycle, required-role publication, failure, and completion. | Compact replayable provenance |

## Prediction Change Map

| Prediction IDs | Identifier status | `r0.2` clarification or addition | Governing requirements |
| --- | --- | --- | --- |
| `001`--`003` | Stable, retained | Signed-estimator identity and the required-domain rules retain their scientific results. | `016`--`020`, `026` |
| `004`--`005` | Stable, retained | Positive coefficient-rescaling predictions retain their scientific results. | `024`--`025` |
| `006`--`008` | Stable, clarified | Exact-root and negative-lobe predictions retain their scientific results; analytic-zero evaluation uses exact `j_{1,1}` where applicable. | `008`--`009`, `015`, `017`, `026` |
| `009`--`012` | Stable, repaired | Exact cancellation remains unsupported, and no near-cancellation case is promoted by the approximate `10^-3` phrase. Without an exact compatible profile and certificate, numerical support is unavailable even if finite arithmetic returns nonzero `C_p`. | `019`--`023` |
| `013` | Stable, retained | Non-finite required inputs remain fail-closed under the exact local/whole-bundle scope. | `003`, `006`--`007`, `015`, `019`, `026`, `029` |
| `014`--`017` | Stable, clarified | Exact half-width, full-square corners, one-point phase, half ties, all bin edges, `n_sub=1`, and odd/even phase behavior use the new equations. Even-`n_sub` zero-phase asymmetry is disclosed and tested. | `007`, `009`--`012` |
| `018`--`020` | Stable, clarified | Exact center and destination equations govern outside-center rejection and in-map crop under the allowed WCS class. | `013`--`014`, `016`--`020` |
| `021` | Stable, retained | Coefficient-squared exposure-time semantics retain their meaning. | `026`--`027` |
| `022`--`023` | Stable, clarified | Missing source authority and unavailable TolTEC numerical parameters remain typed unavailable. | `004`--`007`, `041` |
| `024`--`027` | Stable, clarified | Grouping, destination, association, and atomic required-role publication retain their meanings with compact replay metadata. | `030`--`034`, `044` |
| `028` | Stable, clarified | Fixed-bundle source closure uses the exact PTC, AST, profile, and register identities. | `002`, `035`--`037`, `043` |
| `029`--`030` | Stable, clarified | Deferred-companion and compact replay predictions preserve the five-role scope. | `028`--`029`, `038`--`040`, `044` |
| `031` | Stable, retained | Dimensionless formal-factor semantics retain their meaning. | `024`--`025`, `038` |
| `032` | Stable, retained | No MAP inheritance is authorized by the JINC contract. | `001`, `035`--`038` |
| `033` | New, appended | A stale, mismatched, renamed, aliased, or silently current PTC/AST/profile/registry source cannot satisfy the exact source binding. | `002`, `035`, `037`, `043` |
| `034` | New, appended | A passing upstream admission profile alone never establishes rounded-center admission, pixel membership, finite coefficient, cancellation/numerical support, or bundle validity. | `013`, `019`--`023`, `035`--`036` |
| `035` | New, appended | A skewed or anisotropic target pixel matrix is typed incompatible; no scalar scale approximation or silent square-cache route is permitted. | `007`, `010`--`014` |
| `036` | New, appended | The compact generative record exactly replays the source population, parents, operator and lifecycle; a missing/conflicting required join prevents realized success without creating a sixth role or dense provenance payload. | `031`, `033`, `040`, `044` |

## View And Crosswalk Consequence

Both scientist-facing and engineering-facing views shall include the same six
canonical modules once and in the same order. The final `CROSSWALK.md` shall
contain exactly one row for each `SCI-JINC-REQ-001`--`044`. The shared
prediction trace shall cover `SCI-JINC-PRED-001`--`036` without gaps,
duplicates, or identifier reuse.

Final source and PDF hashes, include-order evidence, crosswalk counts, and
byte-parity checks are intentionally not asserted here. They belong in the
final source manifest and rationale/ECS parity report after all `r0.2` bytes
are settled.

## Claim Boundary

This map records draft textual and identifier changes. It does not establish
implementation conformity, representation fidelity, numerical certification,
validation, response/covariance fidelity, achieved performance, readiness,
production authorization, or scientific-owner approval of `r0.2`.
