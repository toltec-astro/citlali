# SCI-FLT-FIXED v0.1 Edge, Missing, And Non-Finite Method Table

Status: owner-dispositioned Stage A author candidate awaiting exact-byte
approval

| Method | v0.1 disposition | Scientific identity |
| --- | --- | --- |
| Full-footprint-only convolution | **Authorized sole base method** | One fixed linear same-grid operator restricted to rows whose complete kernel footprint is admitted and finite |
| Fixed boundary extension | Deferred | Successor must name the boundary model, units, response, covariance, support, NOI treatment, and validity |
| Truncated unrenormalized convolution | Deferred | Position-dependent effective response/DC gain; separate successor method |
| Support-renormalized convolution | Deferred | Support-conditioned operator with separate response/covariance/NOI identity; separate successor method |

For exact support `K_Theta`, row `p` is scientifically available only when
every `p-r`, `r in K_Theta`:

- lies in the exact parent domain;
- is admitted for this exact FLT input use;
- has a finite available payload; and
- passes every required parent/support predicate.

Rows failing any condition are unavailable, not zero. Base v0.1 performs no
zero, constant, reflected, periodic, or other extension; truncation;
renormalization; wrap; clamp; mirror; edge completion; or implicit replacement
of missing/non-finite values.

The product publishes distinct states for numerical computability, complete
kernel support, FLT input admission, FLT-local output validity, unavailable-row
causes, and downstream eligibility. Parent-shaped storage does not promote an
unavailable edge row into the scientific vector.
