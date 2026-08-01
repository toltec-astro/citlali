# ADR 0009: Science-Map Bundle Admission And Validity

- **Status:** Accepted scientific contract; implementation pending independent
  re-audit
- **Recorded:** 2026-07-31
- **Decision owners:** Citlali scientific/product owner and engineering

## Context

The `SCI-MAP-001` audit found that ordinary mapmaking and observation
coaddition did not carry one reproducible scientific bundle. In particular,
coaddition admitted positional map slots while mutating accumulators, inherited
some identity from the last observation, and did not distinguish numerical
normalization support from science-policy support and final science validity.
The existing `weight_I` plane was also described as inverse variance without
the covariance and independence evidence required for that interpretation.

The project owner accepted the audit's `amend` verdict and selected the bounded
F009/F010 successor contract. The decision preserves the established ordinary
naive estimator and centered common-grid arithmetic. It does not authorize a
new estimator, reprojection, interpolation, correlated GLS, covariance
regularization, JINC mathematics, or a statistical-significance claim.

This ADR records the durable scientific and product decision. F009 and F010
remain `addressed_pending_reaudit`; accepting the contract is not acceptance of
an implementation, validation bundle, or production expansion.

## Decision

### Whole-bundle admission is atomic

An observation is admitted to a coadd as one immutable, ordered map bundle
before any coadd-owned state changes. The admitted identity includes:

- grouping and ordered map-slot identity, including array, network, detector
  or group, Stokes, and applicable frequency identity;
- signal unit and estimator identity;
- response/kernel identity and required-companion inventory;
- coordinate frame, projection, epoch, orientation, pixel scale, reference
  world coordinate, and reference pixel;
- observation and coadd shapes; and
- the versioned coefficient, contribution, support, validity, and non-finite
  policies.

Identity is constructed from authoritative full-precision values. The legacy
float-valued WCS representation is a one-way compatibility projection and is
not an equality authority.

For coadd shape `(R_c, C_c)` and observation shape `(R_o, C_o)`, the only
permitted geometric operation is centered integer common-grid embedding:

```text
R_c >= R_o, C_c >= C_o
(R_c - R_o) and (C_c - C_o) are even
delta_row = (R_c - R_o) / 2
delta_col = (C_c - C_o) / 2
```

The complete observation block must be in bounds, and its full-precision WCS
must identify the same world coordinate after that offset. Shape and the
corresponding reference-pixel offset are the only permitted WCS differences.
A mismatch in any slot, unit, response, frame, projection, scale, orientation,
source center, map order, shape parity, or policy rejects the complete
observation before numerical accumulators, grouping/WCS, membership,
observation numbers, exposure/count state, product inventory, or provenance is
mutated.

Centered placement is not signal centering. The signal-centering operator is
`L = I`: coaddition performs no implicit mean subtraction, null-mode removal,
or source recentering. General reprojection, interpolation, fractional shifts,
and best-effort WCS matching are outside this decision.

### The ordinary coefficient is nonprecision by default

For an admitted numerical contribution, the established operation order is
preserved:

```text
Q += u
N += u * signal
K += u * kernel
coadd_signal = N / Q where Q is finite and positive
```

The applicable `u` is the realized `weight_I` coefficient after observation
normalization and, when enabled, the existing optional global empirical
rescaling. It is a nonprecision gridding/normalization coefficient by default.
Its stored inverse-squared signal unit does not establish that it is a
marginal precision.

An explicitly invalid term is skipped before its numerical payload is
evaluated. A declared contribution requires finite signal, finite positive
`u`, and every declared numerical companion to be finite; violation is a
required pre-mutation failure. Signal, kernel, noise realizations, retained
exposure, and observation count use the same admitted membership and integer
embedding.

An inverse-variance interpretation requires `SCI-PTC-001` to establish the
applicable marginal-precision and independence/covariance conditions. No
coadd uncertainty, standardized statistical significance, correlated GLS, or
covariance-regularized result is implied or authorized while that evidence is
unavailable.

Sequential and OpenMP implementations use a declared deterministic or bounded
equivalence policy and may not perform unsynchronized shared-pixel mutation.
For fully compatible authoritative-valid controls, observation order, centered
offsets, and the established arithmetic operation order remain unchanged.
Within one scan, sequential and requested-parallel ordinary accumulation share
the same detector/sample-ordered primitive and are exact. Scan-farm commits are
mutex protected but may arrive in different orders; policy
`within-scan-exact-scan-farm-2gamma-n-sumabs-v1` bounds each binary64 plane
against the long-double sum of its per-scan planes by
`2 * gamma_n * sum(abs(scan_value))`, where
`gamma_n = n * epsilon / (1 - n * epsilon)`. Integer fact planes remain exact.

### Eight map facts remain distinct

The version-one ordinary-naive array-grouped product contract persists these
logical facts separately:

| Canonical identity | Dtype/unit | Meaning |
| --- | --- | --- |
| `geometric_hits_I` | `int64`, count | Finite in-bounds sample/detector projections before upstream eligibility and estimator selection |
| `contributing_hits_I` | `int64`, count | Terms admitted by the named estimator contribution predicate |
| `coadd_observation_count_I` | `int64`, count | Admitted observation maps contributing to a coadd pixel |
| `upstream_eligible_exposure_I` | `float64`, detector s | Projected detector-seconds eligible under the upstream validity contract before estimator and normalization retention |
| `retained_exposure_I` | `float64`, detector s | Detector-seconds retained after contribution and normalization-support decisions |
| `normalization_support_I` | `uint8`, dimensionless | Authorization for numerical division/population under the named normalization rule |
| `science_policy_support_I` | `uint8`, dimensionless | Result of the separate full-cut science-support policy |
| `science_valid_I` | `uint8`, dimensionless | The only authoritative raw science-validity mask |

`science_valid_I` is exactly the conjunction of normalization support,
science-policy support, finite normalized signal and every declared required
companion, and admitted bundle identity. Neither finite population nor any
one coefficient, exposure, or support plane can substitute for it.

`coadd_observation_count_I` is not applicable to observation maps. In this v1
contract, the complete F010 plane bundle is explicitly unavailable for JINC
and detector-grouped products. In particular, no ordinary positive-coefficient
predicate may be silently applied to JINC; `SCI-MAP-002` retains ownership of
any signed method-specific contribution predicate and successor availability.

`coverage_I` remains only a bitwise compatibility alias of
`retained_exposure_I`, with detector-seconds meaning. It is not wall-clock
integration time, precision, support, confidence, or validity.
`coverage_bool_I` remains only a deprecated, bitwise compatibility alias of
`science_policy_support_I`; it is never a validity authority.

### Threshold algorithms and realized provenance are separate

Both support rules select strictly positive finite coefficient values. For
`N` selected values sorted ascending, the zero-based order-statistic index is

```text
k = floor((floor(0.75 * N) + N) / 2)
```

The realized threshold is the coefficient at `k` multiplied by the applicable
cut, with threshold zero for empty input. Ordinary normalization support uses
`coverage_cut / 10`; science-policy support uses the full `coverage_cut`. Both
require an explicit finite-positive coefficient and use
`coefficient >= realized_threshold`. IEEE `!(w < threshold)` is not an accepted
substitute.

The lifecycle is one-way: requested state resolves to effective,
observation-resolved, and then realized state without back-populating an
earlier authority. Realized provenance preserves losslessly the
algorithm/version identities, coefficient product and lifecycle stage,
requested and realized cuts, thresholds, positive-value count and selected
order-statistic index, finite/positive/comparison conventions, counts for each
fact and state, required companions, admitted identity, coadd membership and
offsets, and exact raw-parent/product digest. Header values alone are not the
realized provenance authority.

Every downstream operator receives raw `science_valid_I` and raw-parent
identity as immutable inputs. It keeps those facts separate from its own
numerical computability, stencil/window support, response, covariance, and
output validity; producing a finite downstream value cannot promote a
raw-invalid pixel.

The coefficient-stage vocabulary is closed for this contract. Threshold
selection records either
`pre-observation-normalization-accumulated-coefficient` or
`pre-coadd-normalization-sum-of-admitted-observation-coefficients`. Published
products record exactly one of
`post-observation-normalization-no-empirical-rescale`,
`post-observation-normalization-global-empirical-rescale-applied`,
`post-coadd-normalization-no-empirical-rescale`, or
`post-coadd-normalization-global-empirical-rescale-applied`. An empirical
refresh changes only the applicable published stage and derived product facts;
it does not rewrite admitted observation evidence.

Before map filtering mutates map-domain planes, the validated raw F010 bundle
is frozen as an immutable snapshot. Filtered signal, coefficient, F010, and
compatibility-alias HDUs carry `RAWSTATE=immutable_input` and one identical,
lossless `RAWPDGST` value. A filtered empirical recalculation cannot mutate
that snapshot or its digest.

## Consequences

- Coadd admission uses a two-phase preflight/commit boundary and cannot leave
  a partially admitted observation when a later slot fails.
- Ordinary compatible inputs retain the established `Q`, `N`, and `K`
  arithmetic and observation order. Changes are confined to the explicitly
  approved invalid/non-finite, support, identity, product, and provenance
  repairs.
- JINC, detector-grouped, and other profiles outside the complete v1 F010
  availability keep their pre-successor legacy coadd arithmetic path. That
  path publishes explicit product-absence reasons and makes no F009/F010
  admission or product-contract claim.
- Product readers can distinguish hits, exposure, numerical support, policy
  support, and final validity without inferring one from another.
- Historical accepted products remain immutable evidence under their original
  contracts. They are not retroactively relabeled as carrying the successor
  F010 bundle.
- F009/F010 remain `addressed_pending_reaudit`. The fresh re-audit must assess
  the exact repair SHA. The human-run `SCI-MAP-001-UNITY-001` exact-SHA gate
  remains outstanding, and conclusions remain conditioned on
  `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and `SCI-VAL-001`.

## Rejected Alternatives

- **Treat positional map order and narrowed legacy WCS as sufficient
  identity:** distinct scientific bundles can alias or be mislabeled.
- **Mutate each slot as it passes:** a later failure can leave a scientifically
  invalid partial coadd.
- **Infer validity from `weight_I`, exposure, finiteness, or
  `coverage_bool_I`:** these are separate facts and do not establish the
  required conjunction.
- **Describe `weight_I` as inverse variance by convention:** dimensional units
  do not establish covariance or independence.
- **Add reprojection, GLS, or a JINC signed predicate inside this repair:** each
  changes scientific scope and requires its own approved contract and evidence.

## Supersession

A successor may broaden registration, precision/covariance interpretation,
JINC product availability, or downstream validity only through an explicit
scientific decision, versioned products/provenance, and affected-package
validation. It must not weaken exact whole-bundle admission or collapse raw
science validity into operator-local support without recording a superseding
decision.

## Evidence

- `SCI-MAP-001` scientific-contract audit at governing source
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- `SCI-MAP-001_BOUNDED_REPAIR_REAUDIT_HANDOFF_2026-07-31.md`
- [`../SCIENTIFIC_CONVENTIONS.md`](../SCIENTIFIC_CONVENTIONS.md)
- [`../../validation/product_contracts.json`](../../validation/product_contracts.json)
