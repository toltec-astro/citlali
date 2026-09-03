# SCI-NOI v0.1 r0.3 Scientific Rationale

Document identity: `SCI-NOI_SCIENTIFIC_RATIONALE v0.1/draft-r0.3`

Scientific owner: Grant Wilson

Status: implementation-blind Stage B draft; not frozen.

Normative authority: the six ordered modules in
`SCI-NOI_NORMATIVE_MODULE_BINDING v0.1/r0.3`, binding-file SHA-256
`b7756dbeeace639e4ec86878466fb2afcc20fe5b8d256b48b6747bff1527e03c`.
This rationale explains that authority and authors no independent normative
equation, requirement, assumption, definition, or prediction.

# 1. Scientific path

```text
Exact retained PTC parent
          |
          v
Active detector sign assignment
          |
          v
Frozen ordinary MAP operator
          |
          v
NOI-GEN realization ensemble
          |
          v
NOI-UNC conditional marginal second moment
          |
          v
NOI-STD independently realized MAP / conditional scale
```

Each arrow is an explicit identity and admission boundary, not automatic
realization. The sign enters once, only on the PTC signal occurrence. The
unsigned denominator, projection, WCS, support, coverage, response, validity,
and every other MAP gate remain fixed.

# 2. Scope and finite design

The owner selected one independent ensemble per exact observation and TolTEC
array. Base v0.1 contains no cross-observation randomization or randomized
coadd.

Within an observation/array, positive coefficient-mass detectors are grouped
by exact readout network. The accepted design draws independent symmetric
sign candidates and retains the first that satisfies every active network's
exact plan-bound tolerance. It fails closed under its finite cap, draws members
with replacement, forces no complement pair, and reports duplicates and
complement orbits separately.

Conditional on successful resolution, every admitted vector has the same base
probability. Summing over the common geometric sequence of earlier rejected
candidates gives the same conditional probability for each admitted vector.
The finite cap therefore changes the probability of resolution, not the
conditional accepted-vector law.

# 3. UNC and STD meaning

Complement symmetry gives active detectors known target-law sign mean zero.
Through the fixed linear operator, the member-map target mean is zero on exact
available rows. Initial UNC therefore computes the design-weighted marginal
second moment about that known center on the common all-member domain. Its
primary name is
`conditional_detector_sign_randomization_marginal_second_moment`. Its equality
to target-law marginal variance is a separate consequence of the known center,
not a license to call it map variance or noise variance.

The parent can contain source and deterministic residual structure, which can
remain visible in members and their squares. The result is not repeated
physical-noise variance, total uncertainty, covariance completeness,
precision, or calibrated significance.

STD divides an independently realized ordinary MAP signal by the square root
of the compatible conditional second moment. NOI cannot manufacture the MAP
numerator through an all-`+1` assignment. The unit is `1`; the claim is only
that the MAP signal is standardized by the stated conditional scale.

# 4. Route status

| Route/surface | r0.3 scientific status |
| --- | --- |
| Ordinary parent route | Identity approved; numerical route unavailable until exact PTC coefficient/QC, numerical `coverage_cut`, MAP admission, and every required parent fact exist. |
| Product scope | Owner-approved: one observation-level ensemble per exact observation and TolTEC array. |
| Detailed design | Owner-approved complete bounded first-accepted rejection design. |
| Numerical GEN | Unavailable until all parent, plan, population, response, and execution prerequisites are realized. |
| Initial UNC | Estimand approved; numerical product unavailable until exact complete admitted GEN and common-domain prerequisites exist. |
| Reciprocal | Not in the r0.3 ordinary base bundle; every reciprocal/precision/weight route unavailable pending exact owner disposition and binding. |
| Initial STD | Method approved; unavailable until independently governed `m_MAP`, compatible `Vhat_cond`, and successor profile binding exist. |
| Profiles | Four r0.18 profiles are immutable and registered; changed r0.3 evaluations require exact approved/bound successors. |
| External transforms | Interfaces remain bounded; base numerical FLT/Wiener/FRUIT routes unavailable pending exact external authority and generations. |

# 5. Claim ceiling

This draft defines scientific identities and prospective evidence duties. It
reports no implementation conformity, numerical validation, calibration,
physical-noise validity, covariance completeness, Gaussian significance,
achieved performance, readiness, freeze, production suitability, or
production authorization.
