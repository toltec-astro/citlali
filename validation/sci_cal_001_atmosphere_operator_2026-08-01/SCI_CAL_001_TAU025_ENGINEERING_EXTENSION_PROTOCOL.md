# SCI-CAL-001 tau225 engineering-availability extension protocol

Protocol ID: `SCI-CAL-001-TAU025-ENGINEERING-EXTENSION-001`
Status: preparation only; **no AM execution authorized**
Authority: `CAL-ATM-D006`, coordination commit
`a58ada21a4f89479dd9a447e29dd01af566ecb1a`

This protocol prepares an independently held-out, direct-AM characterization
of a *future* continuous engineering correction over `0.15 < tau225 <= 0.25`.
It does not select, implement, adopt, or authorize an atmosphere operator or
an operational domain. It does not alter the separate EL25 confirmation
decision, the invalid EL25 execution, or the q0--q75 v2 result.

## Quality policy to be preserved

| Maximum eligible `tau225` in one coherent calibrated observation or declared processing segment | Quality state | Permitted claim |
| --- | --- | --- |
| `0 <= tau225 <= 0.15` | `science_qualification_target` | May later seek strict numerical and observational science-calibration gates. |
| `0.15 < tau225 <= 0.25` | `engineering_availability_target` | May later seek a versioned engineering correction; never a science-quality calibration claim. |
| `tau225 > 0.25`, non-finite `tau225`, or missing calibration identity | `outside_supported_calibration` | No silent extrapolation and no calibrated-science label. |

One declared operator and one quality state apply to the whole observation or
declared segment. A mixed-opacity unit is engineering-qualified when its
maximum eligible `tau225` exceeds .15, unless a later approved contract first
partitions it. This protocol authorizes neither an operator switch at .15 nor
per-sample quality tagging.

## Frozen direct-truth convention

Every eventual direct-truth run must bind all of the following before launch:

- AM 12.2 native executable SHA-256
  `78e721d45b08990069a2d67a5fb337446bcbfb728046940c0d473bea340205fb`;
- AM source payload aggregate SHA-256
  `0cd4ea9d48c3c6da2100a692af1dc24dce5b3c903ced2b07b7372e8e85182fe8`;
- copied AMC profile inventory from `copied_am_manifest.json`, with every
  selected `LMT_<season>_<percentile>.amc` filename and SHA-256 explicitly
  named in the later execution request. Generic q95 is excluded: it is not a
  profile, target, or extension anchor by assumption;
- AM argv convention: selected immutable AMC profile, `0 GHz`, `500 GHz`,
  `10 MHz`, integer zenith angle `90-EL` degrees, and only `Nscale
  troposphere h2o` through argv `%9`; every solved scale targets parsed AM
  225-GHz transmission at EL80 and preserves raw stdout/stderr plus sidecar;
- the exact TolTECA v1 ECSV byte set
  `toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`,
  with its index and a1100/a1400/a2000 member bindings; composite-trapezoid
  throughput integration, denominator normalization, no spectral
  extrapolation, and `S_nu proportional to nu^alpha` for `alpha={-1,0,2,4}`;
- line-of-sight optical depth `-log(T)`, full eligible-sample modified-secant
  airmass, and top-of-atmosphere `X_ref=0`.

The later request must name a distinct cache, execution context, source/profile
matrix, raw-output schema already used by the evidence package, and SHA-256
manifest. It must bind and enforce the existing approved `WARN-001` bounded
warning-bearing numerical-evidence policy verbatim: `WARN-001 admits AM status
1 only as explicitly warning-bearing numerical evidence with all 50,001 rows,
solely the preregistered unresolved-line warnings and canonical summary count
86, 87, or 88, and zero unknown warnings, cache mutation, or errors. Every
other nonzero status fails closed.` New warning classes or any deviation fail
closed. No new output format is proposed here.

## Proposed evidence design for owner approval

The proposed design is deliberately a request, not an execution authorization.
Direct AM truth is required at every selected profile/opacity/elevation tuple.
The profile matrix is **not** inferred from generic q products: the owner must
approve its exact subset of the existing copied AM 12.2 AMC inventory.

| Role | Proposed `tau225` nodes | Proposed elevation nodes (degrees) | Use |
| --- | --- | --- | --- |
| Construction anchors | `.15`, `.20`, `.25` | `25, 35, 45, 55, 65, 75, 80` | Fit only a single declared continuous engineering candidate. |
| Independent opacity holdout | `.1625`, `.175`, `.1875`, `.2125`, `.225`, `.2375` | all proposed held-out elevations | Never used to fit/tune the candidate. |
| Independent elevation holdout | all held-out opacity nodes | `29, 41, 53, 67, 79` | Never used to fit/tune the candidate. |
| Boundary evaluator-only diagnostic | `.15`, `nextafter(.15,-inf)`, `nextafter(.15,+inf)` | all construction and held-out elevations | No-AM algebraic/operator-evaluator continuity diagnostic after the candidate is defined; one candidate identity evaluates all three values, with no direct-AM target or scale search. |

The exact target-transmission literals, achieved `tau225` coordinates, AM scale
search traces, and anti-join against all fitting/tuning coordinates for the
decimal construction and independent held-out nodes must be frozen in a
subsequent execution request. The `nextafter(.15)` triplet is not a direct-AM
truth target and has no AM scale search. If a displayed decimal AM target
cannot represent an intended node within an explicitly approved coordinate
interval, the execution request fails closed rather than silently moving that
node.

## Required gates for a future extension study

1. **Direct truth and node identity.** Every direct-AM grid has the exact
   0--500 GHz/10 MHz frequency lattice, expected profile and executable
   digests, parsed 225-GHz target identity, the bound/enforced existing
   `WARN-001` admission record, 50,001 rows, and SHA-256 raw/sidecar binding.
2. **Physical domain.** Band-integrated transmission is finite and positive;
   LOS optical depth and correction are finite and non-negative/positive,
   respectively. Opacity monotonicity and elevation-direction monotonicity
   are measured and reported separately; a wrong-way feature is not hidden by
   averaging.
3. **Exact nodes and continuity.** The candidate reproduces every declared
   decimal construction node to its frozen numerical node tolerance. Only
   after that candidate is defined, the `.15` left/exact/right `nextafter`
   triplet is a no-AM algebraic/operator-evaluator diagnostic: all three
   evaluations use the same candidate identity and it creates neither a
   direct-AM truth target nor a scale search. Report the maximum relative
   correction discontinuity; an operator-selector handoff is a protocol
   failure.
4. **Independent held-out fidelity.** Evaluate the Cartesian held-out
   opacity/elevation grid, by profile, array, and alpha, only after the
   candidate is frozen. Report maximum, p95, RMS, signed extrema, and their
   locations for fractional extinction-correction error.
5. **Quality/provenance state.** The future observation/segment record must
   carry the quality state, maximum eligible `tau225`, identity/time coverage
   of contributing eligible samples, excluded/missing/non-finite counts,
   declared segment identifier, correction-operator/version digest, direct-AM
   evidence manifest digest, and passband identity. This is a compact policy
   for a future existing provenance record, not a new output format.

## Proposed engineering criterion — owner decision required

Proposed only: require a maximum absolute held-out fractional
extinction-correction error of **at most 5%** across the approved
engineering-domain profile/opacity/elevation/array/alpha grid. Report every
component distribution and do not pool away a maximum. This is a planning
screen for engineering availability, not the science `<=1%`
representation-fidelity criterion, not a 5--10% observational flux claim, and
not an adopted threshold. The owner must accept, revise, or reject it together
with the exact profile matrix, direct-AM node literals, execution cost/cache
request, and an execution request that binds and enforces the existing
`WARN-001` policy before any AM invocation.

## Explicit stop boundary

Until that approval, no AM run, cache creation, candidate fitting, operator
implementation, Citlali/TolTECA change, EL25 decision change, generic-q95
substitution, repair, re-audit, Unity action, production change, or
calibrated-science label is permitted.
