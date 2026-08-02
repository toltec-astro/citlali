# SCI-CAL-001 AM 12.2 successor adoption-study pre-execution clarifications

## Registration status

This additive clarification is frozen on 2026-08-01 before any adoption-study
AM run is launched and before any adoption-study numerical result is
inspected. It resolves three execution and gate ambiguities in
`AM12_SUCCESSOR_ADOPTION_STUDY_PROTOCOL.md`. It does not edit or reinterpret
the frozen protocol's scientific scope, change either candidate, or authorize
an operator, domain, application edit, repair, re-audit, or production use.

## C1 — midpoint coordinate quantization

For each requested arithmetic midpoint `tau_mid`, form the analytic
80-degree target transmission using the frozen modified-secant value

```text
X80 = 1.01538872688246729
T_analytic = exp(-tau_mid * X80).
```

Render `T_analytic` once with AM's C-locale T225 display format `%0.6e` under
the default round-to-nearest mode. Parse that displayed decimal literal as
`T_literal`. The H2O-scale solve matches `T_literal` exactly; it does not
attempt to match an unobservable sub-display value or the separately printed
AM tau column. Define the coordinate used for both direct truth and operator
evaluation as

```text
tau_achieved = -log(T_literal) / X80.
```

Every output row must retain `tau_mid`, `T_analytic`, the exact display
literal, `tau_achieved`, and the signed residual
`tau_achieved - tau_mid`. Evaluating at `tau_achieved` is mandatory. This is
deterministic AM-display quantization, not a discretionary coordinate shift.

For a displayed value with decimal exponent `e`, the spacing of `%0.6e`
values is `10**(e-6)` and its half step is

```text
h = 0.5 * 10**(e - 6).
```

The exact asymmetric propagation of the display half step around
`T_literal = L` is

```text
delta_tau_toward_lower_tau = log((L + h) / L) / X80
delta_tau_toward_upper_tau = log(L / (L - h)) / X80
acceptance_bound_tau = max(delta_tau_toward_lower_tau,
                           delta_tau_toward_upper_tau).
```

The requested-to-achieved residual must not exceed this propagated bound.
The bound is not a tau-column ULP, a fitted tolerance, or permission to move a
holdout after results are seen. All three literals have exponent `-1`, hence
`h = 5e-8` exactly in transmission. Decimal evaluation at 80-digit precision
with the frozen decimal inputs gives:

| Interval | `tau_mid` | `T_analytic` | `%0.6e` literal | `tau_achieved` | signed residual | lower-tau half-step | upper-tau half-step | acceptance bound |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| q0--q25 | `2.524370523370522005e-2` | `9.74693541581147455535088318156280150e-1` | `9.746935e-01` | `2.52437472479008390850963986147116523e-2` | `4.20141956190350963986147116523499926e-8` | `5.05207263490210769585533250450814916e-8` | `5.05207289406423222439544156868184469e-8` | `5.05207289406423222439544156868184469e-8` |
| q25--q50 | `6.941339152892524870e-2` | `9.31944910216666213788824485393985953e-1` | `9.319449e-01` | `6.94134023255157373905797836863371040e-2` | `1.07965904886905797836863371039666709e-8` | `5.28381275864426589515504053295119258e-8` | `5.28381304212738278252825721731728213e-8` | `5.28381304212738278252825721731728213e-8` |
| q50--q75 | `1.2332628558266549315e-1` | `8.82299139444837037445844988932535555e-1` | `8.822991e-01` | `1.23326329611985962328974442523710106e-1` | `4.40293204691789744425237101058733478e-8` | `5.58112588524756115790882535541648871e-8` | `5.58112620153066987858634307860799623e-8` | `5.58112620153066987858634307860799623e-8` |

The runner must independently recompute and assert these literals and values
before launching the midpoint matrix. A mismatch is a preflight failure.

## C2 — G2 positivity tolerance

G2 uses one internally consistent line-of-sight-optical-depth roundoff
envelope. Every evaluated quantity must be finite and strictly positive where
mathematically required. The accepted inequalities are exactly

```text
lambda >= -1e-12
0 < T_eff <= exp(1e-12)
C > 0
C >= exp(-1e-12)
```

where `lambda = -log(T_eff)` and `C = exp(lambda) = 1/T_eff`. Values in the
small interval allowed by the `1e-12` roundoff envelope must be preserved and
reported with their locations; they must not be clipped to zero, one, or a
boundary before any gate is evaluated. Non-finite values, `T_eff <= 0`, or
`C <= 0` fail strictly. Values beyond any of the stated upper or lower bounds
fail G2.

## C3 — G7 challenger status semantics

For each model-lane/operator candidate, the FTS challenger disposition is a
three-state result. Let `D_FTS,ECSV` be the maximum absolute fractional
difference between FTS and primary-ECSV **direct truth** extinction
corrections at identical physical rows.

- `pass`: every challenger G1--G6 representation gate passes and
  `D_FTS,ECSV <= 0.01` for every required band and spectral index.
- `owner_choice_required`: every challenger G1--G6 representation gate passes,
  but `D_FTS,ECSV > 0.01` for at least one required band or spectral index.
  This is a passband-choice sensitivity, not an interpolation failure; no
  adoption recommendation may proceed without an explicit owner passband
  choice.
- `fail`: any challenger G1--G6 representation gate fails. The associated
  model-lane/operator candidate is ineligible; a representation failure may
  not be downgraded to `owner_choice_required` or hidden by ECSV performance.

The machine decision record must preserve the individual challenger gate
results, `D_FTS,ECSV` maximum and location, and exactly one of these statuses.
No averaging across bands, alphas, profiles, opacity intervals, or elevations
changes the status.
