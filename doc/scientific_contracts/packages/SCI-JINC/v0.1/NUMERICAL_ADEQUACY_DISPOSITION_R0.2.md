# SCI-JINC v0.1 — Numerical-Adequacy Disposition r0.2

Status: implementation-blind Stage B author-draft disposition; algebraic
support specified, numerical-realization adequacy typed unavailable

Prepared: `2026-08-29`

Scientific owner: Grant Wilson

## Repair Finding

The admitted Q002 successor Scope Brief records the ODQ-109 principle that
numerical error be materially smaller than an approximately `10^-3` relative
instrument-fidelity scale, while supplying no sharper universal constant,
observable, norm, domain, near-zero rule, or pass/fail inequality. The
admitted Scope Brief is the normative source of the `10^-3` phrase for this
draft, but the phrase is not an exact scientific predicate.

The targeted `r0.2` repair directive requires an exact owner-approved
numerical-adequacy profile and a matching realization-bound certificate. No
such exact profile or certificate is admitted or supplied by this source
packet. The repair therefore does not invent one from implementation,
floating-point folklore, model memory, or the approximate phrase.

**Disposition:** algebraic support remains defined. Numerical-realization
adequacy, including support under numerical near-cancellation, is typed
unavailable until an exact owner-approved profile and matching certificate are
available and compatible.

## Authority And Status Separation

| Control | Exact identity | Consequence |
| --- | --- | --- |
| Admitted successor Scope Brief | `SCOPE_BRIEF.md` at commit `88dcce8b0f7b1d78053b25831b39cf370afd47cc`, SHA-256 `5f2505c2760fc5cb07506249f33f449651aca67cccc9c444305b059674f0ddbd` | **Normative admitted source.** States the approximate instrument-relative principle, algebraic conditioning, and typed-unavailability discipline without an exact certification predicate. |
| Q002 packet control | manifest SHA-256 `52a8e843456a8cb033b7593d9b9f67fb83b0ee565c91c141d8e16d46b906140e` at the same commit | **Packet control.** Binds the exact admitted source bytes. |
| ODQ-109 owner record | `SCIENTIFIC_OWNER_ODQ_109_DECISION_2026-08-28.md`, SHA-256 `a9e44ea09e76cbc68ac70ee3d1e9a862f1b6ab82eff62da5ac9bbac97d28034e` | **Process-only corroboration.** It adds no normative scientific content beyond the admitted Scope Brief. |
| Q002 approval record | commit `ebc0e907fe96163e48818fec99e42cc272b2cfb4`, SHA-256 `c70e8216e816a7f98486b4c61236acc49713a5ce1d6f5ba722ad6e015e0c7e9f` | **Process-only.** Records owner approval of the exact packet; it is not scientific input. |
| Targeted repair directive | SHA-256 `c07505861d91459f69e7d0989f11551e2a14265c916cd5772ea48a86bb186ed2` | **Normative targeted repair input.** Requires an exact predicate or typed numerical unavailability; prohibits `approximately` or `negligible` as the final validity rule. |

The resulting disposition is an `r0.2` Stage B author-draft contract rule. It
is not an achieved certification, implementation assessment, or scientific-
owner acceptance of the repaired Stage B documents.

## Algebraic Support Predicate

For a pixel `p`, algebraic support requires all of the following:

```text
every required admitted input and contribution is finite;
N_p, C_p, and Q_p are finite;
Q_p > 0;
C_p != 0.
```

A finite negative `C_p` satisfies the sign-independent denominator condition.
Exact `C_p=0` is locally unavailable, not zero sky. No finite substitute,
infinity, clipping, positive-denominator restriction, unit-bearing floor, or
`Q_p` denominator is authorized.

The conditioning construction state

```text
D_p   = sum_i I_ip |omega_i kappa_ip|,
rho_p = |C_p|/D_p
```

remains dimensionless when defined. It describes signed cancellation; it does
not itself certify numerical adequacy. No universal `rho_p` cutoff is
authorized.

Passing the algebraic predicate is necessary but not sufficient for a
numerically supported `jinc_map` value.

## Required Numerical-Adequacy Profile

An exact profile `P_NA` must be owner-approved, immutable, versioned, and
selected through the requested/effective/observation-resolved/realized
lifecycle. At minimum it must bind:

| Field | Required exact content |
| --- | --- |
| Identity and authority | Immutable profile key/version, scientific owner, source digest, approval state, compatibility, and supersession rule. |
| Observable | Exact scientific observable or vector of observables whose numerical error is bounded. |
| Error form | Absolute, relative, mixed, or another explicit form; the exact numerator, denominator, and reference quantity. |
| Norm and scale | Exact norm or component rule, reference magnitude/scale, units, and treatment of rescaling. |
| Domain | Exact array, parameter-set, coefficient family, source population, phase, support, WCS, edge, pixel, and lifecycle domain. |
| Numerator/denominator treatment | Exact treatment of errors in `N_p`, `C_p`, their dependence, and the resulting ratio `m_p=N_p/C_p`. |
| Near-zero behavior | Exact rule when the observable, reference, `C_p`, or error denominator is zero or near zero. |
| Aggregation | Pointwise, simultaneous, worst-case, probabilistic, or other exact interpretation across the declared domain. |
| Pass/fail inequality | Exact boundary, boundary polarity, constants, and treatment of equality; no `approximately` or `negligible` token may substitute for it. |
| Error sources | Exact inclusion or exclusion of finite arithmetic, summation/reduction, analytic-function evaluation, phase quantization, cache/index realization, and any oracle approximation. |
| Uncertainty and proof | Exact uncertainty/confidence meaning or deterministic bound, proof/certification method, and omitted terms. |
| Lifecycle and provenance | Realization identity, tool/method identity, inputs, parents, immutable evidence digest, completion/failure state, and re-certification triggers. |

A general profile may not be inferred from an implementation tolerance, a
machine epsilon, a contributor count, a successful example, or the mere
existence of the approximate `10^-3` owner statement.

## Required Realization-Bound Certificate

For every realized numerical route, a certificate `E` must bind the exact
profile and the exact realization it assesses. It must record:

- profile identity/version and digest;
- implementation-independent operator identity, including WCS, center,
  phase, cache, kernel, coefficient, parameter, membership, and edge rules;
- exact realization and numerical-method identities;
- certified observable and complete domain;
- computed or proven numerator and denominator error bounds;
- reference magnitudes, norm, near-zero disposition, and worst-case or
  simultaneous result;
- exact inequality evaluation and pass/fail/unavailable state;
- uncertainty or proof record and every omitted term;
- immutable input/evidence digests and parents; and
- requested, effective, observation-resolved, realized, publication,
  failure, and supersession state.

A certificate for another parameter set, coefficient family, WCS, phase
resolution, numerical method, operator version, or lifecycle generation is
not compatible by numerical similarity.

When a certificate uses an absolute denominator bound
`|C_hat_p-C_p| <= epsilon_C,p`, normalized numerical support additionally
requires the exact separation `|C_hat_p| > epsilon_C,p`. Otherwise zero lies
inside the certified denominator interval and the normalized result is
unavailable. Any further map-error inequality must come from the exact
profile; this draft does not invent one.

## Current Typed State

| Proposition | Current base-v0.1 `r0.2` draft state | Cause and consequence |
| --- | --- | --- |
| Algebraic support | Defined | Finite required inputs, finite accumulators, `Q_p>0`, and `C_p!=0` are exact algebraic conditions. |
| Exact numerical-adequacy profile | Unavailable | Not admitted or supplied by this source packet. No default or approximate substitute is authorized. |
| Matching realization-bound certificate | Unavailable | A certificate cannot exist under this contract without an exact compatible profile and realization. |
| Numerical-realization adequacy | Unavailable | No exact pass/fail inequality can be evaluated. |
| Numerical near-cancellation support | Unavailable | A finite nonzero computed `C_p` and any value of `rho_p` cannot promote the pixel without the profile and certificate. |
| Numerical JINC route | Independently unavailable | In addition to this adequacy barrier, no exact registered/selected/realized JINC-permitted PTC coefficient payload or authorized TolTEC parameter set is admitted or supplied. |

The unavailable numerical state is not numerical failure, zero sky, zero
uncertainty, proof of instability, or an achieved error result. It states only
that the authority needed to decide adequacy is absent from the authorized
source set.

## Compatibility And Supersession

An exact owner-approved profile and a compatible certificate may close this
typed unavailable state in a versioned successor without changing the signed
estimator algebra. Any change to the observable, norm, domain, inequality,
near-zero rule, error-source budget, certificate method, or lifecycle creates
a new profile identity and requires re-certification. Earlier unavailable or
failed states remain immutable for replay.

The contract does not prescribe a summation algorithm, fixed reduction order,
thread order, cache storage, bitwise reproducibility, machine-epsilon formula,
or stronger precision in the absence of the exact profile. Those choices
cannot be declared adequate merely because they are engineering choices.

## Claim Boundary

This disposition makes no implementation-conformity, representation-fidelity,
numerical-validation, observational-validation, achieved-fidelity,
performance, readiness, or production claim. It records why no such numerical
claim is currently available.
