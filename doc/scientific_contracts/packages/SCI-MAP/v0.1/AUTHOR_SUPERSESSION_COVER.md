# SCI-MAP v0.1 — Draft Supersession Cover For Reusable MAP Core

Status: draft for owner review; not authorized for an author packet

This cover accompanies only:

`c28f18ed089657dae278caba2d6d6d65c7ec72f4:doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`

Verified content SHA-256:
`13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`.

The core is reusable implementation-independent science. Its associated audit,
source inspection, findings, repairs, tests, campaigns, re-audits, conformity
claims, and production status are not author references.

## Binding v0.1 Specializations

If Grant approves this cover, the following later decisions control whenever
the reusable core is broader or ambiguous:

1. **Ordinary estimator only.** V0.1 selects finite positive-coefficient
   normalized gridding and the associated response/kernel transformation.
   General GLS remains an analytic benchmark, not an authorized production
   estimator.
2. **Nonprecision coefficient.** The ordinary denominator and published
   coefficient are gridding/normalization facts by default. Dimensional
   inverse-square units do not establish inverse variance, independence, full
   covariance, or statistical significance.
3. **No signal centering.** For v0.1 coaddition, `L=I`. The alternative
   centered/null-space branch in the core is explanatory only and is not the
   adopted estimand.
4. **Centered integer common grid.** Coaddition admits a complete compatible
   observation bundle before mutation and permits only centered integer
   embedding on the same WCS-defined grid. Reprojection, interpolation,
   fractional shifts, wrapping, and implicit source recentering are excluded.
5. **Explicit raw validity.** Hits, exposure, normalization support,
   science-policy support, and final raw science validity are distinct.
   Neither finiteness nor any one coefficient/support/compatibility plane is a
   validity authority.
6. **Eight distinct facts.** The v0.1 raw bundle separately represents
   geometric incidences, estimator contributions, coadd observation count,
   upstream-eligible exposure, retained exposure, normalization support,
   science-policy support, and authoritative raw science validity.
7. **Adopted threshold policy.** The exact separate normalization and
   science-policy threshold rule stated in the approved Scope Brief is binding
   v0.1 policy. The core must not invent a physical rationale or future change
   authority for it.
8. **Same membership and parentage.** Signal, response/kernel, admitted linear
   realizations, retained exposure, and coadd observation count use the same
   membership and placement where their definitions require it. Downstream
   products preserve immutable raw validity and parent identity.
9. **Current unit and capability boundary.** V0.1 is Stokes I. The active
   calibrated signal boundary is `mJy/beam`; other unit tokens are not
   authorized by configuration acceptance alone.
10. **Identity qualification.** Map/group/Stokes identity is explicit and
    cannot be reconstructed from a container position. Detector/acquisition
    binding is occurrence/product scoped; the core's convenient UID wording
    must not be read as a persistent universal detector namespace.
11. **JINC excluded.** The signed JINC estimator, subpixel response,
    conditioning, coverage, and product semantics belong to MAP-002. No
    positive-coefficient predicate or complete ordinary bundle is imputed to
    JINC.
12. **OOF transfer excluded.** MAP-003's residual transfer estimator and
    LMTOOF boundary are a distinct product package, not the ordinary map
    response derived here.
13. **Claim layers remain separate.** Algebraic contract correctness,
    implementation conformity, representation/response fidelity,
    observational performance, and production readiness are different claims.
    The reusable core supplies no achieved implementation or observational
    claim.

## Permitted Use Of The Core

The author may:

- reuse its definitions, equations, conditional reasoning, limiting cases, and
  falsifiable predictions;
- shorten and reorganize them under the SCI-CAL science-rationale house
  standard;
- identify genuine gaps or inconsistencies; and
- state upstream dependencies and unavailable stronger claims.

The author may not:

- consult or cite the associated audit, findings, source trace, implementation,
  repairs, tests, reductions, validation, or current status;
- rewrite the ordinary normalization derivation merely to appear new;
- broaden v0.1 to JINC, maximum-likelihood, OOF transfer, filtering, noise,
  fitting, Beammap, or fruit-loop science;
- infer unresolved CAL, ALIGN/AST, PTC, VAL, or NOI facts; or
- weaken an adopted owner decision to match a remembered implementation.

An owner-approved, content-hashed version of this cover will be listed in the
author-packet manifest before authorship begins.
