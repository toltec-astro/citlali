# SCI-CAL v0.1 — Independent-Core Supersession Cover

Status: owner-approved author reference

Scientific owner: Grant Wilson

Approval date: `2026-08-16`

## Permitted Source

This cover applies only to:

`SCI-CAL-001_INDEPENDENT_CORE.tex` from
`codex/audit-sci-cal-001@27b0916e725696597c3ba84fb6a82bf6cf0ea356`,
with retrieved content SHA-256
`106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`.

The source is admitted because it was frozen as an implementation-independent
scientific derivation. Its equations, definitions, assumptions, uncertainty
reasoning, validity reasoning, and analytic limiting cases may be reused and
improved. The separate audit document and every implementation, finding,
repair, re-audit, test, execution, and conformity record are prohibited.

## Binding Supersessions And Restrictions

The approved [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) is later authority. The author
must apply all of the following:

1. **Measured-channel restriction.** V0.1 applies only to the ordinary `xs`
   detector stream.
2. **Initial output-unit restriction.** V0.1 supports only
   top-of-atmosphere, point-source-peak `mJy/beam`. General target-unit
   equations may be retained as explanatory algebra but do not authorize
   `MJy/sr`, `Jy/pixel`, temperature, extended-source calibration, or
   integrated photometry.
3. **Layered identity.** Acquisition identity, measured-APT identity,
   cross-observation source association, and design identity are distinct. A
   verified ordered-row relation is admissible when it is proven at the
   boundary; an explicit keyed relation is also admissible. Row position by
   itself is not identity, and a perfect design match is not required for
   ordinary use of measured Beammap calibration quantities.
4. **Artifact and occurrence scope.** A UID is local to its immutable artifact.
   Occurrence, semantic-content, and byte-transport identities are distinct.
   Cross-artifact correspondence requires explicit occurrence-scoped endpoint
   references; equal local keys, row positions, paths, or integer spellings do
   not prove correspondence.
5. **Atmosphere authority.** The retained
   `am12_fixed_djf25_piecewise_linear_los_tau_v1` operator is structurally
   adopted for authorship, including continuity in line-of-sight optical
   depth, positivity, the analytic zero anchor, and exact declared nodes. This
   does not establish atmosphere truth, representation fidelity outside
   demonstrated support, observational repeatability, absolute photometric
   accuracy, or a general production domain.
6. **Engineering-opacity restriction.** No calibrated SCI-CAL output is
   authorized for `0.15 < tau225 <= 0.25` until a continuous engineering
   operator is separately adopted. No legacy-selector fallback or silent
   extrapolation is permitted.
7. **Passband limitation.** The content-bound TolTECA v1 passband set is an
   approved modeled-array reference with recorded unknowns. It is not a claim
   of detector/network weighting, telescope measurement, uncertainty,
   covariance, normalization, or a settled photon-versus-energy convention.
8. **Uncertainty limitation.** Conditional measurement variance/weight is not
   total calibrated uncertainty. Missing nuisance uncertainty is unavailable,
   never zero, and common calibration terms do not average down merely because
   many detector samples enter a product.
9. **Response limitation.** Originating Beammap beam/template identity and
   realized map/filter response are distinct. SCI-CAL records the response
   basis needed to interpret `mJy/beam`; it does not certify empirical
   downstream response fidelity.

Where the independent core conflicts with this cover or the approved Scope
Brief, the later approved material governs. The author must surface any other
conflict instead of silently choosing one source.

## Information Firewall

This cover does not authorize opening the historical scientific-contract
audit beside the independent core, any Citlali source, executable contract,
test, repair branch, re-audit, numerical evidence package, Unity material, or
active ALIGN work.
