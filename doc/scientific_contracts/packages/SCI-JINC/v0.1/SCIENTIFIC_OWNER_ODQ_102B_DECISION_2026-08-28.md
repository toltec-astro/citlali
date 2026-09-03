# SCI-JINC-ODQ-102B — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

## Approved Scientific Disposition

SCI-JINC preserves the generic dimensionless radial coordinate

```text
r'_a = r/s_a,
```

where `s_a` is an explicit array-associated angular scale. The Schloerb JINC
reference supplies scientific precedent for the realization `s=lambda/D`, but
it does not authorize the current TolTEC realization or any TolTEC numerical
parameter set.

The current implementation's `s_a=lambda_a/(45 m)`, shape parameters, and
mode-dependent `r_max` values are inherited implementation defaults with
partially recoverable development history. They are evidence only, not current
TolTEC scientific authority. In particular, `45 m` must not be silently
interpreted as an effective aperture, illumination diameter, beam-derived
diameter, or any other physical quantity. Historical comments and commit
messages are provenance clues, not authorization, and no normative rule may be
reverse-engineered from the values.

Stage B may define the meaning, units, array association, admissibility,
identity, provenance, and unavailable-state behavior of `s_a`, `a`, `b`, `c`,
and `r_max`. Scientifically appropriate parameters may be array-associated
even where the present implementation stores a common value. Requested,
effective, observation-resolved, and realized parameter-set identities remain
distinct. With no scientifically authorized parameter set, the affected
numerical JINC route is unavailable; there is no hidden default or inherited
fallback.

Deriving or validating optimum numerical parameter sets for `a1100`, `a1400`,
and `a2000` is explicitly deferred to a separate downstream scientific
exercise with a stated optimization objective and appropriate TolTEC/LMT beam,
response, and/or noise evidence. SCI-JINC v0.1 does not prejudge whether that
study retains, approximates, or replaces the inherited values.

## Evidence-Only Baseline And Future Hypothesis

The exact inherited values, revisions, file digests, and recoverable history
are preserved only in `PRIOR_WORK.md` and `INTERNAL_DOSSIER.md`, both prohibited
author inputs. A future study may test the hypothesis that the inherited `b`
values and the historical `50 m` to `45 m` denominator change attempted to
track an effective TolTEC/LMT angular response rather than an ideal 50-m
diffraction scale. Available provenance is insufficient to adopt that
hypothesis as authority.

## Stage Consequence

`SCI-JINC-ODQ-102B` is closed by a semantic/no-numerical-route disposition,
not by approval of numerical values. This decision changes sanitized Stage A
author-control bytes and therefore remains subject to renewed exact-byte
approval under `SCI-JINC-STAGE-A-Q002`. It does not launch Stage B, authorize
the deferred optimization study, or establish implementation conformity,
validation, achieved performance, readiness, or production status.
