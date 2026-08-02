# SCI-CAL-001 owner direction and provenance correction — 2026-08-01

## Record identity

- Package: `SCI-CAL-001`
- Date: 2026-08-01
- Authority: project owner acting as calibration and atmosphere scientist
- Governing repair base: `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- Repair-line evidence head at dispatch: `ae99be1cef8c390d0e7490835ffca1f31da7ebc0`
- Direction: evaluate a separately versioned AM 12.2 successor; evaluation only
- Adoption status: not adopted
- Operator authorization: none
- Operational-domain authorization: none

This additive record resolves the model-lineage path requested in
`OWNER_DECISION_BRIEF.md`. It authorizes the bounded numerical study in
`AM12_SUCCESSOR_ADOPTION_STUDY_PROTOCOL.md`; it does not authorize Citlali
application changes, repair implementation, re-audit, Unity work, or
production use.

## Owner direction

The recovered AM 12.2 calculation suite is scientifically promising enough to
evaluate as a new, explicitly versioned atmosphere model. The study is not a
claim that the copied suite regenerated the historical generic-q products.
Generic-lineage archaeology may continue as nonblocking historical work, but
it is no longer the selected path to an operational successor.

The evaluation domain is frozen to the regime in which TolTEC is intended to
operate:

```text
zenith tau225: 0 <= tau225 <= 0.158313198574890929
elevation:     20 deg <= EL <= 80 deg
```

The upper opacity endpoint is the exact repair-base generic-q75 coordinate.
This is a study domain, not an already approved production domain. Values
outside it are unsupported and must fail closed; percentile names are never a
substitute for testing the numerical `tau225` coordinate.

The generic and copied q95 evidence is retained as a diagnostic archive only.
Missing generic q95 bytes, the generic q95 provenance gap, and the legacy
q95/a2000 elevation feature are not gates for this q0--q75 successor study.
The owner states that TolTEC would not ordinarily be used in q95 conditions.
No q95 model, q75--q95 interpolation, or q95 extrapolation may enter the
candidate operator.

The successor study shall compare two AM 12.2 anchor constructions:

1. one fixed `LMT_DJF_25` atmospheric profile independently H2O-scaled to the
   exact q25, q50, and q75 opacity anchors; and
2. conditioned DJF nodes using `LMT_DJF_25` at q25, `LMT_DJF_50` at q50, and
   `LMT_DJF_75` at q75, each with the minimum H2O-scale adjustment needed to
   match its exact target.

The primary spectral artifacts are the versioned TolTECA ECSV passbands. The
three local FTS spectra are challengers. Every lane is evaluated for source
power-law indices `alpha = -1, 0, 2, 4`; no one of those spectra may be
silently treated as universal.

## Provenance correction

The copied annual and seasonal AM 12.2 NPZ products are demonstrably distinct
registered product identities from the generic TolTECA q products. In
particular, none is the generic q95 datafile ID 461 with expected MD5
`0ca7b331823237767d26016d19bffb3d`. The copied products also differ
numerically from the generic q25/q50/q75 grids and the legacy q95 polynomial
surface.

Those facts establish a **product-identity mismatch**. They do not establish
that AM 12.2, the copied AM source release, or an equivalent AM 12.2 build did
not generate the historical generic products. The exact executable, AMC
profile construction, H2O scales, raw run records, and packer associated with
the generic products have not been recovered. The correct generator-lineage
status is therefore:

```text
historical generic-q association with AM 12.2: not established
```

Any package statement that the copied suite "is not the historical q-model
lineage" is superseded by this narrower interpretation. In particular, the
frozen `FOLLOWUP_STUDY_PROTOCOL_ADDENDUM.md` remains immutable protocol
evidence, but its triggering statement is interpreted as a registered-product
mismatch, not documentary proof of a different generator. Numerical mismatch
alone may not be promoted into generator-custody evidence.

This correction does not allow copied products to be renamed as generic q
products, does not close the missing generic q95 artifact, and does not change
any frozen numerical result.

## Acceptance hierarchy retained

The owner retains three separate gates:

1. software and scientific-contract correctness;
2. no more than 1% fractional extinction-correction error added by the
   numerical representation relative to its declared raw AM 12.2 truth over
   the study domain; and
3. later observational evidence for approximately 5--10% absolute flux
   accuracy and provisional approximately 5% observation-to-observation
   repeatability, with no significant residual opacity or airmass trend.

Passing the second gate does not claim one-percent physical photometry.
Repeated samples do not average down common calibrator, Beammap-extinction,
bandpass, selector, or airmass systematics.

Zenith `tau225` must be applied with each eligible sample's full airmass and a
top-of-atmosphere pivot `X_ref=0`. The late `SCI-CAL-001-XAUD-001` handoff
remains an open dependency: any eventual operator may use aligned elevation
only with explicit ordered sample identity, timing-gap/interpolation origin,
duration, and original-versus-synthesized eligibility. That dependency does
not change the atmosphere equations or broaden this study.

## Stop boundary

The next permitted action is the deterministic, digest-bound adoption study.
Stop after its numerical decision package. A scientific owner decision is
still required before selecting an exact production operator and operational
domain. Do not modify Citlali application code, contact Unity, launch repair
implementation, launch a re-audit, or edit the coordination registry under
this direction.
