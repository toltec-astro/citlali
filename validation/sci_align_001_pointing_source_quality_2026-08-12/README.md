# SCI-ALIGN-001 pointing source-quality audit

This bounded pre-timing audit inventories the 108 canonical
`diagnostics/standard_trial` pointing products supplied in the owner-retrieved
Unity inventory dated 2026-08-12.

The only automatic selection is `a1100.sig2noise >= 60`. Broad or elliptical
sources are not rejected. The a1100 map fit, residual, secondary-peak, and
connected-residual statistics prioritize a human morphology review; they do
not automatically exclude an observation. No PTC signal sample or timing-fit
result enters the audit.

The expected local workflow is:

```bash
$HOME/tolteca/bin/python \
  tools/diagnostics/audit_sci_align_001_pointing_source_quality.py freeze \
  --schema-audit <downloaded-inventory>/pointing_standard_trial_schema_audit.csv \
  --protocol validation/sci_align_001_pointing_source_quality_2026-08-12/frozen_protocol.json \
  --unity-root /work/toltec/commissioning2025-test \
  --local-root "$HOME/work_toltec/local_data" \
  --output <run-root>/frozen

$HOME/tolteca/bin/python \
  tools/diagnostics/audit_sci_align_001_pointing_source_quality.py run \
  --frozen-input <run-root>/frozen/frozen_input.json \
  --output <run-root>/result
```

The PDF contact sheet contains the S/N-selected observations ordered by the
strongest absolute smoothed off-core residual fraction, solely to put the most
informative morphology review pages first. Positive secondary-peak candidates
are marked with an `x` but remain human-review flags rather than exclusions.

## Frozen local result

The owner-retrieved schema audit has SHA256
`a9cbccf935aa626c2d9a531e71e55bac3be4e6028d8c2d8a01bf7dc6c37b3944`.
All 108 canonical PPT and a1100 pointing-map identities were frozen before the
map metrics were interpreted. The result contains 66 observations with
`a1100.sig2noise >= 60` and 42 below the threshold.

Seven S/N-selected observations contain a positive secondary-peak candidate
under the descriptive thresholds: 123426, 131090, 136278, 137554, 137574,
143139, and 148719. Fourteen contain at least one coherent residual component:
123426, 130156, 131088, 131090, 134234, 134642, 134644, 136278, 137554,
137574, 143139, 143441, 148717, and 148719. These lists prioritize visual
review and are not rejection lists.

The checksum-protected PDF has 22 pages and was rendered and visually reviewed
in full. It is legible and places the raw a1100 map directly above its
Gaussian-plus-plane residual. The source FWHM ellipse and any positive
secondary-peak candidates are overlaid. The PDF SHA256 is
`21668adaad40bba48a20f0c8e77951ec4ca79ba25548151ba0ebe54f305fdca4`.

The bounded conclusion is only that the S/N gate reduces the available
pointing corpus from 108 to 66 and that the retained set now has an explicit
human morphology-review surface. No morphology disposition has been assigned,
by the numerical audit, and no timing result entered either the selection or
the review ordering. The subsequent project-owner disposition is recorded
separately below.

## Project-owner morphology disposition

After reviewing the complete contact sheet, the project owner identified the
secondary structures as expected but unwanted aberration lobes, principally
coma with astigmatism in some observations. They are not considered plausible
competing identifications of the primary source crossing. All 66 S/N-selected
observations therefore receive the disposition
`pass_with_recorded_aberration_structure`; none is excluded on morphology.

The typical strongest absolute smoothed residual is near 10% of the fitted
primary amplitude (median 8.3%, 75th percentile 11.3%), although the tail is
materially larger (90th percentile 17.9%, maximum 37.2%). Static asymmetric
beam structure can still interact with a symmetric source template and scan
angle, so the residual metrics remain sensitivity covariates. The timing study
will compare the full cohort with the strongest-aberration observations
removed and, where feasible, compare empirical map-derived templates with
symmetric Gaussian templates. This disposition makes no timing-causality
claim.
