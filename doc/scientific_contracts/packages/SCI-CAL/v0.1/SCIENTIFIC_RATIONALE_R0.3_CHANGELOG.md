# SCI-CAL v0.1 Scientific Rationale r0.3 Change Log

Date: 2026-08-16

Scope: tightly constrained correction of rationale r0.2 after
scientific-owner review. No new requirements, numerical science, achieved
validation claims, or engineering-contract changes were introduced.

## Corrections

- Reassigned scientific ownership using the producer--transformer--consumer
  rule: Beammap/source-APT production owns source calibration and flxscale
  meaning; TolProj owns association and explicitly approved child
  transformations; SCI-CAL applies the selected child factor and target
  atmosphere once; MAP/FLT owns realized downstream response.
- Corrected the executive summary and lineage figure so the selected
  observation-specific child APT, not a Beammap APT, feeds SCI-CAL.
- Replaced the ordering shorthand with three explicit cases: a fixed scalar
  with a single-detector linear operator, a detector-diagonal factor with a
  detector-mixing operator, and a sample-dependent atmosphere factor.
- Made the atmosphere equation orientation-neutral by introducing H and
  defining transmission and correction by declared ordinate orientation.
- Removed unsupported total-intensity and array-layout claims about xs.
- Replaced the potentially confusing G notation with an abstract response R,
  and stated that R is not the APT responsivity field.
- Revised the main-text voice to distinguish available materials, approved
  definitions, open scientific decisions, and criteria for validated science
  use.
- Added the required companion owner-decision ledger and refreshed the
  scientist/engineering crosswalk and consistency report.
- Separated scientific-contract version v0.1 from document revision r0.3 in
  the title, running header, metadata, filenames, and package governance.
- Removed the internal production note from the title page, expanded APT at
  first use, and applied the final sentence-level corrections requested by the
  scientific owner.
- Clarified the adopted opacity thresholds, changed Q06's consequence to
  contract-supported numerical calibration, and renamed Table 2's status
  column.
- Expanded the live decision ledger to separate resolution authority and date
  and to identify every affected document family.

## Architecture status

After the r0.3 owner voice review and final cleanup, this science-rationale
architecture is frozen as the template for later packages. There is no further
stylistic round. A later SCI-CAL rationale revision requires a formal Q01--Q09
resolution, a normative engineering-contract change, changed validation
evidence, or a genuine scientific inconsistency.
