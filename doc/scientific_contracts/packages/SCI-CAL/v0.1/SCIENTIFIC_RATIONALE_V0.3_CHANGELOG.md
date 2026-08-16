# SCI-CAL Scientific Rationale v0.3 Change Log

Date: 2026-08-16

Scope: surgical correction of v0.2 after scientific-owner review. No new
requirements, numerical science, achieved validation claims, or engineering
contract changes were introduced.

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

## Architecture status

After the v0.3 owner voice review, this science-rationale architecture is
frozen as the template for later packages. Further SCI-CAL rationale changes
require an explicit owner decision or an engineering-contract change.
