# POINT strategy for team review

Recorded November 4, 2026, from Grant Wilson's owner direction. This is the
proposed observing policy for team review; implementation and scientific
qualification remain pending.

The primary purpose of POINT is to measure azimuth and elevation offsets,
targeting approximately **1.5 arcsec combined two-dimensional accuracy**.
Using pointings to assess changes in telescope gain is a separate, optional
use. POINT operates on individual observations; celestial coadding belongs
to the SCIENCE workflow.

All three arrays will be reduced, with their individual offsets, fitted beam
shapes, and fit assessments retained. The adopted pointing solution follows
these priorities:

1. **Beam-quality override comes first.** If the **1.1 mm fitted FWHM exceeds
   8 arcsec in either dimension**, flag possible severe deformation or defocus
   and adopt the **usable 2 mm solution alone**, even when the arrays disagree.
   The 8 arcsec threshold is provisional.
2. **With roughly nominal beams, average the consistent arrays.** Agreement is
   typically around 1 arcsec or better. A clear outlier differing by
   **5 arcsec or more** from an otherwise agreeing pair is excluded from the
   arithmetic mean, while its measurement and exclusion reason remain in
   the report.
3. **With roughly nominal beams and no agreeing pair, report failure.** If all
   three pairwise offset separations are **5 arcsec or more**, the observation
   yields no accepted pointing solution. Its measurements and diagnostics
   remain available.

An observer-facing output will report each array's beam condition, the adopted
offset, the contributing arrays, and the selection or failure reason. The
proposed delivery is an extension of the existing JSON status record alongside
the ECSV measurement table. The LMT monitor and control system can consume the
assessment, and TolTECA will be updated to honor the selected solution while
retaining responsibility for converting measured displacement into a correction.

Team review should address the thresholds and decision rules, particularly
the definitions of consistent and nominal beams, ambiguous array groupings,
and unavailable fits. The accuracy target is not yet demonstrated. The
[method and use acceptance record](POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md)
links existing evidence and distinguishes agreed direction from those open cases.
