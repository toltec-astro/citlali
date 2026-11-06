# POINT strategy

The team approved the strategy on November 4, 2026. This November 5 update
incorporates Grant Wilson's subsequent initial-policy decisions. The
[acceptance record and decision table](POINT_METHOD_AND_USE_ACCEPTANCE_2026-11-04.md#agreed-decision-table)
provide the full cases, precedence, evidence and remaining qualification.
The [implementation scope](POINT_POLICY_IMPLEMENTATION_SCOPE_2026-11-05.md)
is ready for owner review.

POINT measures the source displacement in tangent azimuth and elevation for
an individual observation. The target is approximately **1.5 arcsec combined
two-dimensional accuracy for the final adopted solution**. That accuracy
awaits qualification. The initial method uses the current elliptical Gaussian
plus constant fit. Per-array offsets, fitted beam shapes, fit status and
unavailable uncertainties remain explicit.

| Condition | Adopted solution |
| --- | --- |
| a1100 larger principal-axis FWHM is strictly above 8 arcsec | Use usable a2000 alone. If a2000 is unusable, withhold automatic pointing. This override takes priority over all other rows. |
| Three usable arrays, all pairwise separations below 5 arcsec | Arithmetic mean of all three. |
| Three usable arrays, exactly one pair agrees below 5 arcsec | Mean of that pair. This includes a1100/a1400 agreeing while a2000 is the outlier. |
| Three usable arrays with ambiguous agreement, or all pairs discrepant | Withhold and report the reason. |
| Two usable arrays agree below 5 arcsec | Mean of the pair. |
| Two usable arrays disagree by at least 5 arcsec | Use a2000 if it is one of the usable pair. Otherwise withhold. |
| One usable array, override absent | Use that array and report that cross-array comparison was unavailable. |
| No usable array | Withhold and retain the per-array results and causes. |

Separations are two-dimensional distances in the common tangent plane.
Equality at 5 arcsec counts as disagreement. The strict 8 arcsec beam trigger
is the agreed initial setting with provisional scientific qualification.
Assessment of a1100 shape when its fit is unusable is identified in the scope.

Background diagnostics use **30 <= r < 60 arcsec around each array's measured
centroid**. Report RMS about the background mean, including real finite
background structure, and robust scatter alongside it. Report fitted peak
above the fitted constant divided by background RMS as map contrast.

Require at least **75% coverage overall and 50% in each of the four
azimuth/elevation quadrants**. These initial safeguards await qualification
on incomplete coverage. Insufficient coverage alone permits an otherwise
valid pointing with a visible background-quality warning. Map contrast is
diagnostic-only initially. New contrast rejection and tuning-warning
thresholds await evidence.

The observer report will identify the adopted measured offset, contributing
and excluded arrays, selection reason, beam assessments, background diagnostics
and warnings. The per-array measurements remain available. TolTECA needs a
bound consumer change to honor the selection while constructing the telescope
correction with the established signs and User/Paddle composition.

Implementation scope approval is next. Gain monitoring and calibration have
their own acceptance requirements. SCIENCE owns celestial coaddition and the
deferred RA/Dec known-offset qualification.
