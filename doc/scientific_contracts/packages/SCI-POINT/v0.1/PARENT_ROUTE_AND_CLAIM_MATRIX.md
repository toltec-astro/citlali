# SCI-POINT Candidate Parent Route And Claim Matrix

Status: owner-approved eligible families; no numerical route yet available or bound

Each row is a different scientific method candidate. A product filename,
directory name, or compatible array shape cannot substitute for the exact
route identity.

| Candidate parent | Potential POINT interpretation | Additional required state | Stage A disposition |
| --- | --- | --- | --- |
| ordinary observation-local SCI-MAP array map | source displacement/amplitude/shape under ordinary MAP response | exact MAP product, WCS, support, response, covariance, calibration, generation | eligible distinct route; exact numerical binding still required |
| observation-local SCI-JINC array map | source displacement/amplitude/shape under signed JINC response | exact JINC normalization, coefficient, support, response/covariance, generation | eligible distinct route; not equivalent to MAP |
| observation-local SCI-FLT-FIXED product | fit after declared fixed convolution | exact parent plus filter operator/kernel, response, edge/support, covariance state | eligible distinct route; conditional on exact numerical binding |
| observation-local SCI-FLT-MATCHED product | location-indexed matched-template amplitude and centroid relation | exact template, complete support, output unit, response, covariance option, phase/origin | eligible distinct route; fitting a matched-filtered peak is not automatically the same estimator |
| NOI standardized-signal companion | search/diagnostic scale only | exact signal parent and NOI uncertainty product | not a POINT amplitude/displacement parent in base v0.1 |

## FRUIT Lineage Rule

FRUIT is not a separate POINT parent family. A terminal result created through
FRUIT must already be typed as an exact observation-local MAP, JINC,
FLT-FIXED, or FLT-MATCHED product. POINT uses the corresponding row above and
also binds the complete FRUIT method, terminal iteration, generation, response,
support, uncertainty, and lineage state. An intermediate FRUIT iteration is
not a base-v0.1 parent.

## Coadd Deferral

Coadd parents are outside base v0.1 by owner direction. A future successor may
define a separate coadd estimand, observation association, response/covariance,
and use boundary.

## Claim Rules

For any admitted route, POINT may claim only the fitted quantities under the
exact parent's response, support, frame, unit, calibration, and uncertainty
state. A transformed route may improve practical localization or amplitude
stability without proving equivalence to the ordinary-map fit. Route choice
must be requested or scientifically resolved; no automatic fallback or silent
selection is proposed.

`raw` and `filtered` are implementation labels, not sufficient scientific
method identities. Stage B must use exact parent product and operator names.

## Availability

All four map families are scientifically eligible under approved ODQ-003.
Each numerical route remains typed **unavailable for SCI-POINT v0.1** until
its exact predecessor authority, required state, numerical product, and POINT
compatibility boundary are present and bound. Unavailability of one route
does not authorize substitution or fallback to another. This does not
invalidate predecessor products for their own authorized uses.
