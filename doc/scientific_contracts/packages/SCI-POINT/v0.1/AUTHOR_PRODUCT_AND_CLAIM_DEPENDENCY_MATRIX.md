# SCI-POINT Product And Claim Dependency Matrix

Identity: `SCI-POINT_PRODUCT_CLAIM_DEPENDENCIES v0.1/r0.3`

| Product or claim | Required available facts | Facts that may remain typed unavailable | Blocking result |
| --- | --- | --- | --- |
| `POINT-FIT-RESULT` | exact numerical parent signal; parent WCS and physical tangent metric; approved `POINT-COMPATIBILITY-METHOD`; complete admitted support; exact objective/fit-weight state; successful fit realization; component states | response and covariance status records may be unavailable unless the selected method requires their numerical values | no numerical fit or fit-derived product |
| amplitude, centroid, widths, orientation | conforming fit and exact component-identifiability state | unrelated formal, empirical, or downstream state | affected component unavailable/failed without erasing conforming siblings |
| `POINT-SOURCE-ASSOCIATION-STATE` | exact source-reference authority and branch-independent association method/domain | fit response or formal covariance unless association method requires it | source attribution unavailable or failed |
| processed-map displacement measurement | fitted centroid; expected source position; identical tangent chart/basis; established association | parent response and centroid-bias may be explicitly unavailable but must be retained | no source-attributed displacement |
| unbiased telescope-pointing or correction-use claim | processed-map displacement plus named-use-required response center, centroid bias, uncertainty, and pointing-support policy | nothing required by that named use | eligibility ineligible or decision-unavailable; missing bias is not zero |
| formal errors | conforming fit and approved `POINT-FORMAL-ERROR-METHOD` | separately scoped empirical/NOI uncertainties | formal errors unavailable; fitted values may remain |
| formal standardization | fitted amplitude and available positive finite formal amplitude error | unrelated covariance representations | diagnostic unavailable |
| dynamic-range diagnostic | fitted amplitude and approved `POINT-FULL-MAP-RMS-METHOD` on the exact related parent | formal-error method | only dynamic-range product unavailable |
| photometric transfer | source-associated amplitude plus CAL/TolProj-owned amplitude-admission profile and source/reference authority | unrelated QC/correction use state | photometric eligibility ineligible or decision-unavailable |
| telescope-QC diagnostic display | required fit facts plus QC-owner applicability/eligibility/action | correction and photometric eligibility | display not realized; cannot rescue another use |

Unavailable response or covariance is handled according to the selected row
and named-use policy; it neither universally erases every processed-map fit nor
universally authorizes response, bias, uncertainty, correction, calibration,
or photometric claims. A finite fit supplies only the products whose complete
dependencies are satisfied.
