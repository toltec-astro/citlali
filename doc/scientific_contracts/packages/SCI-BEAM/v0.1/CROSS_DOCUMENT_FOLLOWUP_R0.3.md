# SCI-BEAM v0.1 r0.3 — Cross-Document Follow-Up Delta

Status: future governance work; no other package edited by this revision

Date: 2026-08-17

The r0.2 follow-up register remains applicable. R0.3 adds or strengthens only
these future alignments:

| Owning authority | Required r0.3 alignment |
| --- | --- |
| ALIGN/AST and pointing | Preserve the explicit centroid-to-detector sign/frame transformation, derive effective rotation from parent-sample contribution support, and require the same immutable APT artifact for bracketing pointing and associated science observations unless an authorized transform proves equivalence. |
| Weighting/noise | Consume finite strictly positive sens; preserve signed flxscale separately; own every admission, covariance, and normalization choice for weights proportional to inverse sens squared. |
| SCI-CAL and calibration schemas | Require the source flux in TOA mJy per fixed nominal beam and keep map-fit covariance distinct from derived calibration covariance. |
| Beammap APT/product authority | Resolve fit Jacobian/covariance and derived-product Jacobian/covariance separately, retaining material dependence between stages. |
| Validation and future audit | Test both flxscale signs, strictly positive available sens, centroid-sign inversion, parent-sample versus pixel-weight rotation, exact APT-artifact identity, and fit/derived covariance separation. |

Each change requires the owning authority's normal review. This delta does not
change any adjacent package, implementation, schema, or production state.
