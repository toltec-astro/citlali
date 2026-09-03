# SCI-AST v0.1 Stage B r0.2 Requirement/Equation/Prediction Change Map

Status: semantic-change traceability for a targeted author draft; all stable
requirement and prediction identifiers are preserved

| Targeted repair | Canonical equations/definitions changed | Stable requirements amended | Stable predictions amended | Semantic effect |
| --- | --- | --- | --- | --- |
| Exact stable slot | `align-parent`, `theta-align`, correction support/covariance, detector-direction composition, RTC signal/response | REQ-017, 052, 073-077 | PRED-035-037 | `s` is stable `(o,s)`; `j` is local row only; `theta^A_ds` replaces `theta^A_dj`. |
| Sky/readout symbol separation | `unit-basis`, `exp`, corrected-boresight through frame-operation, TAN forward/inverse/lonlat, projection Jacobian | REQ-036-044 retain IDs and corrected referenced notation | PRED-022 | `u_sky` is unit direction; TAN axes are `zeta`; `x/r` are exclusively readout coordinates. |
| Exact circular topology | `wrap`, `norm`; antipodal-unavailability rule | REQ-038 | PRED-018 | Shortest signed difference is exactly `[-P/2,P/2)`; antipodal interpolation needs explicit unwrap authority. |
| Role-factored parents | `atomic-bundle`, `align-parent`, new `tangent-parent`, `pixel-parent`, `nominal-pixel-parent`, amended `rtc-parent` | REQ-056-060, 073-077 | PRED-025, 030, 035-036 | Direction, tangent, pixel, nominal-pixel, and RTC roles have layered atomic facts; downstream failure does not erase upstream truth. |
| Exact ALIGN profile | `align-parent`; exact-import definition and compatibility/supersession prose | REQ-006 | PRED-049 remains the parent/version falsifier | Canonical AST authority names `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; shape/name similarity is not compatibility. |
| Geometry candidate versus MAP deposition | new `geometry-incidence`; amended `gpi-parentage`; projection-ownership definition | REQ-060, 080-083 (content of 081-082 amended) | PRED-036, 038-040 (038-039 amended) | `I^geom_pi` is a non-estimator candidate with no contribution/normalization/conservation/MAP-validity/response/covariance claim; `G_pi` is exact MAP-owned deposition only. |
| Ordinary nonpolarimetric coordinate path | `ASM-014`, frame/product-family definition, and scientist narrative wording clarified; no canonical equation changes | No requirement ID or content change | PRED-047 wording clarified | AST does not interpret polarization. Optional HWPR timing remains only an ALIGN-parent fact and authorizes no demodulation, polarization calibration/response, or Stokes reconstruction. Raw KID `x` is not identified as Stokes I. |
| Audience separation | Scientist narrative selects explanatory equations; engineering view retains all formal modules | No IDs changed | No IDs changed | Complete 90 requirements and 50 predictions remain in Engineering Conformance; rationale becomes an 8-10 page narrative with three figures. |

No requirement or prediction was added, removed, or renumbered. Unlisted
requirements and predictions retain their prior scientific content while
inheriting the corrected canonical notation and parent identities where they
refer to shared definitions or equations.
