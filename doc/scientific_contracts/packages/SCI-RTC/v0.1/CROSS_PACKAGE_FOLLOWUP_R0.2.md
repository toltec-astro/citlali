# SCI-RTC v0.1/r0.2 cross-package follow-up list

These are routed questions, not authority imported into SCI-RTC.

| Package / boundary | Follow-up required | RTC dependency |
| --- | --- | --- |
| SCI-CAL | Define ordering and commutation for detector-static `flxscale`, sample-dependent target-atmosphere correction, detector mixing, and temporal filtering; define cross-plan calibration transfer evidence. | OWNER-001, OWNER-024, OWNER-035; REQ-068 |
| SCI-BEAM | Supply beam covariance or exact projected response identity, source-class domain, uncertainties, and complete Beammap RTC lineage for `flxscale`. | OWNER-012, OWNER-032, OWNER-035; EQ-023 |
| SCI-AST | Define output-time/coordinate correction authority and scan-velocity/frame validity for delayed RTC response. | OWNER-013, OWNER-034; EQ-027, REQ-067 |
| SCI-PTC | State which RTC-influenced samples/support it may consume, how donor/filter correlations enter cleaning and weights, and what response it preserves. | REQ-041--045, REQ-052 |
| SCI-VAL | Retain cause-preserving synthesis/replacement influence and define any additional role-specific eligibility without erasing RTC's fixed exclusions. | EQ-020b, REQ-019--020, REQ-046--047 |
| SCI-MAP | Bind exact RTC parent, cadence, timing, response, support and covariance; define response-aware map use for differing scan directions and plans. | REQ-037--052, REQ-067--068 |
| Downstream FLT | Compose map-domain filtering with the full RTC response; do not infer interchangeability from shared labels such as low-pass or notch. | REQ-037--041, REQ-068 |

Additional routing remains for ALIGN timing/synthesis, source fitting and
Pointing/OOF centroid interpretation, NOI covariance products, and FRUIT or
other online learning/restart designs. None is silently resolved here.
