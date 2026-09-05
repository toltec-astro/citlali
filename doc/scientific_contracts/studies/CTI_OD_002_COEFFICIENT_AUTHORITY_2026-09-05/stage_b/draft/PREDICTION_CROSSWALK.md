# Prediction/evidence crosswalk

Draft: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**  
Status: **proposed predictions; no evidence performed**

| Prediction | Required discriminating observation | Evidence layer |
| --- | --- | --- |
| UFC-PRED-001 | Every decoded member, pairwise ratio, and population mean is exact one. | Algebraic conformance |
| UFC-PRED-002 | One-member support realizes one coefficient. | Algebraic conformance |
| UFC-PRED-003 | Empty support realizes no coefficient or empty-pass decision. | Algebraic and profile conformance |
| UFC-PRED-004 | Relabeling alone leaves values one; any actual support change changes generation identity. | Identity/representation conformance |
| UFC-PRED-005 | Replay is numerically identical, identities remain distinct, and unbound substitution is rejected. | Lifecycle/replay conformance |
| UFC-PRED-006 | Any constant override fails exact family selection. | Registry/profile conformance |
| UFC-PRED-007 | Non-one and nonpositive/nonfinite/unrepresentable payload cases are rejected at their stated producer/consumer layers. | Payload and consumer conformance |
| UFC-PRED-008 | Missing payload remains unavailable without unity fallback. | Failure-route conformance |
| UFC-PRED-009 | Scalar one without complete broadcast identity yields structural unavailability. | Structural-binding conformance |
| UFC-PRED-010 | Coefficient fidelity and signal validity can differ without either overwriting the other. | Cross-profile conformance |
| UFC-PRED-011 | MAP sees coefficient one while its own factors and gates remain effective. | MAP boundary conformance |
| UFC-PRED-012 | JINC’s PTC factor cancels to `kappa_ip` while JINC’s signed rules remain effective. | JINC boundary conformance |
| UFC-PRED-013 | Removing either permission affects only that named consumer. | Registry/boundary conformance |
| UFC-PRED-014 | No selection and no default leave both routes unavailable. | Policy/failure-route conformance |
| UFC-PRED-015 | NOI requires exact rational-parent binding even when the PTC factor is `1/1`. | NOI parent-binding conformance |
| UFC-PRED-016 | PTC-disabled and missing PTC realization produce no coefficient or consumer route. | Lifecycle/failure-route conformance |

No evidence named here has been performed.
