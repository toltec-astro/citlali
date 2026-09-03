# SCI-ALIGN / SCI-AST Stage B r0.3 Horizontal Contract-Coherence Audit

Status: implementation-blind document-coherence audit; not scientific
approval, implementation conformity, empirical adequacy, validation, freeze,
observational performance, readiness, or production authorization

Prepared: `2026-08-22`

## Scope And Method

This audit compares only the final SCI-ALIGN and SCI-AST Stage B r0.3 contract
packets produced under directive SHA-256
`a947ab55eab404f7740d47d3c87766a77f4714bd6ce5c6c9b578c6dc5aff9c0f`.
It does not inspect implementation, schemas, tests, prior audits, repairs,
validation evidence, production products, or historical behavior.

The comparison covered both scientist-facing rationales, both complete formal
views, all six canonical modules per package, the exact shared boundary,
notation/change maps, semantic-change maps, crosswalks, owner ledgers,
availability registers, parity reports, manifests, and document-QA records.

## Result

No material SCI-ALIGN / SCI-AST contract contradiction was found in the final
r0.3 packets. The packets are therefore returned for scientific-owner freeze
review. This result does not itself approve or freeze either contract.

## Horizontal Findings

| Audit question | Evidence and disposition |
| --- | --- |
| Shared notation | Both packets use `s` for stable ALIGN detector-reference slot, `j` for local storage row only, `n` for stable RTC output sample, `d` for detector occurrence or stable detector identity, and `p` for map pixel. ALIGN no longer uses `p` as a detector index. `x/r` are reserved for paired KID readout coordinates. |
| Shared interface identity | Both scientific bodies name exactly `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`. The two boundary files compare byte-for-byte equal and each has SHA-256 `04357d36b302d607b95950f529044e178deb2528d0c6f656d90da93067a5da36`. The digest is confined to packet-control records rather than the scientific boundary body. |
| Circular topology | Both packages use the same shortest signed interval `[-P/2,P/2)` and make the exactly antipodal case unavailable absent explicit unwrap authority. |
| ALIGN exposure taxonomy | Physical acquired exposure and valid-original exposure are distinct. Original-invalid support may retain nonzero physical acquired exposure and zero valid-original exposure. Synthesized/surrogate and missing/unoccupied support add zero acquired exposure. Later guarded, retained, or use-qualified facts do not rewrite acquisition. AST imports these facts without reinterpretation. |
| Time identity | The ALIGN detector-reference occurrence time is an exact grid/time identity. Boresight and other observing-state fields are evaluated or mapped at that time. Native producer timestamps remain support metadata and do not become competing current-sample times. AST consumes that exact relation without clock reconstruction. |
| Paired readout mapping | ALIGN preserves one source relation, timing relation, weights, and slot identity for paired `x/r`, without cross-coordinate mixing or filling an unavailable coordinate from its mate. AST treats the pair as immutable upstream readout facts rather than coordinate validity. |
| AST spherical oracle | Both AST views define the tangent vector, its norm, unit tangent direction for positive norm, the exponential-map expression, and the zero-norm branch. Derivative and finite-difference predictions cite the complete oracle. |
| Role-factored AST parents | Detector direction, tangent coordinate, continuous pixel, optional nominal pixel, and RTC-output-grid coordinates have extending, dependency-specific parents. Downstream failure does not erase an available upstream fact. |
| RTC-grid ownership | `SCI-AST:rtc_output_grid_coordinates@1` applies to every ordinary science product whose numerical signal is on an RTC output grid. RTC supplies exact parent facts; AST binds them without reconstruction and owns the requested coordinate bundle. Missing RTC parents make only that role unavailable. Phase-zero coincidence does not erase the RTC parent. `AST-OWNER-Q008` is closed for this path. |
| No angular signal filtering | AST expressly forbids applying RTC signal-filter coefficients to angular coordinates as if they were filtered sky response. RTC temporal response remains a parent fact. |
| Geometry representation and count | Each selected geometry artifact declares raw, derotated/horizon-fixed, or another exact representation, the transformations already embodied, and application counts. AST applies only the absent transformation and may not infer or double-derotate. |
| AST/MAP boundary | AST owns continuous astrometric coordinates and their exact parents. MAP owns deposition kernel, normalization, boundary treatment, conservation rule, and realized numerical map contribution. Base AST output contains no general kernel-dependent candidate stencil. Materialized `G_pi` requires an exact MAP-owned request and does not transfer scientific ownership. PTC/producer coefficients, named-use eligibility, calibration, response, and uncertainty retain their named owners. |
| Typed unavailability | Both packets preserve cause-specific, dependency-limited unavailability; unavailable is not zero and one cause does not rewrite another role's facts. Open owner questions block only their named field or claim. |
| Stable identifiers | Exact r0.2-to-r0.3 identifier-list comparison found no renumbering: ALIGN retains 55 requirements, 26 predictions, and 12 assumptions; AST retains 90 requirements, 50 predictions, and 15 assumptions. |
| Formal parity and artifact control | Both formal parity reports cover every directive item. Both durable verifiers pass. ALIGN repeated fixed-epoch builds are byte-identical; AST rebuilds retain the recorded hashes. All 56 final PDF pages were rendered and visually inspected. |

## Reconciliations Closed During Audit

Three packet-control remnants from the copied r0.2 bundles were corrected
without changing scientific authority: auxiliary AST records now consistently
show Q008 closed for the ordinary RTC-grid role; both PDF QA reports carry the
final r0.3 metadata, page counts, and hashes; and the AST boundary proof names
the final shared digest. No stable normative identifier changed.

## Review Handoff

The appropriate next action is scientific-owner freeze review of these exact
r0.3 bytes. Any later modification to a canonical source, shared boundary,
manifested control record, or PDF invalidates the corresponding digest and
requires the affected document-coherence checks to be rerun. Implementation
conformity and observational validation remain separate future activities.
