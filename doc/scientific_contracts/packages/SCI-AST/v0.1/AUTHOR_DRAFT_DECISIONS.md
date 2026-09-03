# SCI-AST v0.1 — Stage B Author Draft Decisions

Status: implementation-blind Stage B author record; not scientific approval,
conformity, validation, freeze, or readiness

Prepared: `2026-08-22`

This record captures every author-introduced choice, question, packet
inconsistency disposition, and unavailable claim encountered while producing
the canonical draft from the approved seven-logical-item packet. It does not
replace the exact owner-decision register.

## Packet Verification And Isolation

Before authorship, every approved item was checked against
`AUTHOR_PACKET_MANIFEST.md`:

| Approved item | Verified SHA-256 |
| --- | --- |
| `SCOPE_BRIEF.md` | `73114320f36996a606392294617cb6b075155fa47a6eec22744c23a831678287` |
| `AUTHOR_DECISIONS_AND_OPEN_QUESTIONS.md` | `25bf534a4caaccd663a27630c9a7be83fbcdbf6b037f0f9d225da4c3ec9a891b` |
| `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` | exact r0.3 digest recorded in `SOURCE_MANIFEST.md` and boundary equality report |
| `SCI-RTC_TO_SCI-AST_SAMPLE_GRID_BOUNDARY.md` | `87dc8c7d82e1940df8857fb00886747cd5da72db88423bff96c3f574d2378ccd` |
| `DETECTOR_GEOMETRY_FIELD_ROTATION_BOUNDARY.md` | `249dc7e66aab261b46a64e85778819b98728afb444e3f951ad18ce5fed7a0515` |
| `AUTHOR_SUPERSESSION_COVER.md` | `faf45024232bbfb0a531d02fca9d1b4682990aaf5d4a6494249093241b8eb9cf` |
| frozen core at `e3553bc0fcaa158ed4d986f59e9f25e5e2eeac7a:doc/audits/packages/SCI-AST-001_INDEPENDENT_CORE.tex` | `ed1fe3bf68ed53974b8c910bd3824717eb0cf5ff11d0b27c0fdf27aa6e606276` |
| `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `af86869d8f95f3704b7544c1b692acc1da000af4b00575d4915737483d97ed6e` |

No implementation, schema, test, audit report, repair, validation evidence,
other scientific package, web source, or unlisted repository material was
consulted for scientific content.

For the Stage B r0.3 targeted revision, the controlling scientific directive was
the supplied `pasted-text.txt` with SHA-256
`a947ab55eab404f7740d47d3c87766a77f4714bd6ce5c6c9b578c6dc5aff9c0f`.
The final shared boundary installed by the ALIGN-led coordinator is
`SCI-ALIGN_TO_SCI-AST v0.1/r0.1`; its exact digest is recorded in the package
manifest and equality report.
It carries the r0.3 shared exposure/time amendment and is byte-identical to
the ALIGN package copy.

## Author Choices

| Author ID | Choice | Reason and limit |
| --- | --- | --- |
| `SCI-AST-AUTH-D001` | Name the requested exact owner-decision artifact `OWNER_DECISION_REGISTER.md`. | Administrative filename supplied by the Stage B coordinator; it changes no scientific content. |
| `SCI-AST-AUTH-D002` | Make `src/common/notation.tex`, `definitions.tex`, `equations.tex`, `assumptions.tex`, `requirements.tex`, and `edge_cases.tex` the only canonical scientific modules, and have both audience documents input all six. | Ensures the scientist and engineer documents are two views of one authority rather than divergent copies. |
| `SCI-AST-AUTH-D003` | Assign `SCI-AST-REQ-001` through `SCI-AST-REQ-090` sequentially and reserve later numbers for append-only successors. | Provides stable, granular, crosswalkable obligations without renumbering approved owner decisions. |
| `SCI-AST-AUTH-D004` | Assign `SCI-AST-PRED-001` through `SCI-AST-PRED-050` sequentially and `SCI-AST-ASM-001` through `SCI-AST-ASM-015`. | Separates falsifiable future truth conditions from bounded assumptions and normative requirements. |
| `SCI-AST-AUTH-D005` | Use the exact boundary profile identities/revisions `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`, `SCI-RTC_TO_SCI-AST_SAMPLE_GRID v0.1/r0.1`, and `SCI-AST_DETECTOR_GEOMETRY_FIELD_ROTATION v0.1/r0.1` as the compatibility anchors. | Similar field names, shapes, cadence, or sample count do not establish compatibility. |
| `SCI-AST-AUTH-D006` | Represent selected geometry by typed operators `G_gamma`/`C_gamma` (rendered as `\mathcal G_\gamma`/`\mathcal C_\gamma`) carrying gauge, pivot, affine limitations, support, and lineage. | Avoids inventing a universal pure known-pivot rotation while still expressing the ordered composition. |
| `SCI-AST-AUTH-D007` | Retain the core half-open nearest-center nominal pixel rule as the v0.1 named rule, while permitting only a separately versioned rule with equivalent boundary tests. | Preserves the admitted core equation and the packet requirement for an exact rounding convention; does not transfer MAP deposition policy to AST. |
| `SCI-AST-AUTH-D008` | Use a typed availability vocabulary with `available`, `available_conditional`, `unavailable_input`, `unavailable_authority`, `unavailable_unsupported`, `not_applicable`, and `not_persisted_standard`, always with an exact reason. | Makes owner questions and absent terms explicit. Producers may refine these states but may not turn unavailable into zero. |
| `SCI-AST-AUTH-D009` | Treat exact spherical composition as mathematical truth/offline oracle and the established small-angle relation as the conditional production relation. | Applies the binding supersession cover without creating an automatic runtime fallback or numerical threshold. |
| `SCI-AST-AUTH-D010` | Treat exact packet approval as activating `AST-SCOPE-D001`–`AST-SCOPE-D027` for Stage B authorship while preserving their admitted text verbatim. | The approved manifest records the condition that the older Stage A files described as pending; this does not approve the Stage B draft. |
| `SCI-AST-AUTH-D011` | Present a concise 8–10 page scientist narrative with three explanatory figures, while the Engineering Conformance Specification alone renders all six canonical formal modules in full. | Audience and navigation choice only; the engineering view remains the complete formal authority with all equations, 90 requirements, and 50 predictions. |
| `SCI-AST-AUTH-D012` | Do not add a new scientific-owner question. | The approved packet is sufficient to define a conditional authority: its eight typed owner questions already make every unresolved interface or quantitative claim unavailable without supplying a default. |
| `SCI-AST-AUTH-D013` | Use `s` for stable ALIGN slot `(observation,s)`, `j` only for local storage row, and `n` for stable RTC output sample; replace canonical `theta^A_dj` with `theta^A_ds`. | Applies the targeted directive and prevents storage layout from becoming external identity. |
| `SCI-AST-AUTH-D014` | Reserve `x/r` for paired KID readout coordinates, use `u_sky` for unit direction, and `zeta_1,zeta_2` for TAN axes. | Removes the prior readout/sky/TAN symbol collision without changing equations' scientific meaning. |
| `SCI-AST-AUTH-D015` | Factor direction, tangent, pixel, nominal-pixel, and RTC parent identities and their atomic validity records by dependency. | Preserves dependency-limited availability while allowing product-family atomic failure. |
| `SCI-AST-AUTH-D016` | Reserve `G_pi` for a complete exact MAP-owned deposition request and end base pre-MAP AST output at continuous pixel, optional nominal containing pixel, and bounds state. | Removes the general r0.2 candidate-stencil artifact; kernel-dependent neighborhood or support requires the exact MAP plan. |

## Bounded Horizontal-Audit Dispositions

These are traceability/editorial dispositions, not new scientific decisions.
No audit report or new external scientific input was opened.

| Disposition ID | Exact bounded correction | Applied disposition | Scientific limit |
| --- | --- | --- | --- |
| `SCI-AST-AUDIT-DISP-001` | Expand the shared ALIGN-boundary packet-input crosswalk to expose its direct downstream realizations rather than relying only on later inventory rows. | The `SCI-ALIGN_TO_SCI-AST_BOUNDARY.md` row in `CROSSWALK.md` now directly traces `REQ-006`–`022`, `REQ-056`–`057`, `REQ-073`, `REQ-080`–`088`, and their primary predictions. | No requirement, prediction, ownership boundary, or scientific meaning changed. |
| `SCI-AST-AUDIT-DISP-002` | Restate the exact detector-reference and nominal-slot/time meaning in the shared canonical authority. | `src/common/definitions.tex` now states that detector-reference is the selected detector-stream reference interface, clock relation and grid, never a reference detector, and that nominal slot/time alone does not prove a physical event, physical integration or acquired exposure. | This is a self-contained restatement of the existing exact ALIGN import; no requirement was created or renumbered. |
| `SCI-AST-AUDIT-DISP-003` | Replace wording that could imply raw KID `x` is Stokes I with the shared ordinary-path terminology. | `SCI-AST-ASM-014`, the frame/product-family definition, scientist narrative, `SCI-AST-PRED-047`, crosswalk, change/availability maps, and owner-ledger annotation now state that only the ordinary nonpolarimetric coordinate path is admitted; optional HWPR timing remains only an ALIGN-parent fact; AST authorizes no demodulation, polarization calibration/response, or Stokes reconstruction; raw KID `x` is not Stokes I. | Editorial horizontal-audit clarification only. All stable IDs and scientific boundaries are preserved; no polarization interpretation or new owner decision is introduced. |

## Packet Inconsistencies And Literal Dispositions

| Inconsistency or tension | Disposition used in the draft |
| --- | --- |
| Approved packet inputs retain stale “pending approval” or “Stage B not authorized” prose, while the corrected manifest records exact approval and authorization on `2026-08-21`. | The manifest governs packet-control status. The stale prose is treated as historical state embedded in the approved bytes, not as a reversal of the later control record. No packet content bytes were edited. |
| The independent core uses `D <= D_min` with a fixed positive-domain proposal. | Binding decisions `SCI-AST-001-D002` and the supersession cover govern: the mathematical domain is finite `D > 0`, with no universal positive `D_min`; a separately preregistered footprint bound may be tighter. |
| The independent core describes exact spherical composition as production authority. | Binding decision `SCI-AST-001-D007` governs: exact spherical composition is the offline oracle; the established small-angle production relation is admitted only within a preregistered adequate envelope, with no automatic fallback. |
| The independent core requires stable UID joins. | Binding decision `SCI-AST-001-D001` governs: either a proven observation-local row relation or explicit keyed binding is admissible; bare row coincidence remains invalid and design identity is optional. |
| The independent core includes broad response/covariance derivatives. | Binding decisions `SCI-AST-001-D006` and `AST-SCOPE-D021` govern: v0.1 publishes only the typed map-center astrometric Jacobian and available map-center terms, after a family requests exact domain/codomain. |
| The independent core’s AST eligibility language can be read as a global decision or as letting signal validity control coordinates. | `AST-SCOPE-D004` and `AST-SCOPE-D023` govern: occurrence identity, signal validity, ALIGN validity, AST validity, and use-specific eligibility are distinct with no rescue or precedence rule; finite `x/r` is not required for coordinates. |
| The independent core permits an automatically selected or data-derived center. | `AST-SCOPE-D006` and `AST-SCOPE-D019` govern: an upstream authority selects the physical center; AST realizes it and never searches, fits, derives, recenters, or defaults. |
| The independent core names TolTECA directly as pointing-support authority, while the revised packet generalizes producer identity. | The revised boundary governs: the exact named pointing-support producer selects records/native support; TolTECA may be that producer only when explicitly identified per instance. |
| The core’s provisional consumer/status, application/audit identity, and production prose are not scientific content of the library package. | Excluded under the supersession cover. No implementation, audit, validation, or readiness statement was imported. |
| Exact spherical equations use a simple tangent-displacement detector composition, while the geometry boundary forbids assuming universal pure known-pivot rotation or resolved affine terms. | The exact exponential map is retained as oracle/limiting case; canonical composition uses the selected typed geometry relation with gauge, pivot, affine limitations, support, and lineage. |
| The TAN/WCS “projection” term could be confused with MAP sample deposition and `G_pi`. | Definitions and requirements reserve TAN/WCS astrometric projection for AST and sample-to-pixel deposition/gridding for MAP. Base AST output contains coordinates/optional nominal pixel/bounds only; exact `G_pi` materialization is a delegated service only for a complete MAP-owned request with all parents. |

## Preserved Owner Questions And Blocked Claims

No answer was inferred for `AST-OWNER-Q001`–`AST-OWNER-Q007`. Their exact text
and states are preserved in `OWNER_DECISION_REGISTER.md`. `AST-OWNER-Q008` is
closed by the r0.3 owner directive for the ordinary RTC-grid science chain.
The resulting unavailable outputs/claims are:

| Question | Unavailable output or claim until answered |
| --- | --- |
| `AST-OWNER-Q001` | Unconditional aligned telescope/boresight/observing-state field admission or equivalence. |
| `AST-OWNER-Q002` | Unconditional physical-center provenance by Point, OOF, Beammap, or Science family. |
| `AST-OWNER-Q003` | A complete family-specific map-center uncertainty statement. |
| `AST-OWNER-Q004` | Quantitative achieved adequacy for small-angle geometry, time representation, or large coordinate-array precision. |
| `AST-OWNER-Q005` | Any MAP-003-specific retained-WCS/crop/dual-axis AST interface. |
| `AST-OWNER-Q006` | Unconditional observation-specific geometry association, pointing transfer, or cross-realization equivalence. |
| `AST-OWNER-Q007` | Any available family-specific `astrometric_map_center_jacobian`. |
| `AST-OWNER-Q008` | Closed for the ordinary chain: `SCI-AST:rtc_output_grid_coordinates@1` applies whenever the numerical science signal is represented on an RTC output grid. Missing exact RTC parents make only that role unavailable. |

## Other Explicitly Unavailable Claims

The Stage B author draft does not claim:

- scientific-owner approval of the Stage B scientific content;
- implementation or schema conformity;
- representation fidelity or compatibility with any current product;
- successful numerical, unit, property, round-trip, derivative, covariance,
  failure-injection, real-observation, or Unity validation;
- achieved small-angle, time, coordinate-array, WCS, or astrometric precision;
- observational performance or absence of regressions;
- contract freeze or compatibility of a successor revision;
- production readiness or authorization;
- MAP-003 disposition, dense covariance, polarimetry, Galactic simulation,
  general reprojection, map accumulation, source fitting, calibration,
  photometry, beam inference, or downstream eligibility policy.

Document compilation, durable structural verification, Poppler rendering, and
page-by-page visual inspection are document-production checks only. They do
not change any unavailable scientific or engineering claim above.
