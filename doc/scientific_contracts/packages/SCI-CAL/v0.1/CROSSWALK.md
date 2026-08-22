# SCI-CAL v0.1 Draft Crosswalk

Status: refreshed for science rationale r0.5 and engineering conformance r0.4,
2026-08-20

The engineering PDF includes the canonical modules under `src/common/`; the
scientist-facing PDF explains and routes that same authority without embedding
the full audit-oriented clauses. The numbered requirements therefore have one
normative engineering home, not parallel restatements. "Scientist-facing
authority" points to the rationale, approved boundary, definition, equation,
or assumption that gives each requirement its scientific meaning.
"Engineering observable" identifies implementation-independent evidence; it
does not map to current source files, functions, classes, configuration, or
tests.
The shared structural state is `science-qualification-eligible`, never an
achieved `science-qualified` or `calibrated-science` claim; SCI-CAL-REQ-049
defines the separate evidence gate for either achieved claim.

| Requirement | Scientist-facing authority | Engineering observable and proposed falsifier |
|---|---|---|
| SCI-CAL-REQ-001 | Scientific boundary; operation definition; SCI-CAL-ASM-002 | Requested/effective channel identity; non-`xs` no-output case, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-002 | Science rationale Sections 5 and 9, Q05; point-source-peak and reference-plane definitions; SCI-CAL-ASM-005 and 010 | Exact target token, plane, 272/214/150 GHz references, retained spectrum convention, and downstream target-color-correction boundary; unsupported-target case, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-003 | Science rationale Sections 1 and 9, Q01; notation table; SCI-CAL-ASM-001 and 004; Eq. `eq:affine-input` | Dimensionless fitted `delta_f/f_res`, positive optical-power direction, no additive CAL operation, Tune validity, and high-noise condition; SCI-CAL-EDGE-028 and 029 |
| SCI-CAL-REQ-004 | Validity and quality tuple; Eq. `eq:validity-tuple` | Enumerated states, including `science-qualification-eligible`, and reason codes; sentinel-injection cases across SCI-CAL-EDGE-006, 010, 022, and 028 |
| SCI-CAL-REQ-005 | Science rationale Appendix B and Q01--Q09; validity discussion; Eq. `eq:reconstruction`; SCI-CAL-ASM-009 | Requested/effective/resolved/realized records plus immutable decided owner snapshot and explicit unavailable products; two-observation replay, SCI-CAL-EDGE-030 |
| SCI-CAL-REQ-006 | Acquisition-occurrence definition; notation table | Observation/Tune plus network/interface and local slot in every key; global-column reorder, SCI-CAL-EDGE-016 |
| SCI-CAL-REQ-007 | Binding definition; SCI-CAL-ASM-001 | Keyed relation or ordered proof with scope/order/cardinality; keyed and ordered permutation cases, SCI-CAL-EDGE-014 and 015 |
| SCI-CAL-REQ-008 | Association-edge definition; Eq. `eq:valid-support` | Single-valued complete selected support; missing/duplicate cases, SCI-CAL-EDGE-017 and 018 |
| SCI-CAL-REQ-009 | Science rationale Section 3 and Figure 1; source/child-APT and identity-layer definitions; SCI-CAL-ASM-003 | Immutable source and observation-specific child artifacts/digests plus local row occurrences and explicit identity-transform state |
| SCI-CAL-REQ-010 | Science rationale Section 3; association-edge and producer--transformer definitions; SCI-CAL-ASM-003 | Both endpoints, method/version, disposition, quality evidence, and TolProj authority without source-factor reinterpretation; SCI-CAL-EDGE-018 |
| SCI-CAL-REQ-011 | Identity-layers definition | Separate acquisition/APT/association/design and occurrence/semantic/byte identities; irrelevant design change, SCI-CAL-EDGE-019 |
| SCI-CAL-REQ-012 | Notation array/network convention; identity and binding definitions | Constraint check, permutation invariance, and observation lifetime; SCI-CAL-EDGE-016, 017, and 030 |
| SCI-CAL-REQ-013 | Science rationale Section 3 and Q03; selected-absolute-factor definition; SCI-CAL-ASM-005; Eq. `eq:multiplier` | Finite nonzero selected-child factor plus complete producer-owned generating record; missing-field, sign, and reciprocal challenges |
| SCI-CAL-REQ-014 | Science rationale Section 3 and Q04; once-only lineage definition; Eq. `eq:embodied-pointing` | TolProj association/child-transform authority, transfer domain, retained systematics, pointing disposition, and ancestry event; SCI-CAL-EDGE-013 |
| SCI-CAL-REQ-015 | Once-only composition definition; Eq. `eq:once-only` | Per-role factor-instance application counts; unity and factor challenges, SCI-CAL-EDGE-011 through 013 |
| SCI-CAL-REQ-016 | Eq. `eq:multiplier` and `eq:signal`; factor interpretation | Elementwise reconstruction and factor-order permutation with invariant value and preserved lineage |
| SCI-CAL-REQ-017 | Relative-responsivity definition | Role/unit/normalization/support record and absence from canonical multiplier; responsivity insertion challenge |
| SCI-CAL-REQ-018 | Sensitivity definition; uncertainty interpretation | `sens` role and exclusions; signal-factor insertion and unavailable-weight cases, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-019 | Opaque-total-factor definition | Exact compatibility decomposition; opaque, omitted, duplicate, and inverted aggregate challenge, SCI-CAL-EDGE-012 |
| SCI-CAL-REQ-020 | Science rationale Figure 1, Sections 3 and 9; producer--transformer--delivery--consumer and canonical-lineage definitions; Eq. `eq:reconstruction` | Resolvable role-separated factor ledger, Q01--Q09 snapshot, photometric/order states, and one package record; deterministic reconstruction and compact-link resolution |
| SCI-CAL-REQ-021 | Atmosphere notation; SCI-CAL-ASM-006; Eq. `eq:los-coordinate` | LMT WVR `tau225` from `tel*.nc`, full sample airmass, model/time support, and zero pivot; SCI-CAL-EDGE-001 through 003 |
| SCI-CAL-REQ-022 | SCI-CAL-ASM-006; Eq. `eq:atmos-domain` | Approximate five-minute WVR cadence, recorded bracketing interpolation, and no endpoint extrapolation; SCI-CAL-EDGE-007 and 008 |
| SCI-CAL-REQ-023 | Science rationale Section 4 and Q06; atmosphere-operator definition; SCI-CAL-ASM-011; Eqs. `eq:operator-nodes` through `eq:operator-orientation` | Exact operator, contract/node digests, fixed AM profile, array/passband assignment, support, and fail-closed behavior |
| SCI-CAL-REQ-024 | Eqs. `eq:operator-nodes` through `eq:operator-orientation` | Broadband trapezoid integration, PCHIP elevation, piecewise-linear opacity, exact zero/nodes, and seam continuity; SCI-CAL-EDGE-001, 004, and 005 |
| SCI-CAL-REQ-025 | Eq. `eq:atmos-invariants` | Positive finite and strict-direction property checks; increasing-opacity and plateau cases, SCI-CAL-EDGE-002 and 003 |
| SCI-CAL-REQ-026 | Eq. `eq:atmos-domain`; SCI-CAL-ASM-006 and 011 | Intersection support and no fallback/clamp/extrapolation; SCI-CAL-EDGE-006 through 010 |
| SCI-CAL-REQ-027 | Validity/quality definition; Eq. `eq:atmos-domain`; claim-layer table | One deterministic CAL class per observation, nominal 0.15 guidance, 0.025 momentary-excursion tolerance, and recorded classifier evidence |
| SCI-CAL-REQ-028 | Validity/quality definition; SCI-CAL-ASM-009 | Supported engineering-observation samples use the same correction operator but carry no science-quality claim; SCI-CAL-EDGE-009 |
| SCI-CAL-REQ-029 | Validity/quality definition; Eq. `eq:atmos-domain` | Affected-sample rejection for negative, non-finite, absent, above-0.25, and out-of-support values without invalidating unrelated supported samples; SCI-CAL-EDGE-006 and 010 |
| SCI-CAL-REQ-030 | Eq. `eq:validity-tuple`; SCI-CAL-ASM-009 | Exactly one class per observation, no science/engineering splitting, and owner-controlled policy; SCI-CAL-EDGE-009 |
| SCI-CAL-REQ-031 | Science rationale Sections 4--5 and Q05; passband and photometric-convention definitions; SCI-CAL-ASM-010 | Exact TolTECA-v1 energy-weighted array-average passband, authorized alpha set/default, no spectral extrapolation, and unavailable detector/network variation |
| SCI-CAL-REQ-032 | SCI-CAL-ASM-007; Eq. `eq:conditional-covariance` | CAL/noise-estimator boundary plus downstream ordered-covariance consistency fixture, SCI-CAL-EDGE-021 |
| SCI-CAL-REQ-033 | Conditional-uncertainty definition; Eq. `eq:variance-weight` | CAL emits no variance/weight; downstream square/inverse-square consistency case, SCI-CAL-EDGE-020 |
| SCI-CAL-REQ-034 | Conditional-uncertainty definition; Eq. `eq:variance-weight` | CAL labels measurement uncertainty downstream/unavailable rather than synthesizing zero/infinite values or `sens`; SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-035 | Nuisance-uncertainty definition; uncertainty interpretation | Six-category completeness ledger with status and correlation scope; remove each category in turn, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-036 | Eqs. `eq:nuisance-covariance` and `eq:common-rank-one` | Within-array common-scale scope, observation-common WVR driver, array-dependent response, and unavailable cross-array covariance; SCI-CAL-EDGE-023 |
| SCI-CAL-REQ-037 | Conditional versus nuisance definitions; Eq. `eq:full-covariance` | Total/significance claim withheld until complete ledger and cross terms; missing-nuisance case, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-038 | SCI-CAL-ASM-007; Eqs. `eq:atmos-derivatives` through `eq:full-covariance` | Future-systematics propagation regime; seam-spanning, large/asymmetric, discrete, and same-data cases, SCI-CAL-EDGE-005 and 024 |
| SCI-CAL-REQ-039 | Science rationale Sections 2 and 9, Q02; Eq. `eq:conditional-covariance`, `eq:variance-weight`, and `eq:companion-transfer`; SCI-CAL-ASM-002 | Exact local multiplier/support/lineage identity and enforced CAL-before-PTC order |
| SCI-CAL-REQ-040 | Science rationale Sections 3 and 5; response-basis and producer definitions; SCI-CAL-ASM-008; Eq. `eq:beam-peak` | Producer-owned Beammap/source APT, selected child APT, template occurrence, and retained ellipse/frame/unit/fit/uncertainty metadata |
| SCI-CAL-REQ-041 | Response-basis definition; SCI-CAL-ASM-008; Eq. `eq:elliptical-solid-angle` | Separate originating and realized response records; labeled circularization case, SCI-CAL-EDGE-027 |
| SCI-CAL-REQ-042 | Point-source-peak definition; Eqs. `eq:beam-peak` and `eq:peak-response` | Unit-peak unresolved-source fixture and explicit renormalization proof, SCI-CAL-EDGE-025 and 026 |
| SCI-CAL-REQ-043 | Response interpretation; Eq. `eq:peak-response`; SCI-CAL-ASM-008 | Before/after kernel response and claim status; SCI-CAL-EDGE-026 |
| SCI-CAL-REQ-044 | Operation and point-source definitions; scientific boundary | Unsupported-unit and photometric-meaning no-output cases, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-045 | Science rationale Sections 7 and 9 and Appendix B; validity/quality definitions; Eq. `eq:validity-tuple` | Complete tuple, reason-code coverage, and machine-distinguishable Q01--Q09 limitations without sentinels |
| SCI-CAL-REQ-046 | Science rationale Tables 3--4; validity/quality definitions; Eq. `eq:valid-support` | Distinct invalid/outside/engineering/uncertainty states; supported engineering signal under the same operator without a science claim |
| SCI-CAL-REQ-047 | Science rationale Sections 3--5 and 9 and Appendix C; canonical-lineage discussion; Eq. `eq:reconstruction` | One package record resolving source/child/delivery identities, generating/transform records, photometric/order states, Q01--Q09 snapshot, atmosphere, uncertainty, and response |
| SCI-CAL-REQ-048 | Canonical-lineage discussion; Eq. `eq:reconstruction` | Product links resolve canonical record, source APT, selected child APT, and owner-decision snapshot without copied dense tables |
| SCI-CAL-REQ-049 | Science rationale Sections 5, 8, and 9; claim-layer section and table | Separate structural, atmosphere-representation, same-Beammap closure, independent transfer, repeatability, absolute-performance, and owner-acceptance statuses |
| SCI-CAL-REQ-050 | Science rationale Section 8 and Q09; proposed-evidence section; SCI-CAL-ASM-011 | Concrete Beammap/APT/library/TolProj closure and associated-pointing transfer records by array; no arbitrary matrix/sample minimum; benchmarks are reporting values and final acceptance is owner-owned |

## Coverage statement

- Requirements present in the shared authority: SCI-CAL-REQ-001 through SCI-CAL-REQ-050.
- Requirements represented in this crosswalk: SCI-CAL-REQ-001 through SCI-CAL-REQ-050.
- Engineering-only normative requirements: none.
- Current implementation mappings: intentionally none.
- Scientific validation executed during authorship: none.
