# SCI-CAL v0.1 Draft Crosswalk

Status: implementation-blind author draft, 2026-08-16

The scientist-facing and engineering-facing PDFs include the same files under
`src/common/`. The numbered requirements below are therefore one authority,
not parallel restatements. "Scientist authority" points to the rationale,
definition, equation, or assumption that gives each requirement its scientific
meaning. "Engineering observable" identifies implementation-independent
evidence; it does not map to current source files, functions, classes, or tests.
The shared structural state is `science-qualification-eligible`, never an
achieved `science-qualified` or `calibrated-science` claim; SCI-CAL-REQ-049
defines the separate evidence gate for either achieved claim.

| Requirement | Scientist-facing authority | Engineering observable and proposed falsifier |
|---|---|---|
| SCI-CAL-REQ-001 | Scientific boundary; operation definition; SCI-CAL-ASM-002 | Requested/effective channel identity; non-`xs` no-output case, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-002 | Point-source-peak and reference-plane definitions; SCI-CAL-ASM-005; Eq. `eq:multiplier` | Exact target token, plane, and response meaning; unsupported-target case, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-003 | Notation table; SCI-CAL-ASM-001 and 004; Eq. `eq:affine-input` | Typed signal declaration and baseline disposition; missing-baseline and affine cases, SCI-CAL-EDGE-029 |
| SCI-CAL-REQ-004 | Validity and quality tuple; Eq. `eq:validity-tuple` | Enumerated states, including `science-qualification-eligible`, and reason codes; sentinel-injection cases across SCI-CAL-EDGE-006, 010, 022, and 028 |
| SCI-CAL-REQ-005 | Validity discussion; Eq. `eq:reconstruction`; SCI-CAL-ASM-009 | Requested/effective/resolved/realized records; two-observation order replay, SCI-CAL-EDGE-030 |
| SCI-CAL-REQ-006 | Acquisition-occurrence definition; notation table | Observation/Tune plus network/interface and local slot in every key; global-column reorder, SCI-CAL-EDGE-016 |
| SCI-CAL-REQ-007 | Binding definition; SCI-CAL-ASM-001 | Keyed relation or ordered proof with scope/order/cardinality; keyed and ordered permutation cases, SCI-CAL-EDGE-014 and 015 |
| SCI-CAL-REQ-008 | Association-edge definition; Eq. `eq:valid-support` | Single-valued complete selected support; missing/duplicate cases, SCI-CAL-EDGE-017 and 018 |
| SCI-CAL-REQ-009 | Measured-APT row occurrence and identity-layer definitions; SCI-CAL-ASM-003 | Immutable artifact identity/digest plus local row/source occurrence; artifact or row identity removal case |
| SCI-CAL-REQ-010 | Association-edge definition; SCI-CAL-ASM-003 | Both endpoints, method/version, disposition, and quality evidence; all dispositions, SCI-CAL-EDGE-018 |
| SCI-CAL-REQ-011 | Identity-layers definition | Separate acquisition/APT/association/design and occurrence/semantic/byte identities; irrelevant design change, SCI-CAL-EDGE-019 |
| SCI-CAL-REQ-012 | Notation array/network convention; identity and binding definitions | Constraint check, permutation invariance, and observation lifetime; SCI-CAL-EDGE-016, 017, and 030 |
| SCI-CAL-REQ-013 | Selected-absolute-factor definition; SCI-CAL-ASM-005; Eq. `eq:multiplier` | Finite nonzero oriented factor record with recipient, plane, uncertainty, and lineage; sign/reciprocal challenge |
| SCI-CAL-REQ-014 | Once-only lineage definition; Eq. `eq:embodied-pointing` | Pointing disposition and ancestry event; corrected-APT challenge, SCI-CAL-EDGE-013 |
| SCI-CAL-REQ-015 | Once-only composition definition; Eq. `eq:once-only` | Per-role factor-instance application counts; unity and factor challenges, SCI-CAL-EDGE-011 through 013 |
| SCI-CAL-REQ-016 | Eq. `eq:multiplier` and `eq:signal`; factor interpretation | Elementwise reconstruction and factor-order permutation with invariant value and preserved lineage |
| SCI-CAL-REQ-017 | Relative-responsivity definition | Role/unit/normalization/support record and absence from canonical multiplier; responsivity insertion challenge |
| SCI-CAL-REQ-018 | Sensitivity definition; uncertainty interpretation | `sens` role and exclusions; signal-factor insertion and unavailable-weight cases, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-019 | Opaque-total-factor definition | Exact compatibility decomposition; opaque, omitted, duplicate, and inverted aggregate challenge, SCI-CAL-EDGE-012 |
| SCI-CAL-REQ-020 | Factor-instance and canonical-lineage definitions; Eq. `eq:reconstruction` | Resolvable factor ledger and one package record; deterministic reconstruction and compact-link resolution |
| SCI-CAL-REQ-021 | Atmosphere notation; SCI-CAL-ASM-006; Eq. `eq:los-coordinate` | Zenith opacity, full airmass, model/time support, and zero pivot; SCI-CAL-EDGE-001 through 003 |
| SCI-CAL-REQ-022 | SCI-CAL-ASM-006; Eq. `eq:atmos-domain` | Endpoint/method/gap/validity record; bracketed and unbracketed cases, SCI-CAL-EDGE-007 and 008 |
| SCI-CAL-REQ-023 | Structural-operator definition; SCI-CAL-ASM-011; Eqs. `eq:operator-nodes` through `eq:operator-orientation` | Content digest, nodes, orientation, support, model and passband record; omission of each required field yields unavailable |
| SCI-CAL-REQ-024 | Eqs. `eq:operator-nodes` through `eq:operator-orientation` | Exact zero/node values, authoritative-ordinate interpolation, and seam continuity; SCI-CAL-EDGE-001, 004, and 005 |
| SCI-CAL-REQ-025 | Eq. `eq:atmos-invariants` | Positive finite and strict-direction property checks; increasing-opacity and plateau cases, SCI-CAL-EDGE-002 and 003 |
| SCI-CAL-REQ-026 | Eq. `eq:atmos-domain`; SCI-CAL-ASM-006 and 011 | Intersection support and no fallback/clamp/extrapolation; SCI-CAL-EDGE-006 through 010 |
| SCI-CAL-REQ-027 | Validity/quality definition; Eq. `eq:atmos-domain`; claim-layer table | Segment wholly within 0 through 0.15 and otherwise complete may be only `science-qualification-eligible`; SCI-CAL-EDGE-001 verifies no promotion to achieved qualification |
| SCI-CAL-REQ-028 | Validity/quality definition; SCI-CAL-ASM-009 | Engineering-opacity segment produces truthful no-output state; SCI-CAL-EDGE-009 |
| SCI-CAL-REQ-029 | Validity/quality definition; Eq. `eq:atmos-domain` | Negative, non-finite, absent, above-0.25, and out-of-support cases; SCI-CAL-EDGE-006 and 010 |
| SCI-CAL-REQ-030 | Eq. `eq:validity-tuple`; SCI-CAL-ASM-009 | One class per predeclared segment and explicit split lineage; boundary-crossing segment, SCI-CAL-EDGE-009 |
| SCI-CAL-REQ-031 | Passband-reference definition; SCI-CAL-ASM-010 | Exact passband-set ID and unknowns ledger; attempts to infer measurement, normalization, uncertainty, or weighting are rejected |
| SCI-CAL-REQ-032 | SCI-CAL-ASM-007; Eq. `eq:conditional-covariance` | Ordered dense covariance fixture with off-diagonal preservation, SCI-CAL-EDGE-021 |
| SCI-CAL-REQ-033 | Conditional-uncertainty definition; Eq. `eq:variance-weight` | Scalar units and square/inverse-square scaling; signed-factor case, SCI-CAL-EDGE-020 |
| SCI-CAL-REQ-034 | Conditional-uncertainty definition; Eq. `eq:variance-weight` | Unavailable state rather than zero/infinite value or undeclared `sens` estimate; SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-035 | Nuisance-uncertainty definition; uncertainty interpretation | Six-category completeness ledger with status and correlation scope; remove each category in turn, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-036 | Eqs. `eq:nuisance-covariance` and `eq:common-rank-one` | Dense detector/array/observation/cohort/global covariance fixtures; sample-count challenge, SCI-CAL-EDGE-023 |
| SCI-CAL-REQ-037 | Conditional versus nuisance definitions; Eq. `eq:full-covariance` | Total/significance claim withheld until complete ledger and cross terms; missing-nuisance case, SCI-CAL-EDGE-022 |
| SCI-CAL-REQ-038 | SCI-CAL-ASM-007; Eqs. `eq:atmos-log-slope` through `eq:full-covariance` | Declared propagation regime; seam-spanning, large/asymmetric, discrete, and same-data cases, SCI-CAL-EDGE-005 and 024 |
| SCI-CAL-REQ-039 | Eq. `eq:conditional-covariance`, `eq:variance-weight`, and `eq:companion-transfer`; SCI-CAL-ASM-002 | Exact multiplier/support/lineage identity across signal, uncertainty, injection, noise, and Jacobian; distinct measured-channel rejection |
| SCI-CAL-REQ-040 | Response-basis definition; SCI-CAL-ASM-008; Eq. `eq:beam-peak` | Selected source/template occurrence and retained ellipse/frame/unit/fit/uncertainty metadata |
| SCI-CAL-REQ-041 | Response-basis definition; SCI-CAL-ASM-008; Eq. `eq:elliptical-solid-angle` | Separate originating and realized response records; labeled circularization case, SCI-CAL-EDGE-027 |
| SCI-CAL-REQ-042 | Point-source-peak definition; Eqs. `eq:beam-peak` and `eq:peak-response` | Unit-peak unresolved-source fixture and explicit renormalization proof, SCI-CAL-EDGE-025 and 026 |
| SCI-CAL-REQ-043 | Response interpretation; Eq. `eq:peak-response`; SCI-CAL-ASM-008 | Before/after kernel response and claim status; SCI-CAL-EDGE-026 |
| SCI-CAL-REQ-044 | Operation and point-source definitions; scientific boundary | Unsupported-unit and photometric-meaning no-output cases, SCI-CAL-EDGE-028 |
| SCI-CAL-REQ-045 | Validity/quality definitions; Eq. `eq:validity-tuple` | Complete tuple plus reason-code coverage for every named cause across edge-case inventory |
| SCI-CAL-REQ-046 | Validity/quality definitions; Eq. `eq:valid-support` | Distinct no-output and uncertainty-only states; SCI-CAL-EDGE-006, 009, 010, 018, 022, and 028 |
| SCI-CAL-REQ-047 | Canonical-lineage discussion; Eq. `eq:reconstruction` | One package record resolving every listed identity, factor, atmosphere, uncertainty, state, and response field |
| SCI-CAL-REQ-048 | Canonical-lineage discussion; Eq. `eq:reconstruction` | Product links resolve record and selected APT without copied dense tables; broken-link and wrong-occurrence cases |
| SCI-CAL-REQ-049 | Three-claim-layer section and table | Independent structural, representation, repeatability, and absolute-performance statuses; reject promotion of eligibility unless separately accepted representation-fidelity and observational-performance evidence satisfies the declared threshold |
| SCI-CAL-REQ-050 | Claim-layer section; proposed evidence section; SCI-CAL-ASM-011 | Preregistered equation/assumption/edge evidence; skipped evidence leaves achieved science qualification unavailable, and provisional targets are not guarantees |

## Coverage statement

- Requirements present in the shared authority: SCI-CAL-REQ-001 through SCI-CAL-REQ-050.
- Requirements represented in this crosswalk: SCI-CAL-REQ-001 through SCI-CAL-REQ-050.
- Engineering-only normative requirements: none.
- Current implementation mappings: intentionally none.
- Scientific validation executed during authorship: none.
