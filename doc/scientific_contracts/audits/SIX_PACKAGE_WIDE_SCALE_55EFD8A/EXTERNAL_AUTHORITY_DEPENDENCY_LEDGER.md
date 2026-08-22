# External Authority Dependency Ledger

Status: **non-authoritative audit draft**
Pinned scientific-library revision: `55efd8a54464636a24e621f6d1b60486d235b20e`
Scope: external and cross-package authorities required to close the
six-package-to-MAP handoff; no assertion that a named external authority
currently exists or conforms

## 1. Reading rule

This ledger distinguishes three different conditions:

1. **External authority required** — RTC or MAP needs a fact whose scientific
   meaning must be supplied by a named producer, registry, artifact, or owner
   outside the consuming package.
2. **Package-internal open gate** — the package's own current owner ledger
   leaves a decision unresolved. More external data alone cannot close it.
3. **Evidence not admitted here** — an authority may exist, but it was outside
   this clean-room source set and therefore was not verified.

These classes SHALL NOT be collapsed into “missing implementation.” A
scientific-owner decision cannot be replaced by observed software behavior,
and an external scientific authority cannot be fabricated by a consumer
default.

ALIGN, AST, RTC, CAL, PTC, and VAL are primary audited packages, not external
authorities. Rows involving them are retained only where a missing exact
cross-package boundary, source-current Registry binding, owner disposition, or
route profile must be distinguished from a genuinely outside fact. Their
current normative cores were inspected; the disposition column says exactly
what was and was not admitted.

Severity terms:

- **BLOCKER** — no numerical MAP route or no scientifically complete handoff;
- **HIGH** — a restricted route may exist, but the named claim or consumer
  cannot proceed;
- **MEDIUM** — provenance/representation or future-scope authority remains
  incomplete without changing the ordinary numerical result already admitted.

## 2. External authority dependencies and cross-package closure gates

| ID | Authority / artifact | Exact fact required | Consumer(s) | Authority state | Source binding at pinned revision | Consequence while absent | Blocked scope | Owner | Finding class | Locked consumer anchors |
|---|---|---|---|---|---|---|---|---|---|---|
| EXT-001 | TEL clock/event authority | Time scale, native sample-time occurrence, cadence, gaps/discontinuities, event semantics, requested versus realized timing, validity, uncertainty, and failure | ALIGN, AST, RTC, CAL, PTC, VAL, MAP provenance | External owner named by role, exact artifact unnamed | **Absent** from admitted packet | Time coincidence cannot establish identity; time-dependent coordinate/calibration facts remain under-bound | Main identity/coordinate chain where these facts are required | TEL timing scientific owner | EXTERNAL AUTHORITY REQUIRED | ALIGN `SCI-ALIGN-REQ-001--003`, `020--022`; RTC `SCI-RTC-REQ-003`, `028--029`, `084`, `115`; MAP `SCI-MAP-REQ-003`, `005`, `043`, `046` |
| EXT-002 | TEL field/pointing and Earth-orientation/refraction authority | Observation/field identity, pointing-support selection, requested/realized boresight, source/target center, frame/equinox, EOP/refraction model, validity, uncertainty, and cause-specific failure | AST, CAL, MAP; generic SRC/Pointing/OOF/BEAM consumers | External owners named by role; exact product chain unnamed | **Absent** from admitted packet | AST/MAP may not select pointing, infer a center, or invent placement/frame transforms | Geometry-dependent coordinates and all corresponding MAP/pointing claims | TEL/pointing/frame scientific owners | EXTERNAL AUTHORITY REQUIRED | AST `SCI-AST-REQ-021--022`, `036--053`, `065--072`; MAP `SCI-MAP-REQ-003`, `005`, `043--045` |
| EXT-003 | Detector geometry / APT realization | Array/network/detector occurrence, focal-plane geometry, orientation, validity interval, replacement history, acquisition-column mapping, covariance, and field-rotation realization | ALIGN, AST, RTC, CAL, PTC, VAL, MAP | APT/geometry owner named by role; exact artifact unnamed | **Absent**; exact detector-geometry/field-rotation boundary body also absent | Row/local UID is insufficient; geometry-dependent AST roles stay unavailable | Geometry-dependent coordinate and response products | APT/detector-geometry owner | EXTERNAL AUTHORITY REQUIRED | AST `SCI-AST-REQ-023--034`; RTC `SCI-RTC-REQ-003`, `084`, `115`; MAP `SCI-MAP-REQ-003`, `043` |
| EXT-004 | RTC→AST sample-grid boundary artifact | Exact RTC product/plan/grid, representative ALIGN slot, time, phase/delay, segment, support/response/status, replacement/drop mapping, and failure relation for the AST RTC-grid role | AST, MAP; VAL provenance joins | Primary RTC and AST owners are known; this is **not an external-package fact** | RTC and AST frozen cores were inspected; the exact boundary body/digest is **absent** | Compatible clauses cannot be frozen as one versioned transfer identity | Exact MAP coordinate-parent source closure | RTC and AST scientific owners | IDENTITY OR GRID GAP | RTC `SCI-RTC-REQ-028--029`, `037`, `041`, `086`, `114`; AST `SCI-AST-REQ-074--079` |
| EXT-005 | BEAM/flxscale, nominal-beam, and calibrator/source-model authority | Beam convention/identity, passband dependence, flxscale/calibrator relation, point-source-equivalent scale, response template/basis/normalization, validity, uncertainty, and lineage | CAL, MAP response; generic BEAM/SRC/FLT consumers | External owners named by role; exact artifacts unnamed | **Absent** from admitted packet | `mJy/beam` cannot be strengthened into a realized beam, literal peak, or complete source response | Beam/point-source/complete-response claims; calibrated route if nominal-beam authority is required | BEAM/CAL/calibrator scientific owners | EXTERNAL AUTHORITY REQUIRED | CAL `SCI-CAL-REQ-001--003`, `013`, `039--044`; MAP `SCI-MAP-REQ-002`, `008`, `014`, `017`, `022`, `044`, `050` |
| EXT-006A | Required CAL atmosphere/passband/WVR input realization | Every requested multiplicative/additive input, sign/unit, observation/time/field applicability, validity, and failure needed by the selected CAL operator | CAL; MAP calibrated signal | CAL transformation authority was inspected; outside observational-input owners are named only by role | Exact CAL equations/requirements **present**; exact observation input artifacts **absent** | A requested required factor/operator cannot be defaulted; the affected calibrated signal route fails | Main signal only when the selected CAL route requires the missing input | CAL owner plus atmosphere/passband/WVR input owners | EXTERNAL AUTHORITY REQUIRED | CAL `SCI-CAL-REQ-003`, `015--016`, `021--024`, `034--038`, `045--046`; MAP `SCI-MAP-REQ-002`, `004` |
| EXT-006B | CAL nuisance/total-uncertainty authority | Calibration, atmosphere, passband, WVR, cross-array, model and correlation terms not required to form a limited finite signal but required for total uncertainty or stronger response/photometric claims | MAP uncertainty/response; downstream photometric consumers | CAL core explicitly preserves omissions; outside term owners are partly unnamed | CAL source **present**; numerical nuisance/correlation sources **absent or typed unavailable** | Missing terms remain unavailable, not zero | Total uncertainty, significance, precision, and stronger response/photometric claims; not every finite-signal route | Each nuisance scientific owner; CAL owns its declared conditional result | UNCERTAINTY GAP | CAL `SCI-CAL-REQ-032--038`, `039--043`; MAP `SCI-MAP-REQ-019--024` |
| EXT-007 | CAL→MAP route/use profile | Exact ordinary-`xs` producer route, active unit, calibration facts, response basis, lineage, disabled/unavailable behavior, and separation of engineering facts from owner policy | VAL, MAP, calibrated-map consumers | Cross-package route/use owners required; CAL alone is not the owner | CAL primary source was inspected; complete MAP-use/direct-route profile is **absent** | CAL semantic output cannot by itself authorize MAP admission or a PTC bypass | Ordinary map admission and any direct CAL→MAP route | MAP use owner plus route, coefficient, policy, and CAL meaning owners | PROFILE REGISTRY GAP | RTC `SCI-RTC-REQ-103`; CAL `SCI-CAL-REQ-001--005`, `039--049`; MAP `SCI-MAP-REQ-002`, `004`, `008`, `047`, `050` |
| EXT-008 | Source identity/model and provenance registry | Target/source/field identity, source/target center, association method, calibrator/source model, observation/acquisition parent, edits/replacements, validity, and uncertainty | TEL, CAL, MAP; generic SRC/Pointing/OOF/BEAM consumers | External registry/model owners named by role; exact artifacts unnamed | **Absent** from admitted packet | Labels/containers cannot establish source identity, center, model, or calibration association | Source/calibrator/pointing claims and any route requiring those associations | Observation/source/model registry owners | EXTERNAL AUTHORITY REQUIRED | CAL `SCI-CAL-REQ-003`, `006--012`, `021--024`; MAP `SCI-MAP-REQ-003`, `043`, `046`, `049--050` |
| EXT-009 | PTC coefficient/covariance → MAP closure | Conditioned-sample identity; exact `omega_i` family/unit/normalization/support/lifecycle/factors/statistical status; covariance domain; disabled-route disposition | VAL, MAP estimator and uncertainty consumers | PTC is a primary audited package; MAP/use owners are also required | PTC frozen core **present**, but F-001/F-002/F-011 conflicts, coefficient owner decision, and profiles are open | A numerical coefficient or transformed sample cannot be certified as the exact MAP member | Ordinary numerical MAP route | PTC owner plus MAP coefficient/use/profile owners | NORMATIVE INTERNAL CONFLICT | PTC `SCI-PTC-REQ-001`, `010--017`, `047`, `052--069`, `076--088`; MAP `SCI-MAP-REQ-006`, `010`, `019--024` |
| EXT-010 | VAL Core/Registry/use-owner closure | Source-current producer facts and causes; exact object/use proposition; policy owner; immutable profile; evaluation/reasons/knowledge state; consumer-owned action | PTC named uses, RTC donor/replacement consumers, MAP admission/coadd | Producers, named use owners, Registry, VAL evaluator, and consumers have distinct authority | Exact VAL packet was **inspected**; bindings are stale, one registered row incomplete, PTC/MAP profiles reserved, aggregate profile absent | Requested decisions are unavailable; they are not implicit exclusions and no other use can substitute | PTC numerical actions, MAP admission, and policy-authorized coadd | Producer and use owners; VAL Registry binds and Core evaluates only | PROFILE REGISTRY GAP | VAL `SCI-VAL-REQ-001--010`, `019--035`, `044--049`; MAP `SCI-MAP-REQ-004`, `010`, `025--035` |
| EXT-011 | MAP grid/projection request and future `G_{pi}` materializer | Authorized WCS/grid/pixel basis/extent, projection class, normalization/conservation, boundary loss, materializer, validity, and exact request→artifact relation | MAP signal/response/covariance/hits/exposure/support | MAP owns the request and estimator semantics; materializer/authority unresolved | MAP primary source **present**; OD-008 **OPEN**; no bound materialized relation | No contribution set, normalization, response, covariance, support, or numerical map can be formed | Every numerical MAP route | MAP scientific owner, then named projection producer | OWNER DECISION REQUIRED | AST `SCI-AST-REQ-080--083`; MAP `SCI-MAP-REQ-005`, `010--021`, `025--030`; `SCI-MAP-OD-008` |
| EXT-012 | MAP common-grid preparation / future reprojection authority | Whether canonical crop/pad is allowed and who owns it; operator, response/covariance/validity/provenance effects; future reprojection/mosaic owner | MAP coadd and future mosaic consumers | MAP decision owner named; future transform owner unnamed | MAP primary source **present**; OD-009 **OPEN** | Incompatible bundles are rejected; no crop/pad/reprojection/interpolation/mosaic may be inferred | Incompatible-grid coadd and future grid-changing products, not an otherwise compatible single-observation map | MAP scientific owner and future transform owner | OWNER DECISION REQUIRED | MAP `SCI-MAP-REQ-037--045`; `SCI-MAP-OD-009` |
| EXT-013 | Scientific handoff/profile registry | Producer sequence, package/artifact digests, units, identity joins, policies, projection/support state, required products, alternate-route and transition rules | All primary packages, MAP, downstream consumers | Scientific-library/profile-registry owner required | No approved executable six-package→MAP profile exists; this audit draft has no authority | A logical bundle cannot be selected or executed as an owner-approved route | Executable/frozen handoff | Scientific-library/profile-registry owner plus named use owners | PROFILE REGISTRY GAP | VAL `SCI-VAL-REQ-008--010`, `029`, `044--049`; MAP `SCI-MAP-REQ-001`, `009`, `036`, `046--050` |
| EXT-014 | Scientific-owner decision record | Approved wording/date/authority, affected clauses/version, conservative interim state, transition, and evidence for every open gate | Registry and all claim-bearing consumers | Named package/use owners; some specific delegates unnamed | Active ledgers **present**; relevant decisions remain **OPEN** | Only the exact dependent route or claim remains blocked; observed software cannot substitute | Route-affecting decisions block execution; claim-scoped decisions block only their claim | Grant Wilson or delegated named scientific owner(s) | OWNER DECISION REQUIRED | Package owner ledgers; MAP `SCI-MAP-OD-003--009`; MAP `SCI-MAP-REQ-001`, `022`, `031--032`, `036`, `044`, `048`, `050` |
| EXT-015 | Tune/readout native `x/r` boundary | Native I/Q-to-`x/r` quantity, units, sign/reference, paired occurrence identity, timing/detector mapping, validity, causes, uncertainty, and immutable source parent | ALIGN, then the full chain | External tune/readout owner unnamed in admitted packet | **Absent**; only ALIGN's consumer requirements were admitted | ALIGN cannot certify its first paired input identity or scientific meaning from shape/field names | Main six-package chain from its first producer boundary | Tune/readout scientific owner | MISSING PRODUCER GUARANTEE | ALIGN `SCI-ALIGN-REQ-001--007`, `016`, `020--022`, `034--039` |

## 3. Package-internal open gates

The following are not cured merely by locating an external artifact:

| Gate | Package-internal question | Interim contract behavior | External dependency created after decision |
|---|---|---|---|
| MAP OD-003 | Is a typed response-unavailable map usable by a restricted consumer class? | Keep response explicit; never fabricate; response-dependent consumers fail closed | Registered restricted-consumer profile and response-independent rationale |
| MAP OD-004 | What minimum covariance representation persists versus remains lineage-resolvable? | Exact equation, state, assumptions, omissions and lineage; stronger consumer fails closed | Named persistence/operator/summary format and round-trip evidence |
| MAP OD-006 | Is Pointing/OOF arithmetic a registered MAP reuse, and under what AltAz profile? | No mode-specific fitting, astrometric, response or production authority | MODE profile, exact input/output identity and AltAz WCS authority |
| MAP OD-007 | What numerical domain and failure policy applies to dimensionless `coverage_cut`? | Exact value must be effective-policy-authorized or route fails before support rows | Versioned support-policy authority and transition evidence |
| MAP OD-008 | Which `G_{pi}` classes, normalization/boundary rules and producer are authorized? | Declare actual example class; infer no conservation or additional class | Named projection producer and grid/projection artifact |
| MAP OD-009 | Is canonical crop/pad authorized, and who owns future reprojection/mosaic? | Reject incompatible grid; MAP performs no grid-changing operation | Named preparation/reprojection package and operator authority |

The cited RTC clauses define representative occurrence, transitive influence,
direct exclusion, x-only donor recovery, and conditioned-`r`/response/dependent
covariance unavailability across the complete donor causal influence while
preserving the grid, raw-`r` parent, causes, and otherwise valid `x`. They also
define pair/grid identity, response/uncertainty state, and failure behavior.
Those are primary contract facts, not external gaps. What remains outside RTC
is the exact tune/readout, TEL/APT, boundary, calibration-input, use-policy,
profile, and other producer authority to which its rules are joined.

## 4. Cross-package ambiguities requiring profile decisions

### 4.1 Disabled PTC / direct CAL

`SCI-MAP-REQ-002` creates a direct semantic edge from calibrated CAL
ordinary-`xs` to MAP. MAP simultaneously depends on PTC for conditioned-sample,
coefficient, and covariance meaning. No current MAP clause authorizes PTC to be
absent, supplies a default `omega_i`, or defines a CAL-only covariance route.

Classification: **BLOCKER — cross-package profile gap**, not a MAP estimator
defect and not evidence that CAL already owns the missing PTC facts.

Required disposition: MAP plus the route, coefficient, policy, and affected
upstream scientific owners must either require PTC or approve a complete
alternative route with exact producer, coefficient, covariance, lifecycle,
validity, response, transition, and evidentiary authority. CAL owns its
calibrated meaning but does not alone authorize the bypass. Until then,
`direct CAL` stops before numerical MAP.

### 4.2 `G_{pi}` ownership

MAP requires a finite projection relation, but OD-008 does not yet name its
materializer/authority or conservation/boundary semantics. MAP owns the
projection request and estimator choices; ALIGN/AST coordinate authority is
not automatically projection-kernel authority, and an implementation default
cannot close the gap.

Classification: **BLOCKER — MAP-internal owner gate that must create a named
external authority dependency**.

### 4.3 Coordinate diamond

Tune/readout and TEL occurrence identity, the ALIGN pair/slot, RTC stable
sample/pair identity, independent coordinate validity, AST coordinates,
CAL/PTC facts, VAL evaluation identity, and the MAP request/materialized
`G_{pi}` relation must all join on one occurrence identity. Equal shapes,
times, values, or row numbers do not prove the join.

Classification: **BLOCKER — external identity authorities and profile binding
required**.

## 5. Required ledger closure record

An external dependency may be marked closed only when a successor record
contains:

1. exact owner and scientific authority;
2. immutable artifact/profile identity and full digest;
3. applicability domain, units, signs, frames, identities and validity period;
4. producer preconditions and consumer guarantees;
5. unavailable/invalid/failed states and cause-specific actions;
6. uncertainty, response, null-space/omission and influence disposition;
7. transition/deprecation rules and affected contract/profile versions; and
8. evidence appropriate to the claim layer, without treating implementation
   behavior or historical production use as scientific authority.

Until every **BLOCKER** required by the chosen route is closed, the MAP
handoff profile remains non-executable and must hard-stop before numerical
contribution construction.

## 6. Evidence identity

All locked consumer clauses resolve against
`55efd8a54464636a24e621f6d1b60486d235b20e`. Exact admitted paths, full
SHA-256 values, package status, authority role, and exclusions are recorded in
`SIX_PACKAGE_SOURCE_PACKET_REPORT.md`. An “absent” external binding in this
ledger means absent from the admitted clean-room packet, not a claim that no
artifact exists elsewhere.

Current RTC freeze record digest was retained as abbreviated `0cac43…c0bf`;
the corresponding freeze-verification digest was retained as abbreviated
`e9bba3…4229`. No full digest is inferred from those abbreviations.
