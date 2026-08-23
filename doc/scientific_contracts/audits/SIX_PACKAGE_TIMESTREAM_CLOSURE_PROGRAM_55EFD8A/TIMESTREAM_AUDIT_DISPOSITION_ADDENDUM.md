# Six-Package Timestream Audit Disposition Addendum

Status: **owner-directed audit-scope disposition and closure-control artifact;
package repairs and scientific-owner decisions remain pending**

Date: `2026-08-22`

Scientific owner: Grant Wilson

Audited branch: `codex/scientific-contract-library`

Immutable scientific-library source commit:
`55efd8a54464636a24e621f6d1b60486d235b20e`

Immutable baseline-audit commit:
`88c5df87277c22ab807e4a9ba74b7596c9586dc8`

Baseline audit:
[`SIX_PACKAGE_WIDE_SCALE_55EFD8A`](../SIX_PACKAGE_WIDE_SCALE_55EFD8A/AUDIT_EXECUTIVE_DISPOSITION.md)

Primary authorities: SCI-ALIGN, SCI-AST, SCI-RTC, SCI-CAL, SCI-PTC, and
SCI-VAL

Current program endpoint: the complete ordinary processed-timestream chain;
SCI-MAP and later map/coadd work are deferred.

## 1. Authority and preservation rule

This addendum records the scientific owner's direction to defer MAP work and
make the complete timestream process the present closure target. It accepts the
baseline audit's Outcome B and uses its evidence to organize repair. It does
not amend, replace, or retroactively reinterpret any of the twelve files in the
baseline audit directory.

The following identities therefore remain stable and immutable:

- findings `F-001`--`F-025`;
- owner decisions `XOD-001`--`XOD-020`;
- the baseline classifications, explanations, and counted disposition; and
- the distinction between the audited source commit and the later audit
  artifact commit.

This addendum is coordination and owner-disposition material, not a package
scientific authority. It does not select a PTC repair equation, define a
missing policy, approve an external producer, freeze a package, or close a
finding. Each substantive repair still requires its named scientific owner,
an immutable successor or boundary/profile artifact, and an independent
clean-room re-audit.

## 2. Revised endpoint and audit interpretation

The baseline audit correctly named SCI-MAP as a downstream reference rather
than a seventh audited authority. Its MAP handoff stress test exposed real
upstream defects, but MAP readiness became too prominent in the final closure
story. The present program restores the six-package endpoint:

```text
external acquisition authority
        |
        v
     ALIGN occurrence/time/origin/exposure
        |                         \
        |                          +--> AST ALIGN-grid coordinate role
        v
     RTC atomic conditioned bundle + plan + causes/influence
        |                          \
        |                           +--> AST RTC-grid coordinate role
        v
     CAL calibrated ordinary-xs detector samples
        |
        v
     PTC resolved model and transformed calibrated detector timestream

producer facts --> PTC use-owner dispositions/profiles --> VAL evaluation --> PTC action
```

PTC is the last numerical timestream transformer. VAL is a parallel decision
plane evaluated for exact PTC uses; it is not a numerical signal-processing
stage after PTC.

The endpoint quantity is a calibrated ordinary-`xs` detector sample and its
PTC-transformed calibrated detector timestream. It is not assigned a Stokes
identity here, is not an independent sky estimator, and acquires no map,
pixel, beam-realization, or significance meaning merely from a unit label.
CAL's declared `mJy/beam` convention remains attached to its exact calibration
and response lineage; after PTC it is not proof of a realized unit-peak source
response.

Conditioned `r` remains optional and uncalibrated; it is not a calibrated
science-output branch. Direct native-`r` evidence admitted during RTC plan
resolution is co-equal contamination evidence and may affect pair masks,
segments, guards, resets, support, and plan selection. Once that state is
resolved, the realized conditioned-`r` output does not feed back into plan
resolution and does not enter CAL. For fixed resolved RTC state, the numerical
cross-branch derivatives remain zero.

## 3. Logical timestream bundle `B_TS`

The closure-program logical endpoint is:

```text
B_TS = (
  acquisition and native readout identity,
  ALIGN occurrence, time, origin, support, and exposure,
  AST ALIGN-grid coordinate state,
  RTC atomic conditioned bundle and resolved plan,
  AST RTC-grid coordinate state,
  CAL calibrated-signal and calibration state,
  PTC transformed signal and frozen application state,
  PTC use-specific supports and VAL decisions,
  causes, influence, and exposure lineage,
  response and null-space state,
  uncertainty and covariance-availability state,
  QC and diagnostic state,
  lifecycle, provenance, and publication state,
  product availability and failure state
)
```

`B_TS` is a typed logical bundle, not a promise that every member is numerical
for every route. A complete disabled, not-requested, rejected, or unavailable
disposition may carry no transformed value while still requiring exact cause,
scope, lifecycle, and independent response/covariance states. A numerical
signal may be complete at a lower readiness tier while response or total
uncertainty remains explicitly unavailable.

| `B_TS` member | Scientific owner and required meaning | Snapshot obligation or gap | Primary source anchors |
| --- | --- | --- | --- |
| Acquisition and native readout identity | Tune/readout authority owns the native paired `x/r` mapping, sign, reference, Tune association, occurrence identity, validity, and runtime binding | Exact producer interface is outside the admitted packet; no shape/value inference is allowed | F-017/XOD-015; ALIGN `REQ-001`, `REQ-007--011` |
| ALIGN occurrence, time, origin, support, and exposure | ALIGN owns ordering, slot/interval identity, mapping, original/synthesized origin, physical acquired exposure, valid-original exposure, and zero added acquired exposure for synthesized values | Frozen authority is coherent; downstream preservation and exact external timing/input bindings remain required | ALIGN `REQ-003--006`, `REQ-015`, `REQ-027--030`, `REQ-050` |
| AST ALIGN-grid coordinate state | AST owns coordinate construction on the ALIGN role from exact TEL/pointing/geometry parents; RTC consumes this role only under exact identity | External producer identities remain claim-local gates | AST `REQ-006--031`, `REQ-036--060`, `REQ-073`; RTC `REQ-009`, `REQ-060`, `REQ-113`, `REQ-121` |
| RTC atomic conditioned bundle and plan | RTC owns stable output sample `n`, representative ALIGN parent, selected time/segment/phase, conditioned `x`, optional conditioned `r`, plan, support, response, causes, influence, and realization state | Formal `K` collision needs a bounded notation map; optional `r` compatibility remains role-specific | F-008; RTC `REQ-028--029`, `REQ-037--052`, `REQ-103--143` |
| AST RTC-grid coordinate state | RTC owns the output grid and parent facts; AST owns the continuous coordinate/direction role and its geometry, validity, response, and uncertainty state | Exact RTC-to-AST and detector-geometry/field-rotation boundaries are absent | F-006/F-007/XOD-008/XOD-009; AST `REQ-074--079` |
| CAL calibrated-signal and calibration state | CAL owns admission and binding checks, the exact consumed-instance record, the once-only transformation of admitted ordinary `xs`, the realized CAL state, declared output unit/reference convention, and CAL-local validity, response, uncertainty, and lineage. External producers retain ownership of acquisition/APT associations, calibration-factor meaning, and atmosphere/passband inputs | CAL needs final owner acceptance/freeze; exact external interfaces and runtime binding rules remain role-specific | F-016/F-017; CAL `REQ-001--003`, `REQ-006--031`, `REQ-039--050` |
| PTC transformed signal and frozen application state | PTC owns `Z`, learned evidence, selected estimator/model, centering and scaling, removed component/subspace, exact application map, derivative/linear part, null space, fallback, and output state | Frozen transformed-signal identities conflict; exact concrete ordinary route is not yet selected | F-001; `TS-CLAR-001`; PTC `REQ-019--050`, `REQ-069--089` |
| PTC supports and VAL decisions | PTC use owners separately disposition basis-fit, loading-fit, operator-application, output-retention, coefficient/QC, response-companion, and empirical/simulation identities as a registered profile, an explicitly unsupported use, or an owner-proved equivalent replacement; Registry binds supported profiles and VAL evaluates them | PTC truth rule is incomplete; seven use identities are reserved but lack complete owner dispositions; source bindings are stale | F-002--F-005/F-020; XOD-002, XOD-003, XOD-004, and XOD-006 |
| Causes, influence, and exposure lineage | Each producer owns facts/causes and direct/transitive influence; ALIGN owns physical and valid-original exposure; use owners own only named policy | Exact exposure occurrence/parent relation through RTC/CAL/PTC is not source-closed; `independent_exposure` is a policy proposition, not the exposure quantity | F-019; `TS-CLAR-002`; ALIGN `REQ-027--030`, `REQ-050`; VAL `REQ-019--020` |
| Response and null-space state | Each transformer owns its local fixed-state response. PTC's fixed-state role freezes membership, support, masks, model, and selection; a separately named full-procedure role reruns the procedure and records changed selection, support, rank, masks, and other `delta_state`. PTC also owns removed modes and its exact response companion | Source/beam-to-PTC response is a distinct readiness tier; typed unavailability is allowed below it | Timestream facet of F-012; PTC `REQ-021--023`, `REQ-061--066`, `REQ-083`, `REQ-087--088` |
| Uncertainty and covariance availability | Each producer owns supplied conditional terms, correlations, omissions, and applicability; no missing term becomes zero | Conditional and total uncertainty remain distinct; stronger claims require the exact missing terms and owners | Timestream facet of F-013; ALIGN `REQ-038--039`; AST `REQ-022`, `REQ-029`, `REQ-054--060`; RTC `REQ-042--045`; CAL `REQ-032--038`; PTC `REQ-057--060` |
| QC and diagnostics | PTC owns loadings, centering/scaling state, candidate evidence, classifications, diagnostics, refinement history, and their exact populations | A fitted loading is not a weight or precision; PTC coefficient/QC population policy remains in scope, while selection of any MAP-facing family is deferred | PTC `REQ-042--060`, `REQ-084--086`; timestream facet of F-004; F-023 deferred |
| Lifecycle, provenance, and publication | Every package owns immutable requested/effective/resolved/applied/realized generations and exact parent links; required outputs publish atomically | PTC disabled role map is incomplete; CAL/VAL/future PTC successor need final manifests and bindings | F-011/F-016; PTC `REQ-046--050`, `REQ-069--076`, `REQ-088`; VAL `REQ-027--029` |
| Availability and failure state | Producer owns product realization; response, covariance, and evidence remain independent axes; required failure propagates at declared scope | Every omitted or unavailable role must remain typed; no numeric sentinel, partial required bundle, or inferred fallback | F-011; PTC `REQ-069--088`; VAL `REQ-004`, `REQ-011--013`, `REQ-036--038` |

## 4. Explicit downstream deferral fence

The present closure program shall not select, author, register, freeze, or
re-audit:

- a MAP admission profile or direct CAL-to-MAP route;
- a Stokes-I product assignment;
- `G_{pi}`, sample-to-pixel deposition, pixel membership, projection class,
  normalization, boundary loss, or conservation rule;
- a MAP-facing analysis coefficient, weight, or precision interpretation;
- `coverage_cut`, MAP support, contribution, retained/projected exposure, or
  final pixel validity;
- response-unavailable map consumer policy;
- observation coadd, canonical-grid preparation, reprojection, or mosaicking;
  or
- MAP freeze or MAP implementation/readiness claims.

AST coordinate/WCS metadata remains in scope when it describes the exact
timestream coordinate role. It does not authorize a MAP pixel relation.

Deferral never means closure. MAP-only findings and the MAP facets of mixed
findings remain open in the baseline audit. The closure register records only
a `ts_reaudit_result`; it cannot declare a mixed original finding globally
closed.

## 5. Readiness ladder

The successor audit shall report a cumulative base chain
`TS-A -> TS-S -> TS-C` plus orthogonal response and uncertainty claim tiers.
`TS-R` and `TS-U` each extend `TS-C`; neither implies the other. `TS-T` is a
claim-specific state whose prerequisites must be named explicitly.

| Level | Exact claim | Required state |
| --- | --- | --- |
| `TS-A` — architecture coherent | The authority graph is acyclic and signal, coordinate, policy, response, uncertainty, exposure, and lifecycle meanings remain distinct | Stable package roles and no unresolved contradiction in the claimed topology. The baseline audit substantially establishes this level, subject to successor regression review |
| `TS-S` — source-closed processed signal | One exact ordinary acquisition→ALIGN→RTC→CAL→PTC signal route has frozen sources, exact identity/units/order, corrected PTC mathematics, complete dispositions for all seven PTC use identities, complete profiles for every supported/requested `TS-S` use, lifecycle/failure semantics, and truthful unavailable states | F-001--F-004, F-008, F-009, F-011, the timestream facets of F-016 and F-020, the native/signal, TEL clock/event timing, required CAL/PTC, and any coordinate/source/beam interface facets of F-017 needed to resolve the selected signal plan, and `TS-CLAR-001` closed for this route |
| `TS-C` — coordinate- and acquisition-accounting-complete timestream | `TS-S` plus exact ALIGN- and RTC-grid AST roles, remaining pointing/geometry/frame associations, and physical/valid-original exposure lineage through PTC | F-005, F-006, F-007, F-010, the remaining pointing/geometry/frame facets of F-017, the timestream exposure facet of F-019, and `TS-CLAR-002` closed for this route |
| `TS-R` — response-qualified timestream | `TS-C` plus an admitted source/beam basis and complete source-to-PTC conditional response for fixed resolved `Theta`, using frozen membership, support, masks, model, and selection; or a separately named full-procedure response that reruns the procedure and records changed selection, support, rank, masks, and other `Delta_state`; exact null-space state is retained | Source/beam/response facets of F-017 and the upstream facet of F-012 closed for the named role; response-companion use has a complete supported profile if requested; no MAP deposition response implied |
| `TS-U` — conditional-uncertainty-qualified timestream | `TS-C` plus exact conditional covariance or lineage-resolvable representation through PTC, preserving axes, units, support, correlations, approximations, selection/model terms, and omissions | Nuisance/correlation facets of F-017 and the upstream facet of F-013 closed for the named conditional claim |
| `TS-T` — total-uncertainty or other strong claim | Every authority required by one specifically named total uncertainty, significance, precision, photometric, or comparable claim is established; a response-dependent claim requires `TS-R`, an uncertainty-dependent claim requires `TS-U`, and a claim depending on both requires both | Claim-specific owner approval and evidence for every required term, cross-covariance, response role, and applicability statement; never inferred from normalization, unit, sample count, `TS-R`, or `TS-U` alone |

The program works through every lane even when an added claim branch ends in
typed unavailability. Failure to reach `TS-R`, `TS-U`, or `TS-T` does not erase
a valid `TS-C` result, and `TS-C` does not authorize any of those claim tiers.
Roles assigned only to `TS-R`, `TS-U`, or a future empirical/simulation route
may remain explicitly unsupported at `TS-S`.

Observation-instance realization is orthogonal to this package-readiness
ladder. For each run it is separately `unassessed`, `bound`, or
`missing/failed` for the exact selected inputs.

## 6. Finding and owner-decision disposition

The complete row-level tracker is
[`TIMESTREAM_FINDING_SCOPE_AND_CLOSURE_REGISTER.csv`](TIMESTREAM_FINDING_SCOPE_AND_CLOSURE_REGISTER.csv).
Its controlled scope classes are:

- `REQUIRED`: blocks `TS-S` or `TS-C` for the intended route;
- `CLAIM_TIER`: blocks only the named response/uncertainty/stronger tier;
- `DEFERRED_MAP`: excluded from this program and still open downstream; and
- `NONBLOCKING`: retained warning that does not block timestream closure.

The in-scope baseline set is:

- required: `F-001`--`F-011`, `F-016`, `F-017`, `F-019`, and `F-020`;
- claim-tier: the upstream timestream facets of `F-012` and `F-013`;
- nonblocking provenance: `F-022`; and
- downstream-deferred: `F-014`, `F-015`, `F-018`, `F-021`, and
  `F-023`--`F-025`, plus every MAP facet of a mixed finding.

Owner-decision scope is:

| XOD scope | Stable identities | Present disposition |
| --- | --- | --- |
| Active | `XOD-001`, `XOD-002`, `XOD-004`, `XOD-006`, `XOD-008`, `XOD-009`, `XOD-015`, `XOD-017` | Required for the applicable timestream route or its exact use dispositions |
| Split | `XOD-003`, `XOD-016`, `XOD-018` | Timestream facet active; MAP facet remains open and deferred |
| MAP-deferred | `XOD-005`, `XOD-007`, `XOD-010`--`XOD-014`, `XOD-019`, `XOD-020` | Excluded from the present program; no decision is inferred |

The register's `ts_reaudit_result` field uses only `TS_CLOSED`,
`TS_PARTIAL_MAP_OPEN`, `TS_NOT_CLOSED`, or `TS_REGRESSED`; an empty field
means `NOT_RUN`. `TS_CLOSED` is reserved for a wholly timestream-scoped record.
Every mixed finding with a deferred MAP facet must use
`TS_PARTIAL_MAP_OPEN` when its exact timestream facet passes. The original
baseline finding remains open wherever its deferred downstream facet remains
open.

### 6.1 Interpretive clarifications

1. `F-011` remains exactly the baseline finding and class. For closure
   planning it is treated as formal role-map incompleteness: the displayed
   equation omits roles but does not assert that the omitted roles are
   realized. This does not place it in the same mathematical category as
   `F-001` or truth-rule category as `F-002`. An exhaustive PTC disabled-state
   table may preserve the already-declared outward PTC-dependent MAP role as
   explicitly non-realized/deferred; doing so does not authorize a MAP route.
2. `F-017/XOD-015` is split by authority layer. A general contract needs an
   exact producer interface, meaning, version, identity, applicability,
   consumer guarantee, binding rule, and failure semantics. A particular run
   separately binds the observation's Tune, TEL, APT, atmosphere, source, and
   similar instances. Runtime data need not be embedded in the frozen contract
   repository, but a missing interface cannot be repaired by runtime values.
3. `F-019` is split by route. Exact preservation of ALIGN physical acquired
   and valid-original exposure facts through the PTC occurrence/output
   identity is timestream work. A downstream owner may add a separately named
   derived exposure only with an exact formula and parent relation; it may not
   replace, rewrite, or transform away the immutable ALIGN facts. MAP
   retained/projected exposure remains deferred. Weight, hit count, support,
   retention, or sample duration cannot invent exposure.
4. `F-022` is carried as a provenance warning. Frozen ALIGN/AST bytes are not
   reopened solely to remove embedded pre-freeze labels.
5. The endpoint uses quantity-neutral wording: calibrated ordinary-`xs`
   detector sample and PTC-transformed calibrated detector timestream. No
   total-intensity-equivalent or Stokes quantity is introduced by inference.

### 6.2 Post-audit timestream closure checks

These records refine the new endpoint; they are not retroactive baseline
findings:

- `TS-CLAR-001` — one concrete ordinary PTC processed-timestream route. The
  structural PTC freeze leaves `PTC-OD-001`--`PTC-OD-004` open: product-role
  selection predicates, absent-request/default behavior, centering/scaling
  choices, and diagnostic/refinement thresholds. The owner may require a
  complete explicit request rather than choose defaults. Optional conditioned
  `r`, source protection, full-procedure response, and stronger covariance
  roles must each be explicitly requested, unavailable, or deferred.
- `TS-CLAR-002` — exact physical and valid-original exposure lineage through
  RTC, CAL, and PTC. This check is scoped to resolving only the timestream
  facet of `F-019/XOD-018` after all closure evidence is present. It requires
  preservation of the immutable ALIGN facts and permits only separately named
  derived exposure with an exact formula and parent relation; it does not
  create a MAP exposure quantity.

## 7. Constraint versus owner decision

The audit establishes a required outcome without selecting every repair:

| Audit/closure constraint | Decision still reserved to the owner |
| --- | --- |
| PTC must have one algebraically complete transformed-signal/application identity | Exact successor equation, removed-component notation, and dependent clause changes |
| Every PTC use must have a complete T/F/U/C-congruent disposition | Exact predicates, thresholds, permissions, exceptions, and whether the successor retains any local composite |
| Seven PTC use identities require explicit, distinct dispositions | For each use: a complete registered profile when supported/requested, an explicit unsupported declaration, or a replacement only after owner-proved equivalence; VAL Registry binds and Core evaluates profiles but does not author them |
| A concrete ordinary PTC route must be resolvable | Estimator family/request requirement, candidate set, centering/scaling, product-role predicates, refinement policy, and optional roles |
| RTC-to-AST, geometry, timing, acquisition, calibration, and exposure relations require exact authority | Artifact names/versions, owners, domains, equivalence mappings, and runtime binding rules |
| Response and uncertainty claims must preserve exact availability and omissions | Which response/covariance tiers are required for each product role and which stronger claims remain unavailable |
| Frozen sources and profiles must be replayable | Successor versions, approval records, digests, compatibility, and supersession rules |

No implementation default, current numerical behavior, plausible value,
package-local finiteness, or earlier audit recommendation supplies an owner
decision.

## 8. Work packets and dependency gates

The register uses these canonical packet identifiers:

| Program label | Register identifier |
| --- | --- |
| WP-0 | `WP-0_DISPOSITION_TRACKER` |
| WP-1 | `WP-1_PTC_SUCCESSOR` |
| WP-2 | `WP-2_TS_BOUNDARIES` |
| WP-3 | `WP-3_CAL_EXTERNAL_PRODUCERS` |
| WP-4 | `WP-4_SOURCE_HYGIENE` |
| WP-5 | `WP-5_VAL_BINDINGS_PROFILES` |
| WP-6 | `WP-6_TS_CLAIM_TIERS` |
| WP-7 | `WP-7_FREEZE_REAUDIT` |
| Downstream sentinel | `DEFERRED_MAP` |

`DEFERRED_MAP` is a scope sentinel, not a repair packet or a closure state.
Multiple packet identifiers in one register cell are separated by semicolons.

### WP-0 — disposition and tracker

Create this addendum and closure register without editing the baseline audit or
any package authority. This packet records scope and evidence requirements; it
does not close a finding.

### WP-1 — bounded PTC successor

Prepare the package-local artifacts needed to resolve `F-001`, `F-002`,
`F-009`, `F-011`, and the PTC-owned portion of `TS-CLAR-001` in one coherent,
versioned PTC successor. Produce the clause/equation/prediction change map and
seven PTC-owned use-disposition records, including complete profiles only for
supported/requested roles. `TS-CLAR-001` cannot close until its CAL/external
bindings from WP-3, VAL registrations from WP-5, and WP-7 source binding and
re-audit are complete. Do not alter unrelated PTC science, select a MAP
coefficient, or use implementation behavior as authority.

### WP-2 — timestream identity, coordinate, and exposure boundaries

Prepare the owner-approved RTC-to-AST, detector-geometry/field-rotation, and
exposure-lineage artifacts needed to close `F-006`, `F-007`, the timestream
facet of `F-019`, and `TS-CLAR-002`. These compose existing package authority and
must not silently change RTC or AST mathematics.

### WP-3 — CAL and external producer closure

Produce CAL's scientific-owner review and freeze artifacts for later WP-7
binding, and resolve the selected route's `F-017` interface obligations.
Separate static/operator authority from observation-instance realization. A
dependent optional role may be explicitly not requested or unavailable; it
may not be filled by inference.

### WP-4 — bounded source hygiene

Resolve `F-008` and `F-010` through owner-approved notation errata, semantic
mapping records, or versioned successors with parity. Carry `F-022` off the
critical path. Do not rewrite frozen bytes casually or broaden the science.

### WP-5 — VAL bindings and profiles

Begin final binding only after PTC, CAL, and affected boundaries are stable.
Update exact non-MAP source rows, complete
`SCI-VAL:independent_exposure@1`, register complete profiles for every
supported/requested PTC use, record explicit unsupported or owner-proved
equivalence dispositions for the other reserved identities, disposition CAL
facts for each use under `F-020`, and produce the VAL freeze artifact for
later WP-7 binding. Do not modify VAL Core or register MAP/coadd profiles.

### WP-6 — timestream response and uncertainty claim tiers

Assess the upstream facets of `F-012` and `F-013` and prepare closure evidence
only for the exact named `TS-R`, `TS-U`, or `TS-T` claims. Preserve typed
unavailable states and all omitted terms where the stronger tier is not
established.

### WP-7 — freeze and clean-room re-audit

Verify and bind all required successor authorities, the CAL and VAL freeze
artifacts produced by WP-3 and WP-5, boundary artifacts, profiles, approval
records, and full digests in one immutable source commit. WP-7 does not
originate those scientific-owner decisions. Then launch a fresh six-package
timestream-only audit.

WP-1, WP-2, WP-3, and WP-4 may proceed in parallel where their owner inputs are
independent. WP-5 waits for producer identities to stabilize. WP-6 may develop
alongside producer work but its final result binds the stabilized sources.
WP-7 waits for every required `TS-S`/`TS-C` item selected for the candidate.

## 9. Closure register protocol

The register's `current_state` begins with no closed rows. Normal progression
is:

```text
OPEN
  -> PROPOSED
  -> OWNER_APPROVED
  -> INCORPORATED
  -> SOURCE_BOUND
  -> REAUDITED
```

Exceptional states include `PARTIAL`, `DEFERRED`, `CARRY_FORWARD`, and
`REGRESSED`. The independent outcome is recorded separately in
`ts_reaudit_result` using the vocabulary in Section 6. A wholly
timestream-scoped row may receive `TS_CLOSED`, and a mixed row must receive
`TS_PARTIAL_MAP_OPEN`, only when it has:

1. the exact repair/boundary/profile artifact;
2. successor identity and complete digest;
3. named owner approval;
4. source/closure commit;
5. compatibility and supersession disposition; and
6. independent clean-room timestream re-audit result.

External-interface closure also requires the eight-part record in the
baseline
[`EXTERNAL_AUTHORITY_DEPENDENCY_LEDGER.md`](../SIX_PACKAGE_WIDE_SCALE_55EFD8A/EXTERNAL_AUTHORITY_DEPENDENCY_LEDGER.md#5-required-ledger-closure-record).
The register does not duplicate or weaken that rule.

## 10. Clean-room re-audit protocol

Do not rerun the audit against `55efd8a...`; its immutable result already
exists. Re-audit only after the repair waves produce one new immutable source
commit.

The fresh auditor shall initially receive only:

- the then-current admitted scientific authorities for SCI-ALIGN, SCI-AST,
  SCI-RTC, SCI-CAL, SCI-PTC, and SCI-VAL;
- exact approved timestream boundary, external-interface, source-binding, and
  profile artifacts; and
- a sanitized program charter containing only the endpoint, scope, readiness
  names, and clean-room firewall, plus the exact source/freeze manifests
  required to interpret authority. That charter is not this addendum and
  contains no prior findings, decisions, tracker rows, or baseline scenarios.

SCI-MAP shall not be admitted. Package-owned outward MAP references may be
recorded as deferred without following them.

The auditor shall not inspect Citlali implementation, tests, configuration,
schemas, generated products, validation results, prior audits, repair
directives, this closure register, chat history, web sources, or undocumented
practice as scientific evidence. The prior audit and tracker may be provided
only after independent extraction, findings, and scenario results are locked,
solely to create a regression appendix mapping new results to stable F/XOD
identities.

The auditor shall first derive and lock an independent scenario suite. Only
after independent extraction, findings, and scenarios are locked may a
separate regression appendix replay the upstream premises of baseline
scenarios 1--23, 27--28, and 30, remove their MAP consequences, and add direct
checks for `TS-CLAR-001` and `TS-CLAR-002`. Both the independent report and the
regression appendix shall report `TS-A`, `TS-S`, `TS-C`, `TS-R`, `TS-U`, and
any `TS-T` claim separately.

The re-audit shall make no claim of implementation conformity, representation
fidelity, observational validation, achieved performance, deployment,
production readiness, MAP readiness, Stokes reconstruction, or downstream
consumer acceptance.

## 11. Immediate next gate

After this packet is reviewed, begin WP-1, WP-2, WP-3, and WP-4 as bounded
lanes. Before authoring normative repairs, return the scientific-owner choices
required for the exact PTC successor and concrete ordinary route; do not treat
the candidate repair language in the earlier external response as approval.

Final VAL rebinding remains gated on stable producer successors. The next
successful scientific milestone is a new immutable candidate that can be
audited for `TS-S` and `TS-C` without consulting SCI-MAP.
