# TolTEC RTC–CAL–PTC Chain Integration Profile

| Field | Value |
|---|---|
| Profile | `v0.1 / r0.1` |
| Status | **Draft for scientific-owner review** |
| Method | Implementation-blind horizontal contract-coherence audit |
| Scientific owner | Grant Wilson |
| Source identity | branch `codex/scientific-contract-library`; commit `9564bcca0323dacb8bea13a5ec4bbbf3b908de8f` |
| Packages | SCI-RTC `v0.1/r0.9` frozen; SCI-CAL `v0.1`, rationale `r0.3`, formal authority draft and not frozen; SCI-PTC `v0.1/r0.4` frozen |
| Canonical PDF identities | RTC rationale `0d397c…629e`, RTC ECS `8ff6eb…5805`; CAL rationale `075efa…84d3`, CAL ECS `1a5cd0…326`; PTC rationale `7cb358…fd0`, PTC ECS `1e73d3…b85` |

## 1. Purpose and authority

This profile governs only the scientific relation among SCI-RTC, SCI-CAL, and SCI-PTC products and claims. Each package’s formal core remains authoritative for its internal operation. Owner amendments and approved decisions follow that core; rationales explain but do not override it. This profile cannot repair a package conflict, supply a missing owner choice, or promote draft authority.

The reviewed packet is identifiable and sufficient for a horizontal audit, but it is not uniformly frozen. RTC and PTC are frozen scientific authorities; CAL is a draft whose rationale architecture is frozen only as a template. Consequently, every CAL-dependent statement below is either conditional on CAL producing an admitted product or explicitly unresolved. Nothing here claims implementation conformity, representation fidelity, validation, observational performance, science qualification, production readiness, or current Citlali behavior.

Source abbreviations used below are exact clause families in the canonical shared cores:

- `RTC`: [`SCI-RTC v0.1/r0.9`](../../packages/SCI-RTC/v0.1/README.md), especially `SCI-RTC-DEF-001/004/010–013/018`, `SCI-RTC-EQ-003–006/011–022`, and `SCI-RTC-REQ-001–005/013–020/028–029/037–053/068/083–086/092–105/108`.
- `CAL`: [`SCI-CAL v0.1; rationale r0.3`](../../packages/SCI-CAL/v0.1/README.md), especially `SCI-CAL-DEF-001–017`, `SCI-CAL-EQ-001–013`, `SCI-CAL-ASM-011`, `SCI-CAL-REQ-001–005/013–016/020–048`, and owner questions `CAL-OWNER-Q01/Q02/Q06/Q08`.
- `PTC`: [`SCI-PTC v0.1/r0.4`](../../packages/SCI-PTC/v0.1/README.md), especially `SCI-PTC-DEF-001/041`, `SCI-PTC-REQ-001–003/010–017/021–022/046/057–066/071–078/083/087–089`, and `PTC-OWNER-OD-005/007/009`.

The expanded clause-by-clause mapping is in [`RTC_CAL_PTC_SOURCE_CROSSWALK.csv`](RTC_CAL_PTC_SOURCE_CROSSWALK.csv). Findings and unresolved choices are in the companion findings and owner-decision ledgers.

## 2. The three-package scientific chain

**RTC.** SCI-RTC begins with one admitted, exact aligned raw `x/r` pair. It applies its realized conditioned-`x` operation on the aligned grid, selects the phase-zero output grid, and publishes a consumer-neutral atomic bundle. That bundle retains the immutable raw-`r` parent, pair and mapping identities, realized plan, complete RTC-local response or typed unavailability, full support and influence, separated validity and eligibility inputs, uncertainty status, provenance, and causes. Directly synthesized or donor-replaced representative occurrences are excluded as independent measurements; nonrepresentative influence is preserved for each named consumer’s policy. RTC does not perform absolute calibration or target-atmosphere correction. (`RTC-DEF-001/004/010–013/018`; `RTC-EQ-003–006/011–022`; `RTC-REQ-001–005/013–020/028–029/037–053/103–105`.)

**CAL.** SCI-CAL defines an occurrence-scoped operation on admitted ordinary `xs`: one selected detector factor and one target-observation atmosphere correction are applied exactly once on the same admitted support, with the same multiplier used for conditional measurement uncertainty and any admitted fixed-state companion. It records factor authority, atmosphere authority, detector association, validity, nuisance scope, originating Beammap response basis, and canonical lineage. The output reference plane is top of atmosphere and the active rationale’s intended convention is point-source-equivalent, beam-peak-normalized mJy/beam; a literal peak claim remains conditional on downstream response or explicit renormalization. CAL does not itself certify a downstream realized response or the fixed-nominal-beam identity required by PTC. CAL also has not bound its `xs` to the exact SCI-RTC conditioned-`x` product (`Q01`), has not selected the full RTC/CAL/PTC order (`Q02`), and lacks the content-bound atmosphere operator needed for any numerical calibrated output (`ASM-011`, `Q06`). Its formal core still uses “point-source-peak” where the active rationale limits the claim to point-source-equivalent. (`CAL-REQ-001–005/013–016/020–048`; `CAL-EQ-001–013`; `CAL-ASM-011`; `Q01/Q02/Q06/Q08`.)

**PTC.** SCI-PTC begins only with an exact admitted SCI-CAL product and binds the complete RTC parent separately. It preserves the calibrated unit and convention while fitting and removing a declared correlated subspace. It publishes the transformed detector signal together with the removed subspace, nonrestored additive reference, null space, stage-specific fit/application/output support, coefficient roles, PTC-local response, complete-chain response status, full-procedure response status, uncertainty status, and provenance. It does not repair an unavailable RTC or CAL fact. Disabled PTC produces no PTC product on this route. (`PTC-REQ-001–003/010–017/021–022/046/057–066/071–078/083/087–089`.)

**Verified story.** The RTC→CAL→PTC ordering and product roles are jointly supported by RTC and PTC, and CAL’s local multiplication is compatible with that story. The story is nevertheless **conditional rather than closed** because CAL’s input identity/order, numerical atmosphere authority, cumulative response carriage, and output terminology have not all been resolved in CAL authority.

**Optional `r` branch.** Raw `r` remains an RTC parent and does not pass through CAL. A separately authorized conditioned-`r` product may feed PTC diagnostic-only, inert/advisory use if it carries its own unit, response, support, validity, leakage state, uncertainty, provenance, and exact relation to the CAL grid. No such producer is supplied in base v0.1; absence or incompatibility blocks only this optional branch. (`RTC-REQ-092/103/108`; RTC owner entries `071/074`; `PTC-REQ-078`; `PTC-OWNER-Q001`; `PTC-OWNER-OD-005`.)

## 3. RTC→CAL handoff invariants

| ID | Draft profile invariant | Status | Producer and consumer authority |
|---|---|---|---|
| CHAIN-INV-001 | CAL’s parent shall be the exact RTC conditioned-`x` product, with no reconstruction, reorder, or name-based substitution. | **Proposed; owner decision required.** RTC guarantees the exact object, but CAL does not yet name it. | RTC `DEF-004/018`, `REQ-001–005/103`; CAL `REQ-001/003/020`, `Q01`. |
| CHAIN-INV-002 | Observation, detector occurrence, sample, coherent segment, selected grid, cadence, phase, product, and parent identities remain exact. | **Supported conditionally on INV-001.** | RTC `DEF-001/004/018`, `REQ-006–008/028–029/083–086`; CAL `REQ-003/006–012/047–048`. |
| CHAIN-INV-003 | CAL receives RTC’s declared raw-`x` coordinate, unit, sign, reference, and valid domain without reinterpretation. | **Unresolved.** | RTC `REQ-001–005`; CAL `REQ-001–004`, `Q01`. |
| CHAIN-INV-004 | Raw `r` remains lineage/diagnostic state; it is not calibrated, substituted for `x`, or processed by the `x` numerical operator. | **Supported.** | RTC `EQ-004/006`, `REQ-092/103/108`; CAL `REQ-001`. |
| CHAIN-INV-005 | The handoff carries the complete RTC plan and RTC-local response, or an exact typed response-unavailable state. | **RTC-supported; CAL disposition incomplete.** | RTC `DEF-004/010/018`, `REQ-037–041/103`; CAL `REQ-003/040–043`. |
| CHAIN-INV-006 | RTC support, typed causes, direct representative synthesis/replacement, and transitive influence survive CAL unchanged; CAL may narrow use only by an explicit policy. | **Unresolved consumer disposition.** | RTC `DEF-011–013/018`, `EQ-020`, `REQ-019–020/046–052`; CAL `REQ-003–004/045–048`. |
| CHAIN-INV-007 | CAL does not resample or reinterpret time; its scalar/diagonal operation acts on the admitted RTC support and grid. | **Supported conditionally on INV-001.** | RTC `REQ-028–029/103`; CAL `EQ-004–006/008–009`, `REQ-015–016/031/039`. |
| CHAIN-INV-008 | CAL applies one selected absolute detector factor once and one target-atmosphere correction once. RTC donor `flxscale` ratios remain raw convention transfer, not prior absolute calibration. | **Supported.** | RTC `EQ-001–004`, `REQ-014–016`; CAL `EQ-004–006`, `REQ-013–016/021–031`. |
| CHAIN-INV-009 | CAL propagates an admitted fixed-state response companion with the exact realized CAL multiplier on identical support; missing support or operator authority yields no calibrated output. | **Supported locally; cumulative response unresolved.** | RTC `DEF-004/010`; CAL `EQ-008–009`, `REQ-031/039–043/045–046`, `ASM-011`. |
| CHAIN-INV-010 | Disabled, unsupported, invalid-factor, ambiguous-association, invalid-atmosphere, and outside-support cases do not become identity calibration. | **Supported.** | RTC `REQ-046–053`; CAL `REQ-004/010–012/021–031/045–046` and edge predictions. |

## 4. CAL→PTC handoff invariants

| ID | Draft profile invariant | Status | Producer and consumer authority |
|---|---|---|---|
| CHAIN-INV-011 | PTC consumes the exact immutable SCI-CAL product, not a recalculated or similarly named stream. | **Supported for any available CAL product.** | CAL `REQ-047–048`; PTC `REQ-001/046/063`. |
| CHAIN-INV-012 | The PTC input is top-of-atmosphere and point-source-equivalent with explicit response state; the unit does not prove literal peak preservation. | **Scientifically intended, but CAL wording conflict must be resolved.** | CAL `REQ-002/040–043`, active rationale §§1–2/5; PTC `REQ-001/010/061/087`. |
| CHAIN-INV-041 | PTC’s fixed-nominal-beam identity must be supplied by CAL or by an explicit response-basis conversion/renormalization. | **Consumer strengthening; not currently supported.** | CAL `DEF Response basis`, `REQ-040–043`; PTC `ASM-001`, `REQ-001/010`. |
| CHAIN-INV-013 | The selected factor and target-atmosphere correction have each been applied exactly once before PTC; PTC does not apply either again. | **Supported.** | CAL `REQ-013–016/020–031`; PTC `REQ-001/078/087`. |
| CHAIN-INV-014 | PTC receives calibration lineage: selected factor and detector association, target-atmosphere authority/support, validity, uncertainty, and correlation scope. | **Structurally supported; numerical product unavailable under CAL ASM-011/Q06.** | CAL `REQ-006–16/020–39/047–048`; PTC `REQ-001–003/057–061/071`. |
| CHAIN-INV-015 | PTC receives the complete admitted upstream response ending on the CAL grid exactly once. | **Response gap.** PTC requires the object; CAL does not unambiguously guarantee the cumulative RTC→CAL object. | CAL `REQ-039–043/047`; PTC `DEF-041`, `REQ-061–065/087`. |
| CHAIN-INV-016 | PTC receives the exact CAL detector-time grid; any realized companion already on that grid begins with the PTC-local operator and does not reapply RTC/CAL response. | **Supported.** | CAL `EQ-008–009`, `REQ-031/039`; PTC `REQ-061–065/087`. |
| CHAIN-INV-017 | PTC binds the complete immutable RTC parent needed for raw-`r`, selectors, segmentation, replacement, influence, response, and uncertainty interpretation. | **PTC requirement; end-to-end carriage gap.** | RTC `DEF-018`, `EQ-022`, `REQ-103–105`; CAL `REQ-047–048`; PTC `REQ-002`. |
| CHAIN-INV-018 | PTC may narrow RTC/CAL support for each PTC-owned use, but it preserves every upstream cause and cannot repair or strengthen unknown facts. | **Supported explicit narrowing.** | RTC `EQ-020`, `REQ-019–020/046–052`; CAL `REQ-045–048`; PTC `REQ-003/011–017/089`. |
| CHAIN-INV-019 | PTC preserves the calibrated convention while publishing its own removed subspace, nonrestored additive reference, null space, fit/application/output support, response, covariance status, and provenance. | **Supported.** | CAL `REQ-002/047–048`; PTC `REQ-010/021–022/057–066/071/083/087–089`. |
| CHAIN-INV-020 | PTC-disabled means no PTC product on this route; a direct CAL→MAP route is outside this profile. | **Supported.** | CAL product existence clauses `REQ-045–048`; PTC `REQ-076–077`. |

## 5. Chain-wide invariants

The following invariants apply whenever all prerequisite products exist:

1. **Immutable parents.** Each product binds its immediate parent and exact lineage. No parent is reconstructed from filenames, labels, row positions, or array shape. The CAL→PTC binding is supported; the exact RTC→CAL parent and full RTC-parent carriage remain open under INV-001 and INV-017.
2. **No strengthening.** Unavailable, conditional, incomplete, invalid, rejected, unsupported, or unknown upstream facts remain typed. PTC explicitly refuses repair; CAL numerical atmosphere unavailability terminates the numerical chain.
3. **Signal-role continuity.** RTC conditioned raw `x` changes to a calibrated top-of-atmosphere quantity only in CAL. PTC preserves that unit/convention while changing response and additive/null-space state. The exact RTC input semantics and CAL peak/equivalent naming remain unresolved.
4. **Once-only operations.** RTC’s donor ratio is convention transfer. CAL’s selected factor and target atmosphere each have application count one. PTC begins from the already calibrated parent. An admitted upstream response or realized CAL-grid companion is likewise applied exactly once.
5. **Distinct identities and lifecycle.** Observation, scan, coherent segment, detector occurrence, sample/grid, stage, product, requested/effective policy, learned evidence, resolved plan/model, realized operation, and published artifact remain distinct and one-way.
6. **Typed causes and stage-specific support.** Causes are facts, not universal masks. RTC direct representative exclusion is universal for independent-measurement use; noncenter influence is passed to the named consumer. PTC then applies a conjunctive, PTC-owned use policy and preserves downstream-owned facts. CAL’s disposition of these RTC cause axes is not yet explicit.
7. **Representative occurrence.** RTC’s representative occurrence is exactly `(detector, M n)` before later support effects. Direct synthesis/replacement remains excluded; noncenter influence is use-specific. PTC agrees. CAL may not erase or reinterpret those states while serving as the bridge.
8. **Uncertainty continuity.** Conditional measurement covariance, factor/atmosphere nuisance uncertainty, response uncertainty, selection uncertainty, null-space/model uncertainty, and cross terms remain separately typed. “Unavailable” or “not supplied” never means zero or total. CAL `Q08` and PTC `OD-009` prevent a complete numerical covariance claim but not a truthfully limited conditional product.
9. **Version and policy identity.** Contract revision, RTC plan, CAL factor/atmosphere authority, PTC support policy and estimator request, response definition, and parent identities are immutable inputs to any claim.
10. **Ownership.** Composition does not transfer internal estimator authority. RTC owns conditioning; CAL owns absolute factor/atmosphere calibration; PTC owns correlated-subspace estimation and application.

## 6. Response-companion and uncertainty composition

The domain-aware chain is:

```text
aligned RTC x grid
  --[RTC-local realized response K_RTC; fixed RTC state]-->
RTC conditioned-x grid
  --[CAL local multiplier M_CAL, same support, once]-->
CAL detector-time grid
  --[PTC-local frozen-state operator K_PTC]-->
PTC output grid
```

The complete fixed-state chain response, when available, is the ordered composition of these three domain-qualified objects. RTC’s optional ALIGN composition is a separate upstream extension and must not be hidden inside `K_RTC`. CAL’s factor/atmosphere multiplier is applied to the signal and admitted fixed-state companion once. PTC then applies its local frozen-state operator. A companion already realized on the CAL detector-time grid starts at PTC; it must not receive RTC or CAL response again.

Three stronger response notions remain distinct:

- **PTC propagated companion:** fixed RTC/CAL/PTC state, exact companion parent and domain.
- **PTC full-procedure response:** restarts from the immutable CAL parent, re-estimates PTC state, records state changes, and again discards the nonrestored additive reference. RTC and CAL remain fixed unless a separate whole-chain experiment says otherwise.
- **Whole-chain response:** perturbs an identified source/native parent and reruns every data-dependent RTC, CAL, and PTC operation authorized by that experiment. It is not implied by either fixed-state propagation or PTC full-procedure response.

The unresolved ownership point is the cumulative response ending on the CAL grid: PTC requires it, RTC supplies its local contribution or unavailability, and CAL supplies its local multiplier/companion rule, but CAL does not yet promise one unambiguous cumulative object carried by its product. In addition, a sample-dependent CAL atmosphere correction generally does not commute with RTC temporal filtering. RTC requires exact response composition or an approved noncommutation bound; CAL `Q02` does not yet bind that cross-stage order. Until both matters are resolved, complete-chain response-dependent use is unavailable; package-local response objects remain valid in their declared domains.

Uncertainty follows the same parent and realized support. RTC conditional covariance is conditioned on fixed RTC state and retains selection/model/systematic exclusions. CAL transforms admitted measurement covariance with the same multiplier and retains a nuisance ledger. PTC conditions on fixed learned/resolved state for its local covariance and separately types selection, full-procedure, response, cross-observation, and omitted terms. No layer may relabel partial coverage as total.

## 7. Decision and availability register

| Decision | Required owner action | Conservative state while open | Affected invariants |
|---|---|---|---|
| CHAIN-OD-001 | Bind CAL’s admitted `xs` to the exact SCI-RTC conditioned-`x` bundle and specify coordinate/unit/sign/reference/domain continuity. | RTC→CAL numerical handoff identity is unavailable. | 001–003, 005–007, 017 |
| CHAIN-OD-002 | Select and record CAL after RTC and before PTC, or an alternative fully composed order. | End-to-end operation order is not authoritative. | 001, 008–017 |
| CHAIN-OD-003 | Define who composes, publishes, and binds the complete upstream response ending on the CAL grid, including RTC-filter/CAL-atmosphere noncommutation, typed unavailability, and no-double-application rules. | Complete-chain response-dependent use is unavailable. | 005, 009, 015–016 |
| CHAIN-OD-004 | Supply the content-bound CAL atmosphere operator record required by `ASM-011/Q06`. | No numerical calibrated CAL product; therefore no numerical PTC product on this chain. | 011–020 |
| CHAIN-OD-005 | Resolve CAL’s “point-source-peak” formal wording against the active rationale/PTC “point-source-equivalent” convention, literal-peak condition, and PTC’s stronger fixed-nominal-beam identity. | Only the narrower point-source-equivalent claim is safe; literal peak and fixed-nominal-beam equivalence are withheld. | 012, 019, 041 |
| CHAIN-OD-006 | Specify which RTC causes/support/representative-state axes CAL preserves and how they are carried without becoming CAL-owned eligibility. | Cause-dependent CAL or downstream eligibility claims are unavailable. | 006, 017–018 |
| CHAIN-OD-007 | Bind the CAL→PTC uncertainty/correlation handoff, including the status of CAL `Q08` and PTC `OD-009`. | Truthfully limited conditional uncertainty may exist; complete covariance/total uncertainty is unavailable. | 014, 019 |
| CHAIN-OD-008 | Author a conditioned-`r` producer and exact CAL-grid compatibility contract if the optional branch is desired. | The main `x` chain is unaffected; the optional `r` diagnostic branch is unavailable. | Optional branch only |

## Scientific-owner disposition

The contracts establish a strong and mostly compatible structural chain, including once-only calibration, no repair/strengthening, PTC response-domain separation, typed support, direct-versus-noncenter influence, and no-product disabled semantics. They do **not** yet establish a closed end-to-end numerical chain.

Accordingly, this profile recommends **no end-to-end RTC→CAL→PTC implementation-conformity audit yet**. Package-local vertical audits may proceed only within their separately authorized scopes. The exact blockers are CAL’s unbound RTC input semantics and order (`Q01/Q02`), absent numerical atmosphere authority (`ASM-011/Q06`), missing cumulative-response carriage and RTC-filter/CAL-atmosphere noncommutation closure, incomplete RTC-cause/full-parent disposition through CAL, the CAL peak/equivalent normative wording conflict, and PTC’s unsupported strengthening to a fixed-nominal-beam identity. The open uncertainty and optional-`r` decisions limit stronger claims but do not by themselves invalidate a truthfully scoped main-chain structural profile.
