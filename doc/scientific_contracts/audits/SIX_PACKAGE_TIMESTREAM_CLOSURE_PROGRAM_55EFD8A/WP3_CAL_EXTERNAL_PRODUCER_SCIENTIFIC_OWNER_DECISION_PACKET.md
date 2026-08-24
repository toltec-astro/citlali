# WP-3 CAL And External-Producer Scientific-Owner Decision Packet

Date opened: `2026-08-23`

Status: scientific-owner decision complete; `WP3-OWNER-D001`, upstream
clarification `WP2-FOLLOWUP-D011`, and `WP3-OWNER-D002--D008` are approved;
compact binding-register preparation and verification remain

Scope: `WP-3_CAL_EXTERNAL_PRODUCERS`, the CAL facet of `F-016`, the
timestream-route facets of `F-017`, and the applicable CAL/input contribution
to `TS-CLAR-001`.

This packet records scientific-owner decisions only. It does not inspect
Citlali, assert implementation conformity, execute validation, or authorize
MAP work.

## WP3-OWNER-D001 — CAL Authority Freeze Versus Achieved Performance

Question:

> Should SCI-CAL v0.1—scientific rationale r0.5 and engineering conformance
> r0.4—be frozen now as the exact scientific authority, while observational
> validation and achieved-performance acceptance remain pending?

Recommendation presented to the owner:

> Yes. Q01--Q09 are scientifically decided, the science and engineering views
> are consistent, and the package is mechanically complete. Freeze the exact
> contract authority now without claiming implementation conformity,
> observational validation, achieved accuracy, total uncertainty, science
> qualification, production readiness, or achieved-performance acceptance.
> Treat the 1%, 5%, and 5--10% figures as reporting benchmarks. Later evidence
> attaches under its own identity and changes the frozen authority only if the
> science itself must change. Replace the ambiguous phrase “final scientific
> acceptance” with “achieved-performance acceptance.”

Owner response:

> I approve.

Disposition: **approved**.

Consequences:

1. SCI-CAL v0.1 science-rationale r0.5 and engineering-conformance r0.4 are
   frozen as the active scientific authority.
2. Contract-authority approval and achieved-performance acceptance are
   distinct states.
3. The Q09 validation workflow is authoritative, but no executed evidence or
   achieved claim is inferred.
4. The exact freeze and verification records live in
   `packages/SCI-CAL/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.5.md` and
   `packages/SCI-CAL/v0.1/FREEZE_VERIFICATION_R0.5.md`.
5. This supplies the CAL freeze artifact required by WP-3; `F-016` remains
   open pending VAL freeze, final binding, and WP-7 clean-room re-audit.

## WP2-FOLLOWUP-D011 — Tune/Readout Producer-Interface Closure

The first formulation of `WP3-OWNER-D002` incorrectly placed native
Tune/readout and \(x/r\) acquisition semantics at the RTC-to-CAL boundary.
Owner review established that frozen SCI-ALIGN v0.1/r0.3 and SCI-RTC
v0.1/r0.12 already assign those meanings upstream of RTC:

\[
(I,Q)^{\rm acq}
\longrightarrow \text{Tune/readout}
\longrightarrow (x,r)^{\rm acq}
\longrightarrow \text{SCI-ALIGN}
\longrightarrow (x,r)^A
\longrightarrow \text{SCI-RTC}.
\]

The remaining omission is an exact external producer-interface binding, not
a missing CAL acquisition interpretation.

Question:

> Should we add a bounded, package-neutral Tune/readout producer-interface
> record upstream of ALIGN, without reopening ALIGN or RTC?

Recommendation presented to the owner:

> Yes. Bind the versioned producer/interface meaning and observation-instance
> association required by frozen ALIGN and RTC. Preserve the producer-owned
> units, sign, reference, normalization, transform, Tune/mapping revision,
> paired occurrence identity, applicability, validity, uncertainty state, and
> provenance. Do not invent a new sign convention, make CAL interpret Tune
> data, duplicate ALIGN/RTC mathematics, or freeze observation payloads in the
> repository.

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. The issue is routed to `WP-2A_NATIVE_READOUT_INTERFACE`, the bounded
   upstream facet of `F-017/XOD-015`.
2. The candidate interface is
   `producer_interfaces/v0.1/TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`.
3. Frozen SCI-ALIGN and SCI-RTC remain unchanged.
4. Exact candidate-artifact approval remains pending.
5. The original broad CAL-related formulation of D002 is withdrawn.

## WP3-OWNER-D002 — Conditioned RTC `x` Handoff To CAL

Status: approved

Question:

> Should the RTC-to-CAL signal boundary admit only the exact conditioned
> \(x^{\rm RTC}_{dn}\) member of a complete realized RTC atomic bundle, with
> its inherited upstream coordinate convention and RTC-owned grid, plan,
> support/response state, validity, uncertainty state, lineage, and
> provenance, while treating \(r\) as traceability rather than a CAL numerical
> input and treating calibration-factor and target-atmosphere records as
> separately owned CAL-time joins?

Recommendation:

> Yes. RTC shall guarantee that the handed signal is the exact conditioned
> \(x\) output for the named detector and stable RTC sample on the named
> realized plan and grid. Its raw detector unit or scale, sign, reference, and
> normalization are inherited unchanged from the admitted upstream mapping;
> this boundary creates none of them. The handoff shall preserve the exact
> representative ALIGN parent, complete RTC-local action/influence support,
> realized response or typed-unavailable response state, validity and causes,
> uncertainty/covariance availability, original-occurrence/exposure lineage,
> lifecycle, and provenance. The complete paired RTC bundle remains the
> immutable traceability parent, so raw or requested conditioned \(r\) and
> \(r\)-derived RTC evidence remain reachable, but SCI-CAL neither consumes
> them numerically nor calibrates them. Selected `flxscale`/child-APT and
> target-atmosphere/passband inputs remain separate producer-owned inputs
> joined by SCI-CAL; an RTC handoff may carry their exact references as
> required by frozen RTC authority but does not own, derive, repair, validate,
> or apply them. Missing or ambiguous required \(x\) identity, plan/grid,
> support, validity, or provenance fails the affected CAL admission. A typed
> unavailable response or uncertainty state never becomes zero and blocks
> each dependent claim or operation at the scope required by the frozen CAL
> and RTC contracts.

Owner response:

> yes

Disposition: **approved**.

Consequences:

1. SCI-CAL consumes only the exact conditioned \(x\) member numerically.
2. The complete paired RTC bundle remains the immutable traceability parent;
   CAL does not calibrate raw or conditioned \(r\).
3. The \(x\) unit or scale, sign, reference, and normalization are inherited
   from upstream authority and are not redefined at this boundary.
4. RTC-owned plan, grid, support/response state, validity, causes, uncertainty
   state, occurrence/exposure lineage, lifecycle, and provenance accompany the
   admitted \(x\).
5. Response authority may remain compact, factorized, or otherwise
   reconstructible; the decision does not require dense response
   serialization.
6. Calibration-factor/APT and target-atmosphere/passband records remain
   separately owned CAL-time joins. RTC may preserve their exact references
   without acquiring their scientific ownership.
7. Missing required handoff identity or state fails CAL admission at the
   affected scope; typed unavailable response or uncertainty never becomes
   numerical zero.

## WP3-OWNER-D003 — Absolute Calibration-Factor Handoff

Status: approved `2026-08-24`

Question:

> Should CAL receive one logical, layered calibration-factor bundle while
> preserving separate ownership for source calibration, target
> association/transformation, delivery, and consumption?

Recommendation:

> Yes. The logical bundle shall preserve the source APT and generating record,
> target-to-source association, approved child transformation, exact selected
> child row, delivery record, and selected factor \(F^{\rm sel}_{od}\). The
> SCI-BEAM/source-calibration authority owns the original `flxscale` meaning,
> calibrator model, source atmosphere, beam and spectral convention, validity,
> and uncertainty state. TolProj owns the unique target-to-source association
> and any explicitly approved child transformation, recording an identity
> transformation when the value is unchanged. TolTECA delivers the exact
> immutable child artifact without changing its value or meaning. SCI-CAL
> binds one target detector occurrence to one exact child row and applies that
> row's finite, nonzero \(F^{\rm sel}_{od}\) exactly once. Parent `flxscale`,
> any pointing correction already embodied in the child, `responsivity`,
> `sens`, or an opaque `fcf` shall not become additional CAL multipliers. Row
> position is not detector identity. A TolAPT/design match is not required for
> ordinary measured-APT calibration; if present, it remains separate
> ancestry. Missing, ambiguous, duplicate, or incompatible association,
> artifact digest, detector binding, factor direction, or recipient yields no
> calibrated output at the affected scope. Missing factor uncertainty may
> leave the calibrated signal available only with stronger uncertainty claims
> typed unavailable. Observation-specific APTs remain runtime artifacts rather
> than repository-embedded payloads.

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. One reconstructible logical handoff does not collapse the distinct
   producer, transformer, delivery, and consumer authorities.
2. The selected immutable child row is CAL's sole absolute numerical factor
   input for the v0.1 ordinary route.
3. Parent values and embodied transformations remain lineage rather than
   additional runtime multipliers.
4. Exact artifact, occurrence, association, row-recipient, direction, unit,
   validity, application-count, and uncertainty states remain queryable.
5. Missing factor uncertainty blocks only the dependent uncertainty claim;
   missing required factor identity or value blocks the affected calibrated
   signal.
6. The decision selects no new source-APT policy, child transformation, or
   observation instance beyond the authority already frozen in SCI-CAL.
7. `WP3-OWNER-D005` governs the concrete realization of this logical handoff:
   these authorities remain transitively resolvable through one content-bound
   matched-APT manifest and are not separate CAL payloads.

## WP3-OWNER-D004 — Target-Atmosphere Input Realization

Status: approved `2026-08-24`

Question:

> Should CAL construct its target-atmosphere correction from a layered runtime
> join of authoritative WVR and telescope-state facts with the frozen
> atmosphere operator, rather than consume an opaque precomputed correction
> factor?

Recommendation:

> Yes. The WVR producer owns \(\tau_{225}\), its event time and epoch meaning,
> measurement support, validity, uncertainty state, and exact bracketing
> readings. TEL/ALIGN/AST authority supplies current elevation and, where
> required, full airmass on the exact RTC output-sample identity; airmass
> retains its model identity and CAL shall not silently assume
> \(X=1/\sin(EL)\). SCI-CAL owns only the authorized interpolation and
> evaluation defined by its frozen contract,
> \(C_{a,\alpha}(\hat\tau_{225}(t_n),EL_n)\), using the frozen content-bound
> atmosphere operator and passband identities. WVR interpolation is allowed
> only between valid bracketing states under the declared source-authorized
> rule. Endpoint extrapolation, prior-observation inheritance, clamping, and
> replacement with a legacy or opaque correction are prohibited. Elevation,
> airmass, WVR state, and conditioned \(x\) join on the exact observation and
> RTC sample identity. Observation-level science/engineering classification
> does not enlarge sample-level numerical support. The atmosphere factor has
> application count one. Actual WVR and telescope records remain runtime
> instances. Missing or invalid required WVR, telescope state, operator
> support, or identity yields no calibrated output for the affected sample.
> Output-time correction is not claimed to invert atmosphere variation across
> RTC's full filter support; the applicable RTC response or an approved
> noncommutation bound remains required for each dependent response claim or
> operation.

Owner response:

> yes

Disposition: **approved**.

Consequences:

1. Target-atmosphere calibration is a transparent multi-parent join, not an
   independently authoritative opaque correction product.
2. WVR, telescope-state, operator/passband, and CAL application ownership
   remain separate and reconstructible.
3. No airmass law, interpolation beyond valid brackets, clamping, endpoint
   extrapolation, prior-observation inheritance, or legacy fallback is
   inferred.
4. Observation classification and numerical sample support remain distinct.
5. The exact multiplicative atmosphere correction is applied once on valid
   support and remains unavailable where any required parent or domain state
   is missing or invalid.
6. RTC support and response authority remain attached so output-time
   correction cannot be misrepresented as an automatic inverse of
   support-varying atmosphere.

## WP3-OWNER-D005 — Matched-APT Reference And Beammap Provenance

Status: approved `2026-08-24`

Question:

> Should the exact imported, SHA-bound matched APT and its manifest be the
> sole APT/calibration provenance reference for CAL and AST, with CAL binding
> a detector row and applying its `flxscale`, AST consuming its detector
> position, and all upstream Beammap, matching, transformation, and response
> detail remaining recoverable through that reference rather than duplicated?

Recommendation:

> Yes. Record one exact matched-APT artifact identity, content digest, and
> manifest identity at reduction scope. Bind each admitted detector to its
> exact artifact-local row key or detector UID. CAL consumes that row's
> finite, nonzero `flxscale` exactly once; AST may consume the detector
> position and other coordinate facts owned by the same APT row. The matched
> APT manifest remains authoritative for the originating Beammap observation
> number, association and matching ancestry, approved transformations,
> artifact history, and other APT provenance. CAL shall not copy those facts,
> detector positions, Beammap fit parameters, or response descriptions into a
> new CAL-specific bundle or per-sample lineage. A later response-dependent
> consumer follows the manifest to the exact BEAM/Beammap artifact and binds
> the required response authority at that time. Failure to materialize
> detailed Beammap response information does not block an otherwise valid
> calibration; it leaves only response-dependent claims or operations
> unavailable. A missing or digest-invalid matched APT, ambiguous or missing
> detector-row binding, or missing, nonfinite, or zero selected `flxscale`
> blocks CAL at the affected scope. The nominal-beam calibration convention,
> Beammap response, and realized RTC/PTC response remain distinct scientific
> objects. This decision clarifies the concrete realization of D003 and does
> not add another provenance layer.

Owner response:

> agreed

Disposition: **approved**.

Consequences:

1. The SHA-bound matched APT and its manifest are the single content-bound
   APT/calibration handoff object.
2. The artifact reference is recorded once at reduction scope; only the exact
   APT row key or detector UID need accompany the detector binding. No
   per-sample APT provenance copy is required.
3. CAL reads `flxscale`, while AST reads detector-coordinate facts, from their
   authorized fields in the same referenced artifact; sharing the reference
   does not merge package ownership.
4. The originating Beammap observation number, matching and transformation
   history, and upstream artifacts remain transitively resolvable through the
   manifest and are not duplicated in CAL.
5. Detailed Beammap/BEAM response authority is resolved only when a later
   response-dependent claim or operation requires it. Its absence does not
   by itself invalidate the calibrated signal.
6. Missing required matched-APT identity or integrity, detector-row identity,
   or a finite nonzero selected `flxscale` blocks calibration at the affected
   scope.
7. The nominal-beam calibration convention is not the Beammap response, and
   neither is the realized RTC/PTC response.
8. D003's layered ownership model is retained semantically, but its runtime
   realization is referential rather than a collection of duplicated
   payloads.
9. No CAL successor is presumed necessary merely to serialize Beammap detail.
   Any later clean-room finding of a literal conflict shall be handled as a
   bounded wording correction rather than used to create redundant state.

## WP3-OWNER-D006 — Compact CAL-To-PTC Handoff

Status: approved with owner correction `2026-08-24`

Question:

> What is the minimum CAL-to-PTC handoff that preserves the scientific
> meaning of the calibrated timestream without reproducing the observation's
> upstream history at another processing boundary?

Recommendation as corrected by the owner:

> Pass the data needed by the next stage and reference the history rather than
> retelling it. The primary handoff is the calibrated ordinary-`xs` signal
> \(Y^{\rm CAL}_{dn}\), its detector and RTC output-sample identity, the
> flags/validity required for PTC to interpret it, and references to the
> relevant CAL and RTC products or sidecars. The RTC sidecar owns the
> RTC-specific decisions, validity and diagnostics, response information,
> parentage, and other required RTC facts. The CAL sidecar owns what
> calibration was applied, the matched-APT reference, target-atmosphere
> calibration information, CAL validity/classification, and other CAL-owned
> facts. PTC shall reference those records when their information is needed;
> the boundary shall not re-encode RTC support, exposure lineage, APT or
> Beammap ancestry, WVR history, operator identities, or the full provenance
> graph. PTC knows from the admitted CAL product identity that the CAL-defined
> operations have already been applied and shall not reapply them. CAL
> produces no calibrated \(r\) quantity. PTC's science-signal parent is the
> calibrated \(x\) (`xs`) timestream. The paired RTC \(r\) channel remains
> reachable through the RTC parentage; whether and how PTC uses it as
> auxiliary evidence is owned by PTC. CAL failure does not authorize PTC to
> substitute uncalibrated RTC \(x\). An `engineering-only` classification is
> preserved in the CAL sidecar but does not itself prohibit PTC mathematics;
> named-use admission remains owned by the applicable PTC or downstream
> policy.

Owner response:

> I agree with the scientific core of D006, subject to the simplified
> reference-based boundary above. Do not create new provenance payloads merely
> because information might someday be useful downstream. Each stage records
> the new scientific facts that it owns and references inherited history.

Disposition: **approved with owner correction**.

Consequences:

1. The primary numerical parent of PTC is \(Y^{\rm CAL}_{dn}\) with detector
   identity, RTC output-sample identity, and the required flags/validity.
2. The handoff carries references to the CAL and RTC products or sidecars; it
   does not serialize their contents again.
3. CAL records only CAL-owned new facts in its compact sidecar. RTC facts stay
   in the referenced RTC sidecar, and APT, Beammap, WVR, and other upstream
   facts stay in their authoritative referenced artifacts.
4. PTC does not reapply or reinterpret CAL operations. Detailed factor,
   atmosphere, response, or lineage facts are dereferenced only when a PTC
   operation or scientific claim actually requires them.
5. CAL produces no \(r^{\rm CAL}\). The paired \(r^{\rm RTC}\) remains
   reachable through RTC parentage, and PTC owns any auxiliary use of it.
6. A failed or absent CAL science-signal parent does not enable an RTC-\(x\)
   fallback within the ordinary PTC route.
7. `engineering-only` remains a preserved CAL classification, not an automatic
   prohibition on PTC mathematics; each scientific use owns its admission.
8. Response or kernel composition remains possible by resolving the referenced
   RTC and CAL authority when needed; dense or repeated response serialization
   is not required.

## WP-3 Boundary Serialization Discipline

The owner correction to D006 governs the remainder of WP-3:

> **Pass the data needed by the next stage; reference the history rather than
> retelling it.**

Each stage records the new scientific facts it owns in its product or compact
sidecar and retains resolvable references to authoritative parents. A boundary
must not create a new provenance payload merely because inherited information
might later be useful. Reconstructibility and auditability require resolvable
authority, identity, and integrity; they do not require every processing stage
to serialize the full history of the observation.

## WP3-OWNER-D007 — External-Interface Closure By Reference

Status: approved `2026-08-24`

Question:

> When an authoritative upstream product or sidecar already contains the
> required scientific fact, should WP-3 bind that existing artifact by
> identity and reference rather than define another payload that repeats its
> contents?

Recommendation:

> Yes. A producer-interface contract may identify the authoritative existing
> product or sidecar, its schema or interface version, the observation-instance
> selection rule, required integrity and compatibility checks, and failure
> behavior. It shall not create another observation data product merely to
> duplicate the referenced information. The native \(x/r\) identity remains
> in the Tune/readout and ALIGN/RTC parentage; RTC processing and response in
> the RTC product or sidecar; RTC-grid pointing in the RTC-to-AST reference;
> detector geometry and `flxscale` in the SHA-bound matched APT row; Beammap
> and calibrator ancestry in the matched-APT manifest; WVR opacity and
> telescope state in the existing observation telescope records; the
> atmosphere operator and passband in frozen content-bound CAL authority; CAL
> application and classification in the CAL product or sidecar; and original
> acquisition/exposure in referenced ALIGN/RTC occurrence lineage. If an
> existing product lacks a required authoritative fact, identify that exact
> omission and either add it at its owning stage, leave the dependent use
> unavailable, or return the bounded omission for owner review. Do not repair
> it by constructing a generic provenance bundle in CAL.

Owner response:

> yes

Disposition: **approved**.

Consequences:

1. A boundary contract binds an authority; it need not copy the authority's
   data.
2. WP-3 creates no combined APT/Beammap/WVR/TEL/CAL provenance payload and no
   per-sample copy of observation-level metadata.
3. Static interface records may describe reference resolution, identity,
   integrity, compatibility, and failure without becoming new runtime data
   products.
4. Each runtime stage carries only its newly owned facts and references to its
   authoritative parents.
5. A missing reference or owner fact fails only the operation or claim that
   requires it, except where an already approved boundary declares a broader
   hard stop.
6. A demonstrated owner-level omission is corrected at its owner or remains
   unavailable; CAL shall not synthesize a replacement provenance bundle.

## WP3-OWNER-D008 — Producer-Payload Completeness

Status: approved `2026-08-24`

Question:

> Does the ordinary processed-timestream route require any additional
> observation-level producer payload beyond the products, sidecars, and
> manifests already identified?

Recommendation:

> No. Native \(x/r\), Tune, and detector/tone identity remain in the
> Tune/readout record and approved upstream interface. ALIGN owns acquisition
> timing and original occurrences. RTC owns its processing, grid, flags,
> diagnostics, and response in its product or sidecar. AST owns its realized
> RTC-grid coordinates while referencing RTC, telescope/observing, geometry,
> target/source center, frame, EOP, and refraction inputs. The SHA-bound
> matched APT row owns the selected detector geometry and `flxscale`; its
> manifest and referenced BEAM products retain Beammap, source, and calibrator
> ancestry. Existing telescope records own WVR opacity and telescope state.
> Frozen content-bound CAL authority owns the atmosphere operator and passband
> selection. The CAL product or compact sidecar owns the applied calibration
> and classification. The CAL-to-PTC boundary passes the calibrated signal,
> identity, required flags/validity, and parent references. No combined
> external-producer bundle is required. If AST cannot reconstruct
> scientifically valid pointing on the RTC grid from the referenced
> authorities, the previously approved observation-level hard stop remains.
> WP-3 should now produce only a compact reference-binding register or source
> manifest naming each authority, owner, reference/integrity rule, dependent
> operation, and failure scope. That register is source-closure documentation,
> not another runtime payload.

Owner response:

> approved

Disposition: **approved**.

Consequences:

1. WP-3 has identified no remaining scientific need for a new
   observation-level producer object on the ordinary processed-timestream
   route.
2. The approved AST pointing hard stop remains in force without requiring CAL
   or PTC to copy AST's input records.
3. The remaining WP-3 artifact is a compact reference-binding register that
   points to existing authority rather than serializing observation history.
4. Remaining work is mechanical artifact preparation, content binding, and
   contract verification; it does not authorize inferred facts or claims.
5. Final closure of `F-017` remains a clean-room re-audit disposition rather
   than a claim made by this owner packet.

## Remaining WP-3 Work

Prepare the smallest exact reference-binding register for the selected
ordinary processed-timestream route. It shall distinguish static/operator
authority from observation-instance realization and shall point to existing
products, sidecars, manifests, and package authority without duplicating their
contents. Optional roles may be explicitly not requested or unavailable; they
may not be supplied by inference. MAP-only roles remain deferred. Final
finding closure belongs to WP-7 clean-room re-audit.
