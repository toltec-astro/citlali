# SCI-ALIGN To SCI-AST Scientific Boundary

Boundary profile: `SCI-ALIGN_TO_SCI-AST`

Version/revision: `v0.1/r0.1`

Boundary identity: `SCI-ALIGN_TO_SCI-AST v0.1/r0.1`

Scientific owner: Grant Wilson

Status: Stage B targeted draft r0.3 pending scientific-owner approval

Approval date: pending

Package roles: SCI-ALIGN produces the immutable occurrence/time/mapping and
aligned observing-state relation; SCI-AST consumes that relation and owns the
coordinate transformation; SCI-MAP owns estimator-specific sample-to-pixel
deposition/gridding.

Compatibility/supersession rule: compatibility requires this exact profile
identity plus a declared compatible version/revision and preservation of every
required identity, semantic state, ownership boundary, and typed-availability
rule. Similar shape or field names do not establish compatibility. A successor
must name the profile revision it supersedes and provide an explicit semantic
mapping for every changed, removed, or newly required item.

Prepared: `2026-08-22`

Purpose: define the package-neutral, representation-independent information
that SCI-AST receives unchanged from SCI-ALIGN. This file contains no
package-relative links and is intended to be copied byte-for-byte into both
Stage B author packets after exact content binding and scientific-owner packet
approval.

## Ordinary Chain And Reference Meaning

The ordinary detector-signal order is

`(I,Q)^acq -> Tune/readout -> (x,r)^acq -> SCI-ALIGN -> (x,r)^A`.

Tune/readout owns the native `IQ`-to-`x/r` transformation. ALIGN consumes the
exact native paired `x/r` occurrences and applies one temporal mapping. ALIGN
does not align `I/Q` and then assume that Tune conversion commutes with
interpolation, resampling, or continuity synthesis.

“Detector-reference” means the selected detector-stream reference interface,
its clock relation, and its assigned grid. It never means a reference
detector. The reference interface defines the nominal slot relation; it does
not by itself establish physical event time, physical integration, or acquired
exposure.

The reference interface is denoted `i_ref = D`. Corrected native time uses
`t^ref`, with admitted offsets `delta_(i->ref)` and
`delta_(ref->ref) = 0`. Symbols `x` and `r` are reserved exclusively for the
paired physical KID readout coordinates. Across the package, `s` is the stable
SCI-ALIGN detector-reference slot, `j` is local storage row only, `n` is the
stable SCI-RTC output sample, `d` is a detector occurrence or stable detector
identity, and `p` is a map pixel.

## Stable Slot And Sample Identity

`s` is the stable ALIGN detector-reference slot identity, and `(observation,
s)` is the stable cross-package ALIGN identity. `j` is only a local storage
row derived from `s` and an explicitly selected window; it is never external
identity. `n` is reserved for stable RTC output-sample identity.

Every materialized view carries the explicit relation from stable slot `s` to
local row `j`. Neither AST nor any later contract may reconstruct `s` from a
local array row. Slicing, repartitioning, or changing storage layout preserves
`(observation,s)` unchanged.

Circular fields use the shared shortest signed difference interval
`[-P/2,P/2)`. An exactly antipodal interpolation is unavailable unless an
explicit unwrap authority resolves the path; choosing the represented endpoint
does not resolve the physical ambiguity.

## Canonical Paired-Coordinate Invariant

For one realized ALIGN operator `A_ALIGN`,

`x^A = A_ALIGN x^acq` and `r^A = A_ALIGN r^acq`.

The paired outputs have identical native source occurrences, temporal mapping,
weights, assignment residuals, and detector-reference slot identities
`(observation,s)`. ALIGN
does not mix `x` with `r`, fill either coordinate from the other, or give the
pair different source relations or different origin/synthesis state. The pair
has one common source relation and origin/synthesis state, while numerical
validity and payload availability remain coordinate-specific.

No paired relation crosses a Tune state or Tune/readout mapping-revision
boundary unless a separately named authority defines that cross-boundary
operation and its response and uncertainty. Absent that authority, the
affected paired output is typed unavailable rather than synthesized across the
boundary.

This paired invariant governs the detector-signal source relation; it does not
make a finite numerical `x` or `r` payload an AST coordinate prerequisite. AST
consumes the exact detector occurrence, time, mapping relation, aligned
observing-state fields, and their origin/support/validity/response/provenance
facts. Coordinate-specific `x/r` numerical validity and payload availability
remain transferred producer facts for binding and downstream policy, but do
not by themselves establish AST coordinate invalidity.

An AST coordinate may remain available when `x` is invalid or non-finite, `r`
is invalid or unavailable, the detector signal is excluded from a later
estimator, or no numerical detector-signal product is requested, provided the
exact detector occurrence, time relation, geometry, telescope fields, and
other AST-required inputs remain valid.

## Exact Unchanged ALIGN-to-AST Transfer

The rows below form one immutable transfer bundle, not a menu of independently
substitutable objects. Every required payload is either present with its exact
identity and semantics or carries typed availability and reason. A similarly
shaped value, field, record, plan, or mapping is never a substitute.

| Transfer item | Exact content AST receives unchanged | Required interpretation and availability |
| --- | --- | --- |
| Reference grid and slot identity | Observation identity; selected detector-stream reference-interface identity `i_ref = D`; parent identity; grid and plan identity/version; stable detector-reference slot `s` and identity `(observation,s)`; cadence; phase; nominal slot time `t_s`; nominal output-cell support `I_s` | A regular slot is a nominal output relation only. It does not prove a native acquisition, physical integration, or acquired exposure. `j` is local storage row only; `n` is reserved for RTC output-sample identity. |
| Native and assigned time relation | Native interface identity; native row/occurrence identity; producer-declared native acquisition/integration support; corrected native event time `t^ref` when available; exact detector-reference occurrence time/slot identity `t_s`; assignment residual; offset-record identity | Exact occurrence time `t_s` is grid/time identity, not an interpolated observing-state field. Nominal slot time, corrected native event time, residual, native support, nominal cell support, physical acquired exposure, and valid-original exposure remain distinct. A producer telescope timestamp may remain native metadata but is not a competing current-sample time. Physical-time meaning is no stronger than producer authority. |
| V0.1 interface-offset record | Producer authority; observation-constant value; positive-add sign convention; seconds unit; reference interface `i_ref`; `delta_(i->ref)` and `delta_(ref->ref)=0`; application stage/count; validity domain; uncertainty or bound; residual diagnostics; typed availability | The constant offset is an admitted v0.1 interface model, not a universal clock fact. It is applied exactly once. Drift or state dependence makes the model unavailable outside its declared domain; ALIGN neither fits nor implies a time-varying correction without separate authority. |
| Exact source relation | For every aligned field or detector pair: exact native source rows/occurrences, temporal weights, residuals, stable slot identities `(observation,s)`, support limits, and mapping-exception identity, either expanded or exactly reconstructible from the compact generative record | AST does not choose or alter interpolation and does not reconstruct source relations from clocks or local row `j`. Unavailable expanded detail must be distinguished from exactly reconstructible detail. |
| Paired detector `x/r` source relation and producer facts | `x^A = A_ALIGN x^acq`; `r^A = A_ALIGN r^acq`; identical source occurrences, temporal weights, residuals, slot identities, common source-relation identity, and common origin/synthesis state; coordinate-specific numerical-validity and payload-availability states | No `x/r` mixing, no filling one coordinate from the other, and no crossing Tune or mapping-revision boundaries without separately named authority. AST consumes the source relation and producer facts without requiring either numerical payload to be finite or present. Signal-coordinate invalidity alone does not invalidate an otherwise defined AST coordinate. |
| Aligned boresight and current observing-state fields | Boresight direction; current elevation; current azimuth where required; every other declared telescope/observing-state field needed by AST geometry or field rotation, each evaluated or mapped at exact occurrence time `t_s`; stable field identity and registry version; unit; unchanged producer frame; topology; declared per-field operator; exact source mapping and support; origin; validity; uncertainty availability | These fields describe the science occurrence at the exact detector-reference time. ALIGN maps declared producer meaning; AST receives it without reinterpretation. Field rotation uses these aligned occurrence-local fields. Exact telescope/boresight/HWPR registry entries remain owner questions rather than inferred defaults. |
| Producer-selected pointing-correction records | Exact correction-record identity; parent and selection identity/version; vector meaning; sign and basis; unit; native support mode and times; covariance availability; admitted detector-reference time relation | The pointing-support producer selects the correction record and native support. ALIGN supplies the admitted relation of that record to detector-reference time. AST may interpolate the correction only within the producer-selected native support and never extrapolates. Current science-sample elevation is not inferred from a bracketing correction record. Geometry, field rotation, correction composition, and their uncertainty remain AST-side or selected-geometry authority. |
| Origin, synthesis, and exposure | Original, synthesized, unavailable, original-invalid, and guarded state; continuity availability; source-relation identity; physical acquired exposure `e^acq_sd`; valid-original exposure `e^vo_sd`; zero added acquired exposure for synthesized/surrogate and missing/unoccupied support; later retained/use-qualified facts | Physical acquisition follows original integration support independently of payload validity. An original-invalid occurrence may have nonzero physical acquired exposure and zero valid-original exposure. Origin is never upgraded; later guards or use policy do not rewrite acquisition. Coordinate-specific invalidity does not authorize filling from its pair. |
| Interpolation versus continuity synthesis | The realized declared per-field alignment operator and, separately when authorized, the detector-signal continuity-surrogate identity, exact source rows/weights, causal support, response, uncertainty status, synthesized origin, and zero added acquired exposure | Ordinary field interpolation does not authorize detector-signal synthesis. A surrogate is a separately authorized signal-domain operator and never creates an independent acquisition. |
| Interval identities | Stable physical-scan identity; processing-chunk identity; science-window identity; context-window identity; output-selection identity; each with explicit half-open support and realized state | The identities remain distinct and are not silently renumbered, merged, or inferred by AST. Producer `Hold`/state semantics remain an explicit question. |
| Mapping validity and support | ALIGN-local mapping validity, source support, gap/guard facts, per-coordinate numerical validity, per-field payload availability, and typed reasons | These are producer/transformer facts, not downstream eligibility. VAL evaluates a named-use owner’s policy; AST does not reinterpret ALIGN-local validity as universal admissibility. |
| Residual and uncertainty | Assignment residual; timing uncertainty/bound payload or typed availability; interpolation or synthesis model-uncertainty payload or typed availability; mapping-uncertainty payload or typed availability; selection-uncertainty payload or typed availability | Unavailable is not zero. Residuals are diagnostics of the declared relation, not proof of absolute clock truth. |
| Conditional response and covariance | Exact realized ALIGN operator identity; conditional temporal/nonstationary response payload or typed availability; propagated covariance when admitted inputs and the selected tier support it; mapping/model/selection covariance payload or typed availability | ALIGN supplies only its conditional mapping response. The quantitative response/covariance tier remains an owner choice and blocks only claims requiring that tier. |
| Complete provenance | Requested, effective, observation-resolved, and realized plan identity; offset authority and application record; field-registry version; mapping/operator version; Tune/readout mapping-revision boundary identity; stable-slot-to-local-row relation for each materialized view; source-relation identity; exception/expansion identity; lifecycle and generation provenance | The record must identify the complete realized relation without AST consulting implementation, tests, products, or historical evidence. |

## Five Distinct Causes

The immutable transfer preserves five non-substitutable causes:

1. **Detector-occurrence identity:** whether the exact detector occurrence,
   parent, stable slot `(observation,s)`, and source relation are known.
2. **Signal-coordinate validity:** coordinate-specific numerical validity and
   payload availability for detector `x` and `r`.
3. **ALIGN mapping validity:** whether ALIGN's occurrence/time/source mapping
   and required support are valid.
4. **AST coordinate validity:** whether AST's required observing-state,
   geometry, correction, frame, and projection inputs define the coordinate.
5. **Downstream use-specific eligibility:** whether a named-use owner's policy,
   evaluated by its designated evaluator, admits the occurrence.

None is an alias for another. Signal-coordinate invalidity does not by itself
establish AST coordinate invalidity; AST coordinate availability does not
rescue an invalid ALIGN mapping; and neither authorizes downstream use.

## AST Constraints

SCI-AST consumes this immutable detector occurrence/time/mapping and
observing-state relation independently of whether finite numerical `x/r`
payloads are available. It may construct and correct coordinates, apply
separately authorized pointing corrections, bind detector and coordinate
identities, propagate astrometric uncertainty, construct the TAN/WCS
astrometric projection, and assign its own coordinate-validity facts.

The two telescope families remain distinct. ALIGN supplies aligned boresight,
exact detector-reference occurrence time plus elevation/required azimuth and
other observing-state fields evaluated or mapped at that time. Separately,
the pointing-support producer selects the
correction record and native support; ALIGN supplies its admitted time
relation; AST may interpolate the correction only inside that selected support.
Field rotation uses the aligned science-occurrence observing state, not an
elevation inferred from the pointing-correction record. Field rotation,
geometry, and correction composition do not move into ALIGN.

SCI-AST may not:

- reconstruct clocks or offset application;
- choose, replace, or repeat ALIGN interpolation, assignment, gap, guard, or
  continuity-synthesis policy;
- infer missing event, frame, topology, unit, field, response, covariance, or
  uncertainty semantics;
- replace a missing payload, parent, plan, version, field, record, or mapping
  with a similarly shaped object;
- extrapolate pointing support beyond the producer-selected native support;
- collapse ALIGN-local mapping validity into downstream eligibility; or
- treat nominal slots or synthesized support as evidence of physical
  integration or acquired exposure.
- reconstruct stable ALIGN slot `s` from local storage row `j`.

## Projection Ownership And Terminology

SCI-AST owns the **TAN/WCS astrometric projection** from sky or tangent
coordinates to continuous map coordinates, including its WCS identity,
coordinate validity, and astrometric uncertainty.

SCI-MAP owns the **sample-to-pixel deposition/gridding projection** that uses
an exact AST coordinate parent and a MAP-owned kernel, support, boundary,
normalization, and estimator plan to construct `G_pi`. AST's TAN/WCS
astrometric projection is not `G_pi`, projection/deposition weights, numerical
sample contribution, or map support. AST coordinate facts do not independently
select or authorize estimator-specific `G_pi`.

An AST implementation may materialize `G_pi` only for an exact MAP-owned
projection request bound to the AST coordinate parent, WCS/pixel geometry, MAP
projection-plan identity, kernel/support/boundary convention, and declared
response/covariance role. Materialization does not transfer scientific
ownership or permit AST to select the deposition family.

## Explicit Questions; No Defaults

The exact telescope/boresight field registry, exact HWPR registry, producer
`Hold` and state-transition semantics, detector-signal surrogate family and
thresholds, count/duration/fraction gap limits, quantitative response tier,
and covariance/model/selection-uncertainty tier remain explicit owner
questions. No implementation convention or plausible value supplies a
default. Each unresolved item blocks only the affected field, operator, or
claim; it does not invalidate unrelated exact relations already available.

This boundary records scientific ownership and required information only. It
does not assess implementation conformity, perform validation, freeze either
package, or authorize production use.
