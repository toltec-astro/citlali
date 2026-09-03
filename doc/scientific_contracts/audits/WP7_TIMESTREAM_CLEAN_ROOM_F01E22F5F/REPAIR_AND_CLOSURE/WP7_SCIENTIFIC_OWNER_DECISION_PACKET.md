# WP-7 Repair Scientific-Owner Decision Packet

Status: **owner disposition complete; `WP7-OWNER-D001` approved;
`WP7-OWNER-D002` exact-byte recovery, WVR interpolation, and
unavailable-opacity disposition approved; `WP7-OWNER-D003` approved; D004
approved**

Authorized successor work remains bounded to D001 authority publication and
admission of D002's recovered exact bytes and approved WVR interpolation
and unavailable-opacity authorities, plus publication of the approved D003
classifier and the approved D004 logical-stream terminal clarification. No
external-consumer acceptance or implementation decision is inferred.

Date opened: `2026-08-25`

Scientific owner: Grant Wilson

Repair branch: `codex/scientific-contract-library`

Repair-base commit:
`96b3c66d096cd04f52a44b98c4b630df909eb752`

Frozen clean-room source commit:
`f01e22f5f8d8d92e49ae70312bdc59a81c1540ec`

Comparison inputs:

- `WP7_TWO_AUDIT_COMPARISON_REPORT.md`, SHA-256
  `446792d75d67ce25254af9832436c8f64bd8dcd1bc49f9676a2b1e8aba9e5396`;
- `WP7_TWO_AUDIT_FINDING_CROSSWALK.csv`, SHA-256
  `c424c881e39df9736419c623bb5dbbd56c305b203eeb181edec8ac92250b18d4`.

## 1. Purpose and boundary

This packet presents the smallest scientific-owner decisions and evidence
recovery needed to address the two-audit WP-7 comparison. It preserves the
locked Codex and ChatGPT audits unchanged and does not assign final closure
dispositions.

The bounded repair lane may:

1. publish already-approved authority that was not readable in the clean-room
   packet;
2. recover and admit exact digest-bound numerical objects;
3. request the missing observation-classifier decision;
4. decide the RTC logical-stream terminal boundary; and
5. correct explanatory owner-identifier traceability without changing the
   controlling owner ledger.

It does not authorize implementation work, numerical-algorithm changes,
SCI-MAP work, stronger response/covariance products, a generic exposure
quantity, or changes to the frozen audit artifacts.

## 2. Recovered authority and present evidence

### 2.1 Native producer interface

The clean-room audits saw only
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`, whose embedded status says
approval is pending and whose claim boundary says candidate semantics only.
The repository contains two later governing records that were bound by digest
but not readable in that clean-room packet:

- `WP2_FOLLOWUP_D011_OWNER_DECISION_2026-08-23.md` records the owner response
  `approved` and states that the exact v0.1/r0.1 artifact was approved on
  `2026-08-24`;
- `SCIENTIFIC_OWNER_APPROVAL_2026-08-24.md` promotes exact interface digest
  `f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`
  and explicitly says that it and `SOURCE_MANIFEST.md` govern the retained
  pre-promotion status line.

`producer_interfaces/v0.1/SOURCE_MANIFEST.md` independently records the packet
as approved and binds the decision and approval records. The native-interface
scientific decision is therefore already made. The WP-7 defect is a readable
authority-publication and precedence defect, not missing owner intent.

### 2.2 CAL numerical objects

The frozen CAL authority names but the clean-room packet does not contain:

| Object | Required SHA-256 |
| --- | --- |
| Atmosphere machine contract | `7a064ff768a3de4f427f1338d94ef6cb9026d248f3c3c816fc3dfc96d156e36a` |
| Atmosphere node table | `fd688a4cd3f46585b08631bc63a562aed482feb9b24ec9ee0071b70db7eb8a5f` |
| TolTECA-v1 passband set | `5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433` |

The original archived SCI-CAL-001 atmosphere-model task and its evidence
branch were recovered on `2026-08-25`. The machine contract and node table are
exact Git objects at Citlali evidence commit
`7156881bd1a47e8cece97b8c541a013c93ac03e1` on
`codex/sci-cal-001-atmosphere-operator`. Their SHA-256 values exactly match the
two frozen identities above.

The passband set is the following four-object set at TolTECA commit
`2791e6a1e6349ad1d3ac549a648f41cbc51b98c7`:

| Member | SHA-256 |
| --- | --- |
| `index.yaml` | `74465637294e536c44818099e4858a916fc6b9acbb1ea21b40427d15fb6532d5` |
| `data/a1100_passband.ecsv` | `13b8fd009bb8d7c375d3c46d21e26d0a779f7f00a949a2a5ccd619d1fe56fd72` |
| `data/a1400_passband.ecsv` | `a7b671d9f659cbc98dad99d3015ce81a3d7a3486c702819d9b3305703e7c682e` |
| `data/a2000_passband.ecsv` | `77e4b33c7bbc2c345ef94d41480d5fee5cb096d789f4fe78e1b4f80a37e0d6ff` |

Applying the original canonical member-name/digest aggregation rule in lexical
member order reproduces passband-set SHA-256
`5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`
over `1,297,803` member bytes.

Exact source paths, notices, checksums, and a standalone verifier are staged
under `RECOVERED_CAL_NUMERICAL_AUTHORITY_2026-08-25/`. This supersedes the
earlier filename-only search result. The similarly named
`toltec_beammap/src/toltec_sensitivity/model_passbands.npz` remains a different
object and is not substituted.

The frozen CAL sources also require a source-authorized interpolation between
valid bracketing WVR readings but do not select its exact method, version,
precision, or boundary/tie behavior.

### 2.3 RTC logical-stream terminal boundary

The approved PTC decision identifies the intended use as export of
RTC-conditioned timestreams for a companion ML mapmaker, while
`SCI-RTC-OWNER-001` remains open and requires any additional raw-domain
consumer, exact logical RTC-output subset, and lineage needs to be named.
Existing RTC authority defines the consumer-neutral atomic bundle as a
logical completion unit containing the conditioned outputs and RTC-owned facts;
it does not require one observation-sized serialized object or simultaneous
residence in memory or on disk. The consumer-neutral logical RTC stream and
PTC-disabled terminal route are already authorized and do not depend on naming
a further consumer.

### 2.4 RTC owner-identifier traceability

The controlling `SCI-RTC-OWNER-090--096` ledger rows are unambiguous. The RTC
scientific-rationale executive summary attaches those identifiers to a
different ordering of meanings. This is a source-resolved explanatory defect
and requires no new scientific choice.

## 3. Decision summary

| Decision | Present state | Recommendation | Owner response |
| --- | --- | --- | --- |
| `WP7-OWNER-D001` — native-interface authority publication | Scientific decision already approved; approval/precedence not readable in WP-7 | Approve a successor clean-room packet that admits the exact existing decision, approval, source manifest, README, and interface as one readable authority set while preserving the approved interface bytes and digest | **APPROVED — 2026-08-25** |
| `WP7-OWNER-D002` — CAL numerical authority | Three exact numerical identities recovered and staged; WVR interpolation and unavailable-opacity rules approved | Admit the verified bytes without regeneration or substitution and use the approved versioned interpolation and fail-closed unavailable-opacity rules | **APPROVED — RECOVERY, INTERPOLATION, AND UNAVAILABLE-OPACITY — 2026-08-25** |
| `WP7-OWNER-D003` — observation-wide opacity classifier | Approved policy intent recovered; no existing source defines `momentary` by duration or fraction | Approve `cal_wvr_observation_quality_mean_peak_v1`: time-weighted mean at `0.15`, tolerated peak through `0.175`, engineering support through `0.25`, exact coverage and failure classes | **APPROVED — 2026-08-25** |
| `WP7-OWNER-D004` — RTC logical-stream terminal boundary | Consumer-neutral RTC completion and the PTC-disabled terminal route are approved; the earlier repair wording incorrectly implied observation-sized materialization | Define the endpoint as completion of the logical RTC output stream over the declared domain plus finalization of observation-level RTC facts. Permit incremental consumption and optional explicit materialization; keep external-consumer acceptance outside WP-7 until separately named | **APPROVED — 2026-08-25** |

## 4. `WP7-OWNER-D001`: native-interface authority publication

### Question

Should the next clean-room generation admit the complete already-approved
producer-interface authority set, without modifying the exact approved
interface bytes?

### Recommendation

Yes. Preserve
`TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md` at exact approved SHA-256
`f9659b34a49a07d4287c4a70db798cdd2ec30049531da603fcca1e9d1fdd5969`.
Make the following files readable together in the successor packet:

1. `README.md`;
2. `SOURCE_MANIFEST.md`;
3. `WP2_FOLLOWUP_D011_OWNER_DECISION_2026-08-23.md`;
4. `SCIENTIFIC_OWNER_APPROVAL_2026-08-24.md`; and
5. `TUNE_READOUT_NATIVE_XR_PRODUCER_INTERFACE.md`.

The successor packet manifest shall state explicitly that the approval record
and source manifest promote the exact retained interface bytes and supersede
only the embedded pre-promotion status and candidate-only wording. No
scientific interface semantics, transform, convention, runtime payload, or
implementation claim changes.

This avoids creating a new interface digest merely to rewrite a historical
status line and gives a fresh clean-room reader the exact precedence evidence
that both audits lacked.

### Owner response

**Approved — 2026-08-25.**

The scientific owner approved the recommendation as written. This authorizes
the successor packet to publish the five-file authority set together and to
state the bounded precedence rule explicitly. It does not authorize changing
the approved interface bytes or digest, scientific interface semantics,
transform, convention, runtime payload, or implementation claim.

## 5. `WP7-OWNER-D002`: CAL exact numerical authority

### Question

Can the three exact digest-bound objects be recovered and admitted, and what
exact WVR time-interpolation rule governs valid bracketing readings? What
happens when no admissible telescope `tau225` state covers a detector sample?

### Recommendation

Use two gates in order.

#### Gate A — byte recovery

The project owner or authorized artifact custodian supplies candidate files.
For each file:

1. calculate SHA-256 without modifying it;
2. accept it only if the digest exactly matches the frozen identity;
3. record artifact filename, producing authority, units/schema, and license or
   redistribution boundary where applicable; and
4. admit the exact bytes and their checksum to the successor packet.

Do not regenerate the node table or passbands from the structural prose and
do not substitute a similarly named file. If any object is unrecoverable, the
ordinary nonzero-opacity route remains unavailable until the owner approves a
successor operator/passband generation with new identities and digests.

#### Gate A result — complete

The exact machine contract, node table, owner-decision record, TolTECA-v1
passband members, and their source-license notices were recovered directly
from the two recorded Git commits. The staging verifier confirms the complete
source inventory, every individual digest, the `1,297,803` passband-member
bytes, and the frozen passband-set aggregate. No object was reconstructed from
prose and no similarly named substitute was used.

#### Gate B — WVR interpolation authority

Approve `cal_wvr_tau225_linear_detector_time_v1` with the following exact
policy:

1. The input is the sequence of producer-valid LMT WVR `tau225` records in the
   current observation's `tel*.nc` stream, using each record's source time.
   The detector-reference sample time is mapped to that time basis by the
   approved SCI-ALIGN mapping. Records from another observation are never
   admitted.
2. At an exact matching source time, return the source value exactly. Multiple
   byte-identical records at one time collapse to one record; conflicting
   duplicate values make opacity unavailable at that time and for any bracket
   that would use it.
3. Otherwise, let `(t0, tau0)` and `(t1, tau1)` be the consecutive valid
   source records that bracket `t`, with `t0 < t < t1`. Evaluate
   `w = (t - t0) / (t1 - t0)` and
   `tau225(t) = tau0 + w * (tau1 - tau0)`.
4. Evaluate in IEEE-754 binary64, round-to-nearest ties-to-even, in the written
   subtraction, division, subtraction, multiplication, addition order. Each
   operation rounds separately; contraction to a fused multiply-add is not
   permitted.
5. The WVR source's declared validity governs whether a bracket and its gap
   are admissible. This rule adds no arbitrary elapsed-time threshold.
6. Do not extrapolate, clamp, hold an endpoint, inherit a prior-observation
   value, or interpolate through an invalid or conflicting record.
7. Record the two source record identities, source times and values, mapped
   detector time, interpolation weight, result, source-validity interval, and
   method identifier. An exact-match result records its single source record
   and equality disposition.

#### Gate B result — approved

The scientific owner approved
`cal_wvr_tau225_linear_detector_time_v1` as written on `2026-08-25`.

#### Gate C — unavailable telescope opacity

An explicit fail-closed rule is required even though SCI-CAL already requires
sample-local numerical support. Without it, an implementation could silently
substitute zero opacity, unity correction, a scalar header, an observation
summary, or an endpoint hold.

Approve `cal_wvr_tau225_unavailable_v1` with the following policy:

1. A detector sample has no admissible telescope opacity when it has neither
   a producer-valid exact-time WVR record nor a valid same-observation bracket
   under `cal_wvr_tau225_linear_detector_time_v1`.
2. Absent records, an unbracketed time, a bracket outside source-declared
   validity, a conflicting duplicate, or an unavailable time mapping yield
   `outside_supported_calibration` for the affected sample. Negative or
   non-finite opacity yields `invalid_atmosphere`.
3. The affected sample is excluded from calibrated-signal support and no CAL
   multiplier or calibrated value is emitted for it. Its upstream value may
   remain available only under its upstream identity and validity; it must not
   be passed onward or relabeled as an ordinary calibrated sample.
4. Do not substitute numeric zero, a unity correction, an observation mean or
   median, nearest-neighbor or hold-last/hold-next state, a scalar-header
   fallback, an AM climatology or profile, a configured default, or opacity
   from another observation.
5. Publish a machine-distinguishable cause from
   `wvr_tau225_absent`, `wvr_tau225_unbracketed`,
   `wvr_tau225_gap_outside_source_validity`,
   `wvr_tau225_conflicting_duplicate`, `wvr_tau225_negative`,
   `wvr_tau225_nonfinite`, or `wvr_time_mapping_unavailable`, together with
   the affected sample support and available source lineage.
6. Unsupported samples do not invalidate independently supported samples in
   the same observation. If no samples remain supported, CAL publishes the
   truthful no-calibrated-output state rather than an ordinary calibrated
   product. Observation-wide opacity classification remains governed by the
   separate D003 authority and cannot restore numerical support.

#### Gate C result — approved

The scientific owner approved `cal_wvr_tau225_unavailable_v1` as written on
`2026-08-25`.

### Owner response

**Gate A complete and Gates B--C approved — 2026-08-25.**

After the original atmosphere-model task was located, the scientific owner
directed the exact-object recovery and staging to proceed. This response
authorizes admission of the verified existing objects and approves
`cal_wvr_tau225_linear_detector_time_v1` and
`cal_wvr_tau225_unavailable_v1` exactly as recorded above. It does not select
the observation-wide classifier or alter the recovered numerical authority.

## 6. `WP7-OWNER-D003`: deterministic opacity classifier

### Question

What exact observation-wide classifier implements the approved `0.15`
guidance and `0.025` tolerance for momentary excursions?

### Recommendation

Approve `cal_wvr_observation_quality_mean_peak_v1` with the following exact
policy.

#### Population and observation window

1. The classified window is the closed interval from the first to the last
   detector-reference sample time belonging to the current observation,
   before CAL validity masking. A missing/non-finite endpoint or
   `t_end <= t_start` yields `opacity_quality_unavailable`.
2. Map that interval to the WVR source-time basis with the approved SCI-ALIGN
   authority and construct `cal_wvr_tau225_linear_detector_time_v1` from only
   the current observation's `tel*.nc` records. The complete classified
   interval must be covered. An empty source, missing bracket, disallowed gap,
   conflicting duplicate, or unavailable time mapping yields
   `opacity_quality_unavailable`; negative or non-finite required opacity
   yields `invalid_opacity_input`.
3. The ordered classifier breakpoints are the mapped observation endpoints
   plus every admitted WVR source time strictly inside the interval. Evaluate
   the approved interpolant at both endpoints. Do not resample, smooth,
   cadence-weight, fill a gap, or count detector samples as independent WVR
   evidence.

#### Summary and excursions

4. Compute the duration-weighted mean of the continuous piecewise-linear
   opacity over the complete interval. For chronological breakpoints
   `(t_i, tau_i)`, its area is the composite trapezoid
   `A = sum_i (t_(i+1)-t_i) * (tau_i+tau_(i+1)) / 2`, and
   `tau_mean = A / (t_end-t_start)`. Also record the minimum and maximum; the
   extrema occur at the breakpoints.
5. An excursion is one connected component of the interval on which
   `tau225(t) > 0.15`. Resolve threshold crossings analytically on the same
   linear segments. For a strict interior crossing of threshold `q`, evaluate
   `u = (q - tau_i) / (tau_(i+1) - tau_i)` followed by
   `t_cross = t_i + u * (t_(i+1) - t_i)`; an endpoint exactly equal to `q`
   is the crossing. Record every component's start, end, duration, and peak,
   plus total excursion duration, longest duration, duration fraction, count,
   and integrated excess `integral max(tau225(t)-0.15, 0) dt`. Partition at
   every threshold crossing and compute the excess with the same chronological
   trapezoid rule applied to the nonnegative endpoint excesses.
6. There is no additional duration, count, cadence, or fraction cutoff in v1.
   Here `momentary` has the exact combined meaning that the time-weighted mean
   remains at or below `0.15` and no instantaneous peak exceeds `0.175`.
   Thus excursion persistence affects the class through its contribution to
   the time-weighted mean without inventing a new unstated threshold.

#### Class mapping and boundary behavior

7. Assign exactly one class in this precedence order:

   - `invalid_opacity_input` if a required opacity state is negative or
     non-finite;
   - `opacity_quality_unavailable` if the complete window or required source
     coverage cannot be resolved under items 1--3;
   - `outside_supported_opacity` if complete valid coverage exists but
     `tau_max > 0.25`;
   - `science_qualification_eligible` if
     `tau_mean <= 0.15` and `tau_max <= 0.175`; or
   - `engineering_only` for every other completely covered, finite,
     nonnegative observation with `tau_max <= 0.25`.

   Equality is inclusive at `0.15`, `0.175`, and `0.25`. A value immediately
   above a boundary takes the next less-favorable class. These are
   operational quality classes, not achieved atmosphere-fidelity,
   observational-performance, `science-qualified`, or `calibrated-science`
   claims.

#### Determinism and output record

8. Parse source values, mapped times, and the decimal threshold strings
   `"0.15"`, `"0.175"`, and `"0.25"` into IEEE-754 binary64 with correct
   round-to-nearest, ties-to-even conversion. Evaluate interpolation,
   crossings, segment areas, chronological accumulation, duration, division,
   and comparisons in binary64 round-to-nearest ties-to-even with the written
   operation order and a rounding step after every elementary operation.
   Fused contraction and reassociation are prohibited. Exact binary64 equality
   receives the inclusive disposition above. A non-increasing breakpoint,
   non-finite intermediate, or summary outside the finite input range yields
   `opacity_quality_unavailable` with cause `classifier_numeric_failure`.
9. Publish the classifier and interpolation identifiers; observation and
   source identities; mapped interval; ordered input record identities and
   values; coverage/validity disposition; breakpoint count; minimum, maximum,
   mean, duration, and trapezoid area; the complete excursion inventory and
   aggregate statistics from item 5; threshold constants; precision rule;
   final class; and machine-distinguishable causes.

Sample-level numerical atmosphere support remains independent. An
observation-wide class neither fills an unsupported sample nor authorizes
numerical extrapolation. Conversely, an unavailable, invalid, or
outside-supported observation class does not erase independently supported
sample-level CAL results; their validity and limitations remain explicit.

This rule is recommended because it uses the already-approved `0.15`,
`0.025`, and `0.25` quantities directly and gives irregular WVR intervals
their actual duration. If the owner instead intends `momentary` to impose an
independent maximum duration or fraction, that new threshold must be supplied
and versioned rather than inferred from the approximately five-minute WVR
cadence.

### Owner response

**Approved — 2026-08-25.**

The scientific owner approved
`cal_wvr_observation_quality_mean_peak_v1` exactly as written, including the
explicit absence of an independent duration, count, cadence, or fraction
cutoff. This authorizes the successor packet to publish and bind the complete
classifier authority. It does not authorize implementation work, alter
sample-level atmosphere support, supply achieved atmosphere-fidelity or
observational-performance evidence, or decide D004.

## 7. `WP7-OWNER-D004`: RTC logical-stream terminal boundary

### Question

For WP-7 v0.1, what event completes the terminal RTC route without turning the
logical RTC product into a mandatory observation-sized serialized intermediate
or requiring acceptance by a specific external consumer?

### Recommendation

Define the WP-7 terminal endpoint as completion of the consumer-neutral logical
RTC output stream over the declared observation or processing domain, together
with finalization of the RTC facts that genuinely have observation-level
scope.

The logical RTC output stream is the ordered sequence of conditioned sample
outputs plus the RTC-owned facts needed to interpret them. Its elements may be
produced and consumed incrementally. They need not all coexist in memory, on
disk, or in one file, table, archive, or observation-sized object. Here,
“complete” describes scientific-content and lifecycle completion, not physical
materialization or serialization. Under this clarification, the existing RTC
and PTC terms “atomic bundle,” “publish,” and “export” mean successful
availability and completion of that logical content and its required facts;
they do not prescribe a storage form.

Normal operation may partition the stream into chunks, scans, or other
engineering units. An arbitrary chunk boundary is not scientific support. If
an RTC operator is defined over a scan, segment, or other declared domain, its
chunked execution must preserve the same domain-level scientific result as
non-chunked execution, subject only to the operator's declared numerical
tolerance. RTC may therefore retain the state, overlap, guards, learned
parameters, or deferred decisions needed across implementation chunks. A scan
boundary has scientific force only where the governing operator contract gives
it that force.

This boundary distinguishes three things:

1. downstream-visible incremental conditioned outputs and their required
   interpretive facts;
2. temporary RTC state used to preserve domain-level semantics across chunks,
   which is not thereby a persistent product; and
3. optional materialization requested explicitly for validation, audit,
   diagnostics, user output, or a separately approved downstream contract.

Persist only facts required by the RTC contract. Do not infer mandatory
per-chunk sidecars, per-sample provenance records, or repeated observation
history. Observation-level facts are finalized once at the terminal boundary
unless a governing requirement explicitly assigns a different cadence.

RTC owns the scientific identity, ordering, support, validity, response, and
lineage facts of its logical outputs. Each downstream consumer owns its own
admission and acceptance requirements. The terminal route therefore completes
successfully without CAL, PTC, or MAP, claims no external-consumer acceptance,
and requires no unnamed consumer.

If an external handoff is later placed inside a governed route, its separately
approved consumer contract must name the consumer and define the admitted
logical-stream subset, cadence or grid, coordinate and support semantics,
response, uncertainty, lineage, any serialization, failure behavior, and
acceptance criteria.

### Owner response

**Approved — 2026-08-25.**

The scientific owner approved the recommendation exactly as written. This
authorizes successor authority to clarify that RTC logical completion may be
incremental and that the existing terms “atomic bundle,” “publish,” and
“export” do not require observation-sized materialization or serialization.
It does not authorize implementation work, require a persistent RTC
intermediate, claim external-consumer acceptance, or name an additional
consumer.

## 8. Procedural disposition of `TS-A`

Do not retroactively revise either locked audit's `TS-A` result. Until
`WP7-OWNER-D001` is published in a successor readable packet, record the
native authority state as unresolved for closure purposes.

After the complete approval set is admitted, a new clean-room confirmation
shall determine `TS-A` under the repaired source graph. A permanent rule that
an admitted status conflict may be ignored for architecture-only readiness is
not recommended; the bounded repair should remove the conflict instead.

## 9. Source-resolved RTC traceability correction

After owner disposition of this packet, revise only the explanatory
`OWNER-090--096` list in
`SCI-RTC/v0.1/src/scientific-rationale.tex` so each identifier paraphrases its
controlling ledger row:

| Owner ID | Controlling meaning |
| --- | --- |
| `SCI-RTC-OWNER-090` | Canonical ordinary paired response is exactly `I_2 tensor L_Pi`, with identical ordinary operators and zero cross-coordinate numerical branches |
| `SCI-RTC-OWNER-091` | Learn retains coordinate-specific and genuinely joint evidence with exact origin and cause |
| `SCI-RTC-OWNER-092` | Resolve forms a cause-preserving union in one immutable pair plan; Apply does not discover evidence or mutate the plan |
| `SCI-RTC-OWNER-093` | Accepted hard evidence pair-flags support while preserving direct, joint, and inferred cause distinctions |
| `SCI-RTC-OWNER-094` | Raw validity/evidence, accepted pair action, and conditioned modification/availability/response remain distinct layers |
| `SCI-RTC-OWNER-095` | `r` is an equal contamination sensor; expected optical response is not automatic pathology; protection participates in Resolve |
| `SCI-RTC-OWNER-096` | Canonical action is symmetric after separately admitted affine corrections, with bounded level-shift/notch/donor exceptions and downstream CAL/PTC/VAL ownership preserved |

The controlling ledger, equations, requirements, predictions, and numerical
behavior remain unchanged. Rebuild and verify both RTC PDFs and every affected
source manifest after the correction.

## 10. Explicitly retained limitations

The following comparison issues remain typed limitations and are not opened
by this packet:

- conditional ALIGN/AST realization authorities;
- complete source/beam-to-PTC response without `K_up->CAL`;
- numerical and stronger total covariance without an admitted `C_Y` and
  complete nuisance/selection authority; and
- any generic usable-exposure quantity.

Each requires a separately named deliverable and owner decision before its
scope may expand.

## 11. Successor repair and validation gates

A successor generation is ready for clean-room confirmation only when:

1. the owner responses in Sections 4--7 are recorded in a separate dated
   disposition or explicitly incorporated into an approved successor record;
2. the native approval set is readable and its exact internal hashes verify;
3. the CAL numerical objects are either admitted at exact frozen digests or
   the ordinary nonzero-opacity route remains explicitly unavailable;
4. the WVR interpolation, unavailable-opacity disposition, and opacity
   classifier have exact owner authority or their dependent results remain
   explicitly unavailable;
5. the RTC logical-stream terminal boundary is explicit;
6. the RTC explanatory owner-identifier list matches the ledger;
7. package verifiers, PDF generation checks, source-manifest checks, and
   packet verification pass with zero missing required evidence; and
8. a new archive, source commit, report filenames, and SHA-256 identities are
   used. No locked WP-7 artifact is overwritten.

Implementation conformance, observational validation, achieved performance,
production readiness, SCI-MAP, and stronger tier delivery remain separate
claims.
