# WP-7 Repair Scientific-Owner Decision Packet

Status: **active owner disposition record; `WP7-OWNER-D001` approved;
`WP7-OWNER-D002` exact-byte recovery complete; D002 temporal rule and
D003--D004 pending**

Authorized successor work remains bounded to D001 authority publication and
admission of D002's recovered exact bytes. No WVR, classifier, consumer, or
implementation decision is inferred.

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
4. decide the RTC-terminal consumer boundary; and
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

### 2.3 RTC-terminal consumer

The approved PTC decision identifies the intended use as export of
RTC-conditioned timestreams for a companion ML mapmaker, while
`SCI-RTC-OWNER-001` remains open and requires any additional raw-domain
consumer, exact paired-bundle subset, and lineage needs to be named. The
consumer-neutral RTC terminal product itself is already authorized and does
not depend on naming a further consumer.

### 2.4 RTC owner-identifier traceability

The controlling `SCI-RTC-OWNER-090--096` ledger rows are unambiguous. The RTC
scientific-rationale executive summary attaches those identifiers to a
different ordering of meanings. This is a source-resolved explanatory defect
and requires no new scientific choice.

## 3. Decision summary

| Decision | Present state | Recommendation | Owner response |
| --- | --- | --- | --- |
| `WP7-OWNER-D001` — native-interface authority publication | Scientific decision already approved; approval/precedence not readable in WP-7 | Approve a successor clean-room packet that admits the exact existing decision, approval, source manifest, README, and interface as one readable authority set while preserving the approved interface bytes and digest | **APPROVED — 2026-08-25** |
| `WP7-OWNER-D002` — CAL numerical authority | Three exact numerical identities recovered and staged; WVR temporal method not selected | Admit the verified bytes without regeneration or substitution. Separately select a versioned WVR interpolation rule | **BYTE RECOVERY COMPLETE — WVR DECISION PENDING** |
| `WP7-OWNER-D003` — observation-wide opacity classifier | Exact classifier incomplete | Approve one versioned deterministic classifier record defining every required statistic and boundary; infer no default from `momentary` or the numerical-operator support | **PENDING DECISION** |
| `WP7-OWNER-D004` — RTC-terminal consumer boundary | Terminal publication approved; additional consumer unnamed | For WP-7 v0.1, define successful publication of the complete consumer-neutral RTC bundle as the endpoint. Keep companion-ML acceptance and handoff schema outside WP-7 until separately named | **PENDING DECISION** |

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
exact WVR time-interpolation rule governs valid bracketing readings?

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

Approve one versioned rule that states:

- the WVR source record and time coordinate;
- the interpolation family;
- precision and deterministic evaluation convention;
- valid bracketing and equality behavior;
- maximum permitted gap or a declaration that source validity alone governs;
- missing/nonfinite input behavior; and
- prohibition of endpoint extrapolation and prior-observation inheritance.

No interpolation family is recommended here because the admitted authority
does not supply the scientific basis for choosing one.

### Owner response

**Gate A complete — 2026-08-25. Gate B pending scientific-owner decision.**

After the original atmosphere-model task was located, the scientific owner
directed the exact-object recovery and staging to proceed. This response
authorizes admission of the verified existing objects; it does not select a
WVR interpolation rule or alter the recovered numerical authority.

## 6. `WP7-OWNER-D003`: deterministic opacity classifier

### Question

What exact observation-wide classifier implements the approved `0.15`
guidance and `0.025` tolerance for momentary excursions?

### Required owner fields

The decision shall define:

1. classifier identity and version;
2. the admitted opacity population and validity filtering;
3. the observation-wide summary statistic;
4. the exact excursion statistic, time window, duration/count/fraction rule,
   and treatment of irregular sampling or gaps;
5. the complete class set and mapping;
6. inclusive/exclusive behavior at `0.15`, `0.175`, `0.25`, and every other
   class boundary;
7. missing, empty, conflicting, and nonfinite behavior;
8. deterministic precision and tie rules; and
9. the exact output record: summary, excursions, class, causes, and authority
   identity.

Sample-level numerical atmosphere support remains independent. No classifier
decision may authorize numerical extrapolation beyond the operator domain.

### Recommendation

Approve these fields in one compact, versioned classifier record, then have
SCI-CAL reference it. Do not distribute the algorithm across narrative,
requirements, and an implementation default.

### Owner response

**Pending.**

## 7. `WP7-OWNER-D004`: RTC-terminal consumer boundary

### Question

For WP-7 v0.1, is successful publication of the complete consumer-neutral RTC
bundle the terminal scientific endpoint, or must the packet also authorize a
specific external consumer handoff?

### Recommendation

Select terminal publication as the WP-7 endpoint. The route then:

1. publishes the complete required RTC bundle;
2. terminates successfully without CAL, PTC, or MAP;
3. claims no external-consumer acceptance; and
4. leaves any companion ML mapmaker identity, subset, coordinate/response
   needs, serialization, and acceptance criteria to a separately approved
   consumer contract.

This preserves the prior intended use without turning an unnamed future
consumer into a required authority for the RTC product itself.

If the external handoff is required inside WP-7 instead, the owner response
must name the consumer and approve its exact paired-bundle subset, coordinate,
support, response, uncertainty, lineage, serialization, failure, and
acceptance requirements.

### Owner response

**Pending.**

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
4. the WVR interpolation and opacity classifier have exact owner authority or
   their dependent results remain explicitly unavailable;
5. the RTC consumer boundary is explicit;
6. the RTC explanatory owner-identifier list matches the ledger;
7. package verifiers, PDF generation checks, source-manifest checks, and
   packet verification pass with zero missing required evidence; and
8. a new archive, source commit, report filenames, and SHA-256 identities are
   used. No locked WP-7 artifact is overwritten.

Implementation conformance, observational validation, achieved performance,
production readiness, SCI-MAP, and stronger tier delivery remain separate
claims.
