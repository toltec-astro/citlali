# SCI-RTC — Prior-Work Recovery

This recovery record follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md). It is an
internal Stage A artifact and is not automatically part of the scientific
author packet.

Status: reviewed by the Codex manager; pending scientific-owner review

Investigator/date: Codex manager, `2026-08-17`

## Exact Recovery Snapshots

- scientific-contract library:
  `codex/scientific-contract-library@c2a01842e6b9158c3c33b9bc8f4ae55884300662`;
- integrated application discovery line:
  `origin/codex/refactor-mainline@46ad23888a40f5102cdfd50c06e49a549bdf8a20`;
- RTC independent audit:
  `codex/audit-sci-rtc-001@3319d7424c732c1c9fc300c336e4d428e6f91068`;
- later coordination corpus:
  `codex/register-sci-map-003-audit-disposition@8c581bfb26f01b187f4f1e0565f4457bcc25f099`;
- learned-sampling design:
  `codex/design-rtc-learned-sampling@cbb676d84bc58da4239a906a420a04a326968309`;
  and
- later RTC/PTC coordination:
  `codex/coordinate-rtc-ptc-queue@8fc3b3dc549532f254bc814ef76f9a606c2a8059`.

These identities locate recovery evidence. They are not automatic scientific
authority, implementation-conformity evidence, or production authorization.

## Search Coverage

Recovery began from the program
[`PRIOR_WORK_REGISTRY.md`](../../../PRIOR_WORK_REGISTRY.md) and re-verified:

- the library charter, pilot lessons, package templates, and CAL/MAP/BEAM
  boundaries;
- the complete frozen `SCI-RTC-001_INDEPENDENT_CORE.tex`;
- owner decision `SCI-RTC-001-D001--D004` and the later phase-zero
  downsampling amendment;
- the owner-approved learned-sampling plan and durable ADR;
- the four RTC incoming handoffs from CAL, ALIGN, AST, and MAP-003;
- all later RTC learned-sampling audit, repair, re-audit, and acceptance
  records, solely to determine status and whether they changed scientific
  authority;
- the current application architecture, scientific conventions, raw
  timestream configuration/provenance authority, product contracts, and the
  apparent RTC numerical and orchestration surface;
- PTC owner decisions needed to state the RTC-to-PTC boundary; and
- the frozen SCI-BEAM r0.3 signal/response boundary and the current SCI-CAL
  and SCI-MAP package interfaces.

No Unity system was contacted. No reduction, test, numerical validation,
source audit, repair, or scientific derivation was repeated.

## Reference Digests

| Object | SHA-256 |
| --- | --- |
| `SCI-RTC-001_INDEPENDENT_CORE.tex` | `d6cf49d1a5e17754c55cc4f2c8f4b4f5e276755f247496df888581d890be80b7` |
| `SCI-RTC-001_OWNER_DECISION_2026-08-08.md` | `5f5ebb52735a70510d75f5a6954ef825fc425c00e43ab0aadf2363f2e0609723` |
| phase-zero downsampling amendment | `421f86d4fa17ee03b9ee772c8b6adb6f2d47f3df0e9e873b038ba4889332bb69` |
| `RTC_LEARNED_SAMPLING_PLAN_2026-08-09.md` | `3eccc03f95a1dc14aeb909e850ffb7d885ca50f67efdbb16247257919e0bef53` |
| learned-sampling ADR | `c1088a4ae54bcf2a9b5b9d690c2f4296a797ccdac792ad5151e9a2133643351e` |
| integrated `SCIENTIFIC_CONVENTIONS.md` | `1970d7e31ccbcf77f890ea7c0854fde59d25b2fc745f909a74150360605d3049` |

## Recovered Materials

| Material and exact reference | Classification | Scientific content already available | Limitations or conflicts | Disposition |
| --- | --- | --- | --- | --- |
| Library charter and pilot review | Governing process authority | Recovery-first workflow, information firewall, shared authority, review, QA, and stopping rules | Process rather than RTC science | Adopt |
| `3319d742...:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex` | Reusable independent science | Ordered conditional operator; identity and stage taxonomy; calibration/replacement conversion; FIR/IIR/notch/mask/decimation semantics; conditional moments; local response; aliasing; validity; atomic output bundle; analytic tests | Contains audit-era status prose, provisional upstream meanings, an obsolete mean-downsampling alternative, and a calibrated-path assumption that cannot govern Beammap | Cite through a binding supersession cover; do not repeat derivation |
| `8c581bfb...:SCI-RTC-001_OWNER_DECISION_2026-08-08.md` | Approved scientific decisions | D001 replacement/synthesis influence eligibility; D002 complete-response-or-unavailable; D003 immutable stage identities; D004 exact bounded filter/multirate state | Does not prove implementation or close findings | Adopt in supersession cover |
| `8c581bfb...:SCI-RTC-001_CORE_PHASE_ZERO_DOWNSAMPLING_OWNER_AMENDMENT_2026-08-11.md` | Approved scientific amendment | Phase-zero point selection is authoritative; arithmetic-mean downsampling is not; exact phase, support, representative time, state, transfer, and availability are required | Physical event timing remains unavailable; does not establish an executable candidate | Supersede the broader core alternative |
| `cbb676d8...:RTC_LEARNED_SAMPLING_PLAN_2026-08-09.md` and ADR 0009 | Approved scientific/architectural design | Fixed and learned modes; metadata bootstrap; maximum-safe-reduction objective; exact analytical beam/FIR/alias evaluation; common immutable apply plan; restart and convergence boundaries | Numerical tolerances and Stage B execution remain unapproved; contains implementation staging detail | Abstract binding science into the supersession cover; defer implementation detail |
| Learned-sampling repairs, re-audits, and owner acceptance through `8fc3b3dc...` | Historical audit/repair evidence | Confirms later technical work did not request a new scientific decision and retained the approved plan | Candidate-specific findings and status cannot enter independent authorship | Defer to later conformity program; exclude |
| CAL/ALIGN/AST incoming handoffs `SCI-RTC-001-XAUD-001--003` | Historical post-core evidence | Identify dependencies on calibration, synthesized-sample support/influence, coordinate validity/frame, and detector binding | Source observations and findings do not establish RTC science | Abstract questions and boundaries only; exclude raw handoffs |
| MAP-003 incoming handoff `SCI-RTC-001-XAUD-004` | Historical post-core evidence | Requires exact tracer/response parentage or honest response unavailability | A later audit claim; not RTC contract authority | Abstract parentage question; exclude raw handoff |
| Integrated `ARCHITECTURE.md`, config/product contracts, RTC headers, and orchestration paths at `46ad2388...` | Implementation-informed scope evidence | Apparent signal stages, controls, products, diagnostics, requested/effective/observation/realized lifecycle, and downstream consumers | Current behavior cannot determine correct science and is not audited here | Internal dossier only; exclude |
| `doc/raw_timestream_config_transition.md` and `doc/RTC_FLAGGING_AUDIT_2026-03-16.md` | Mixed architecture and historical audit evidence | Current configuration boundary and earlier flagging questions | Implementation history, not independent scientific authority | Abstract lifecycle boundary; exclude raw documents |
| Current SCI-CAL v0.1 package | Conditional adjacent scientific package | CAL owns calibration-factor meaning, target atmosphere application, and uncertainty lineage | Scientific authority remains unfrozen and several physical/numeric questions are open | Use only as an explicit conditional producer boundary |
| Frozen SCI-BEAM v0.1/r0.3 | Governing adjacent scientific authority | Primary Beammap standardized detector signal is raw `Delta f/f`; complete conditioned response and causal state are upstream inputs | Does not define RTC algorithms | Adopt the exact interface constraint in sanitized form |
| SCI-MAP v0.1 and approved PTC decisions | Conditional adjacent authority | MAP consumes calibrated conditioned samples; PTC is optional after RTC and owns cleaning/coefficient/covariance meaning | Does not establish RTC response or validity | Abstract producer/consumer boundary only |

## Recovery Synthesis

### Questions already answered

1. **Core mathematical structure.** Conditional on a complete realized state,
   RTC is a factorized affine operator; data-derived selection makes the
   unconditional operation generally nonlinear and selection-dependent.
2. **Response.** The truthful RTC response includes every enabled
   response-changing stage, detector mixing, state, mask, and phase, or is
   explicitly unavailable. A partial kernel is not the complete response.
3. **Influence eligibility.** An output influenced by ALIGN synthesis or RTC
   replacement is scientifically ineligible under approved D001, even if it
   is finite; its causes and support remain traceable.
4. **Stage identity.** Outer, inner, full, mini, diagnostic, simulated, and
   processed products are distinct immutable stages with explicit parents.
5. **Multirate identity.** Filter coefficients, normalization, state, edges,
   phase, rate, support, and alias behavior are scientific state, recorded at
   full precision once per coherent segment.
6. **Downsampling.** Phase-zero point selection is authoritative. Arithmetic
   averaging is not an allowed substitute.
7. **Learned sampling.** An optional learned mode may select maximum safe
   reduction from metadata and analytical response constraints, but its first
   applied form is one common immutable observation plan; numeric tolerances
   remain owner decisions.
8. **Lifecycle.** Requested, effective, observation-resolved, learned/resolved
   where applicable, and realized state flow one way.
9. **Output sufficiency.** TOD values alone are not a complete scientific RTC
   output. Identity, response status, support/influence, flags/causes,
   validity inputs, uncertainty availability, provenance, and diagnostics are
   required companions appropriate to the declared product role.

### Reusable definitions, equations, and reasoning

The independent core is adequate for the central operator, response,
covariance, validity, and falsification derivations. A new author should
consolidate and teach that material rather than derive it again. The binding
cover must supersede:

- its optional arithmetic-aggregate downsampling language with phase-zero
  point selection;
- its weaker zero-direct-weight replacement wording with the stronger
  owner-approved transitive-influence ineligibility rule;
- its generic calibrated-path assumption with product-role-specific signal
  domains, including frozen BEAM raw `Delta f/f`; and
- its pre-library upstream status prose with explicit current package
  boundaries and unavailable states.

The learned-sampling plan supplies the already-approved adaptive-planning
reasoning. Only the unresolved numerical tolerances, product-role policies,
and truly missing scientific choices require new work.

### Conflicts and unresolved choices

1. **Signal domain.** The earlier core describes an initially calibrated
   `mJy/beam` path, but frozen SCI-BEAM requires raw `Delta f/f` for its
   primary standardized detector maps. The v0.1 scope must preserve both as
   distinct product roles and prohibit silent cross-use.
2. **Calibration and donor transfer.** CAL owns factor science while RTC owns
   the realized conditioning order. Cross-detector replacement must be
   expressed in the active unit domain and either follow imported calibration
   or prove exact equivalence. Frozen SCI-BEAM gives legacy `responsivity` no
   canonical role, so raw-domain donor transfer requires separate authority
   and cannot be inferred from that field.
3. **Numeric learned policy.** Response-loss/broadening, alias, sampling,
   factor/order/cost, and fallback thresholds remain open owner decisions.
4. **Policy families.** The core supplies constraints but does not select
   universal despike detection, donor, source-protection, FIR, IIR, notch,
   edge, or recovery policies. The contract must distinguish selected policy
   from mathematical obligation without importing current defaults.
5. **Physical time.** The admitted assigned grid can support software response
   accounting, but physical detector integration-event timing and absolute
   correction remain unavailable pending ALIGN authority.
6. **Uncertainty.** Conditional propagation is defined; complete input,
   selection, response, nuisance, and model covariance is often not supplied.
   The contract must preserve typed unavailability rather than inventing a
   mandatory dense covariance product.

### Material excluded from authorship

- all Citlali source, current interfaces, configuration defaults, product
  schemas, and implementation-specific stage order;
- the internal dossier and complete prior-work record;
- every RTC/CAL/ALIGN/AST/PTC/MAP audit report, handoff, finding, repair,
  re-audit, test, reduction, validation result, Unity record, and production
  status;
- learned-sampling candidate source, technical metrics, and repair history;
- active ALIGN work and any inferred physical timing or absolute-placement
  solution; and
- model memory or unlisted external sources as substitutes for the exact
  approved packet.

### Genuinely new work

After owner approval, a fresh implementation-blind author must:

1. reconcile product-role-specific raw and calibrated signal domains within
   one ordered RTC operator without redefining CAL or BEAM;
2. turn the reusable core and supersessions into the library's science-team
   rationale and engineering-conformance views;
3. state how selected despike, mask, filter, state, edge, and fallback policy
   enters the operator without inventing current defaults;
4. integrate fixed and learned sampling under the phase-zero contract while
   leaving unresolved numerical tolerances explicit;
5. define exact availability, identity, support/influence, response,
   uncertainty, and atomic-output obligations for each product role; and
6. produce stable requirements, predictions, owner questions, and a complete
   crosswalk without assessing implementation.

## Proposed Author Reference Packet

The proposed packet contains only:

1. [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md);
2. the pair of [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md)
   and the exact independent core at
   `3319d7424c732c1c9fc300c336e4d428e6f91068:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex`;
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md).

The packet is proposed, not approved. Exact hashes and exclusions are in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## Investigator Attestation

Prior work was recovered before fresh derivation was commissioned. The
independent core and later owner decisions are reused rather than rewritten.
Implementation behavior was used only to establish scope and has not been
promoted to scientific authority. No implementation-blind author has been
launched.
