# SCI-VAL v0.1 — Prior-Work Recovery

This recovery record follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[accepted pilot process](../../../PILOT_PROCESS_REVIEW_2026-08-16.md). It is
an internal Stage A artifact and is not part of the proposed scientific-author
packet.

Status: recovery complete; r0.1 scope review absorbed; scientific-owner
confirmation of the revised Scope Brief and packet remains pending

Investigator/date: Codex manager, `2026-08-20`

Scope revision examined:
`codex/scientific-contract-library@a6e1022323853ebe8efb251c3590540f9981a577`,
plus the exact historical references below

## Search Coverage

Recovery began with
[`PRIOR_WORK_REGISTRY.md`](../../../PRIOR_WORK_REGISTRY.md) and then searched:

- the governing program, pilot review, package index, living architecture,
  scientific conventions, product contracts, configuration contracts, ADRs,
  status, and retained-debt records;
- current SCI-CAL, SCI-MAP, SCI-BEAM, SCI-RTC, and SCI-PTC packages;
- historical audit framework and coordination refs, including every available
  `SCI-VAL-001-XAUD-001--011` handoff;
- current and historical RTC/PTC flagging, validity, support, influence,
  replacement, masking, map-admission, and product-schema material;
- later commits containing `SCI-VAL-001` after the registry snapshot; and
- the current implementation areas named by the historical inventory, solely
  to establish the Stage A interface and quarantine boundary.

No `codex/audit-sci-val-001` branch, frozen SCI-VAL independent core, approved
SCI-VAL contract, or dedicated reusable VAL method note was found. No Unity
system was accessed. No external scientific reference is required to preserve
the recovered project decisions; Stage B may identify a need for a bounded
general reference only after owner approval.

## Exact Recovery Snapshots

| Shorthand | Exact reference | Role |
| --- | --- | --- |
| `LIB` | `a6e1022323853ebe8efb251c3590540f9981a577` | Current scientific-contract library and living refactor snapshot examined |
| `MAIN` | `46ad23888a40f5102cdfd50c06e49a549bdf8a20` | Historical integrated application/documentation snapshot |
| `COORD` | `8c581bfb26f01b187f4f1e0565f4457bcc25f099` | Latest recovered audit ledger and all eleven VAL handoffs |
| `FRAME` | `dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` | Historical audit framework |
| `METHOD` | `4a7916a8ec459f050de236211e5bacfc95695412` | Reusable-method registry searched; no VAL method found |
| `RTC-AUDIT` | `3319d7424c732c1c9fc300c336e4d428e6f91068` | Historical RTC core/audit evidence |
| `PTC-AUDIT` | `01ee247461d6c19bc4db81ccac4fec21af162c88` | Historical PTC core/audit evidence and D002 amendment route |

These identities preserve recovery provenance. They are not all scientific
authority and they do not enter the author packet merely because they were
searched.

### Content-bound historical handoff inventory

All paths below are at `COORD`.

| Handoff | SHA-256 | Scientific status | Recovery disposition |
| --- | --- | --- | --- |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-001.yaml` | `f1542e62eef09b8fa9b122178c10ec629005222c5cd93f231deae85147e309b3` | Approved MAP owner-decision boundary | adopt through current MAP authority; exclude handoff bytes |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-002.yaml` | `6680e13467dfb083e61bb0594aa3d5cc388bc7c107fb3e7de8ba3d1a457679e0` | Post-core MAP evidence | defer to conformity; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-003.yaml` | `35f879195acb9934e1d8e4c19cf01dfaed1c123fde6473fcca0cf6f04d81f655` | Post-core CAL evidence | defer to conformity; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-004.yaml` | `741529bd435c0f8128bbc7ca1e9d0da6fee59928e049bec8026b257c9c37467a` | Post-core ALIGN evidence | defer to conformity; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-005.yaml` | `416d98a12c31eeeb1c82b9ac3c2873e134578e48768398516e1af81ca4cfc357` | Superseded AST evidence | supersede by X006; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-006.yaml` | `7f0244d71780e05ed9d9220fed076bdbfcab558332b5290164343f7e9ae4cdbd` | Corrected post-core AST evidence | defer to validation; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-007.yaml` | `ca07517ca225bb70d2c767b6dabd7b327a320675394bf9cae81e009cee016a36` | Post-core RTC evidence | defer to conformity; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-008.yaml` | `7f85804f47c9d3d9f3295bcb4702f36c8286373d23c7604ef4291d4346f9c6d0` | Superseded PTC claim | supersede by X009; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-009.yaml` | `a6c5b45884e98de5d502afb1deacc58c8eb5b39cbe772f44dce2bc598f0bdf5d` | Owner-amended PTC decision | adopt through current PTC decision log; exclude handoff bytes |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-010.yaml` | `3b4cf9fd9467c9d60d8fd1154515003fba52272b3350e1788025d014bff06cc7` | Bounded TEL-input structural evidence | defer to producer/consumer audit; exclude |
| `doc/audits/handoffs/SCI-VAL-001/SCI-VAL-001-XAUD-011.yaml` | `73e3ef8d6746549da394a79e311f6fe53dfe8b930b68cc15fcb254ce280af1cc` | Post-core MAP-003 dependency evidence | defer to conformity; exclude |

## Recovered Materials

| Material and exact reference | Classification | Scientific content already available | Limitations or conflicts | Disposition |
| --- | --- | --- | --- | --- |
| `LIB:doc/scientific_contracts/README.md` and `PILOT_PROCESS_REVIEW_2026-08-16.md` | Governing program authority | Stage A/Stage B firewall, anti-repetition rule, shared-core and freeze process | Process only; no VAL scientific answer | cite |
| `LIB:doc/SCIENTIFIC_CONVENTIONS.md` | Governing shared authority | Identity, units, missing/disabled/unavailable distinctions, flags/fit-validity authority, one-way state | Mixes validated present behavior and labeled successor contracts | abstract into sanitized conventions |
| `LIB:validation/product_contracts.json` | Executable product authority and implementation-informed scope evidence | Current product identities and persisted missing-value/flag conventions | Schema describes current products; it cannot choose VAL science and is unsuitable for blind authorship | defer to later conformity; exclude from author packet |
| `LIB:doc/adr/0009-science-map-bundle-admission-and-validity.md` | Accepted scientific decision | Upstream eligibility, estimator contribution, two support decisions, and final raw MAP validity are distinct; finite values cannot promote invalid parents | MAP-specific final validity remains MAP-owned | adopt the boundary; abstract exact statements |
| `LIB:doc/scientific_contracts/packages/SCI-MAP/v0.1/` | Current contract-development authority | MAP owns its upstream-admission policy, contribution, normalization, support, response, and output validity; VAL may provide shared evaluation mechanics | Scientific authority is not frozen and several MAP owner decisions remain open | cite only stable approved boundary through sanitized extract |
| `LIB:doc/scientific_contracts/packages/SCI-RTC/v0.1/` | Frozen r0.9 scientific authority and adjacent input | RTC separates origin, direct validity, causes, masks, local supports, influence, response, and downstream facts; direct representative synthesis/replacement is not independent exposure | RTC ownership does not transfer a downstream use policy to VAL | abstract exact interface; do not admit full draft |
| `LIB:doc/scientific_contracts/packages/SCI-PTC/v0.1/` | Owner-approved scope/decisions and current freeze-candidate input | Staged `fit_invalid`, `postfit_output_reject`, `weight_only`, fit-excluded/application, support, response, and coefficient roles; PTC owns its local support composition and use policies | PTC scientific authority is not yet explicitly frozen; older wording incorrectly deferred cause-to-support interpretation to VAL | abstract approved scope decisions; route wording cleanup as follow-up |
| `LIB:doc/scientific_contracts/packages/SCI-CAL/v0.1/` | Current contract-development input | CAL owns detector binding, factor/domain validity, calibration availability, response, and signal-support identity | CAL scientific authority and several owner questions remain open | abstract boundary only |
| `COORD:.../SCI-VAL-001-XAUD-001.yaml` | Approved MAP owner decision handoff | Exact separation among upstream eligibility, contribution, exposure, two supports, and final MAP validity | Historical packaging; current MAP contract/ADR is the preferred authority | adopt via current authority; do not send handoff |
| `COORD:.../SCI-VAL-001-XAUD-009.yaml` | Approved PTC owner-decision handoff | `fit_invalid`, `postfit_output_reject`, and `weight_only` are distinct; only fit-support changes require refit/invalidation | Supersedes XAUD-008; current PTC decision log is preferred | adopt via current owner-approved decision; do not send handoff |
| `COORD:.../SCI-VAL-001-XAUD-005.yaml` | Superseded historical evidence | AST validation-routing proposal | Superseded by XAUD-006 | supersede and exclude |
| `COORD:.../SCI-VAL-001-XAUD-008.yaml` | Superseded historical evidence | Earlier overbroad PTC transitive-invalidation claim | Superseded by XAUD-009 owner amendment | supersede and exclude |
| `COORD:.../SCI-VAL-001-XAUD-002--004,-006--007,-010--011.yaml` | Historical audit/scope evidence | Identifies MAP, CAL, ALIGN/AST, RTC, TEL-input, and MAP-003 validity questions and failure modes | Post-core evidence cannot prescribe independent science; several source packages have since evolved | defer to later audit/validation; exclude from author packet |
| `MAIN:doc/RTC_FLAGGING_AUDIT_2026-03-16.md` | Historical audit/repair evidence | Shows why flag causes, masks, processing order, and eligibility cannot be conflated | Implementation-specific, repaired in places, and not scientific authority | defer and exclude |
| Current `rtcproc.h`, `ptcproc.h`, `naive_mm.h`, `jinc_mm.h`, output and flagging code | Implementation-informed scope evidence | Confirms multiple Boolean flags, detector flags, finiteness checks, masks, weights, and consumer-local selections exist | Current encoding and behavior cannot define the correct VAL contract | quarantine in internal dossier; exclude |
| `5c630912...:handoff/SCI_ALIGN_001_PTC_SCAN_METADATA_DEFECT_2026-08-08.md` | Historical repair/validation evidence | Shows a scan-metadata/product-identity defect route | Explicitly not a validity authority and not integrated as pre-core VAL material | defer and exclude |

## Recovery Synthesis

### Questions already answered

1. A flag describes a cause; it is not automatically a mask, invalidity,
   weight, or universal eligibility action. Current RTC owner-approved scope
   and contract development establish this boundary.
2. Direct validity, numerical validity, origin/synthesis, replacement,
   operator support, transitive influence, response status, uncertainty
   availability, and consumer eligibility are distinct facts.
3. An exact representative occurrence that is directly ALIGN-synthesized or
   RTC-replaced is not an independent detector exposure. Nonrepresentative
   causal influence remains traceable and is not universally converted to
   ineligibility.
4. PTC `fit_invalid`, `postfit_output_reject`, and `weight_only` decisions are
   distinct. Only an actual fit-support change requires refit or fitted-state
   invalidation.
5. MAP owns estimator contribution, normalization support, science-policy
   support, and final raw output validity. VAL supplies upstream eligibility;
   it does not replace those MAP decisions.
6. Invalid payloads are excluded before their numerical value is used. An
   eligible non-finite required payload fails or makes the affected result
   unavailable. Finiteness alone never establishes eligibility.
7. Operator controls such as source masks are distinct from acquisition or
   scientific validity.

### Reusable definitions and reasoning

The recovered project decisions support a compositional model with four
separate objects:

1. producer facts and typed causes;
2. producer-local support and validity;
3. a named consumer-use policy; and
4. a cause-preserving eligibility disposition.

This structure is suitable for the sanitized author packet. Exact equations,
truth domains, use-specific decisive-predicate evaluation, aggregation,
missing-fact behavior, and falsifiable edge cases remain genuine Stage B
work.

### Conflicts and unresolved choices

- Historical XAUD-008 proposed blanket recomputation/transitive invalidation
  after late PTC decisions. Owner-approved XAUD-009 supersedes it: only a
  typed fit-invalid support change requires refit. The earlier claim is not
  admitted.
- Historical and current products often expose Boolean flags, while the
  recovered scientific boundary requires typed cause and decision-stage
  meaning. The contract must remain representation-independent; exact storage
  is engineering-owned.
- Upstream packages preserve noncenter transitive influence but do not select
  one universal downstream action. The r0.1 review recommends a mandatory
  structural base gate plus eight named profiles; this remains an open owner
  disposition rather than recovered authority.
- Missing required validity/influence facts must not be silently interpreted
  as false. The r0.1 review recommends `decision_unavailable` for an undecided
  required gate while allowing a known decisive false predicate to establish
  `ineligible` after the decision domain is known; this remains an open owner
  disposition.
- The current PTC rationale is a committed freeze candidate but lacks an
  explicit scientific-owner freeze statement. VAL can use its approved scope
  decisions conditionally, not promote the full draft to frozen authority.

### Material excluded from authorship

All source paths, line-level behavior, Boolean encodings, findings, defects,
repairs, test fixtures, re-audits, Unity records, validation results,
production restrictions, and historical implementation verdicts remain
outside the author packet. The eleven XAUD files and the historical audit
ledger are recovery evidence, not Stage B inputs.

### Genuinely new work

Stage B must independently derive:

1. the exact typed fact, cause, use, and disposition domains;
2. an order-independent, idempotent, cause-preserving composition algebra and
   use-specific decisive-predicate rules;
3. eligibility behavior for unavailable, invalid, disabled, missing,
   non-finite, synthesized, replaced, influenced, fit-excluded,
   output-rejected, and weight-only states;
4. sample-to-detector and detector-to-sample aggregation rules without false
   promotion or overbroad invalidation;
5. policy identity, lifecycle, provenance, and exact replay obligations;
6. the boundary between direct exclusion, operator-local support, consumer
   eligibility, and final product validity;
7. failure scope and atomicity for missing or contradictory facts; and
8. falsifiable predictions and validation layers for the logical contract.

No new estimator, threshold, replacement, filtering, mapmaking, or
uncertainty algorithm belongs in this work.

## Proposed Author Reference Packet

Only these sanitized package-local files are proposed:

1. `SCOPE_BRIEF.md`;
2. `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`;
3. `AUTHOR_CROSS_PACKAGE_BOUNDARY_PROFILE.md`; and
4. `DECISION_LOG.md` after every admitted decision is owner-approved.

The exact bytes and hashes will be frozen in `AUTHOR_PACKET_MANIFEST.md` only
after owner review. No historical handoff, current source, audit, full
contract draft, test, validation record, or product schema is proposed.

## Investigator Attestation

Prior work was recovered before new derivation was commissioned. Existing
approved distinctions are reused rather than re-derived. Conflicts and
supersessions are explicit. Implementation behavior was used only to define
scope and quarantine, and was not promoted to scientific authority. No Stage
B author has been launched.
