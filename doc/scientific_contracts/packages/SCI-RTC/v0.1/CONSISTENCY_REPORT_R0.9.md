# SCI-RTC v0.1/r0.9 Consistency Report

Date: 2026-08-20

Status: fresh implementation-blind consistency review of the r0.9 final
candidate. This review assesses internal scientific-contract coherence and
artifact integrity only. It is not an implementation-conformance,
observational-validation, production-readiness, or scientific-owner-freeze
disposition.

## Authority and scope

- Binding r0.9 input: `SCIENTIFIC_OWNER_DECISIONS_R0.9.md`, SHA-256
  `90cad00151d975e0bb2a432c907f4a2198a1f3645f52c645c7e71cfa58ac57cb`.
- Binding r0.8 Decision 9 remains intact.
- Review surfaces: scientific rationale, six-file shared normative core,
  engineering conformance guidance, owner ledger, decision log, crosswalks,
  verifier, and canonical PDFs.
- Implementation, runtime behavior, sibling-package content, and production
  evidence were not used to resolve scientific choices.

## Decision consistency

| Decision | Rationale | Normative core | Engineering check | Ledger/log | Result |
| --- | --- | --- | --- | --- | --- |
| Context-neutral operation availability | explicit | DEF/ASM/REQ | explicit | 076 / D010 | consistent |
| Context → plan → realized record | explicit | DEF/REQ | explicit | 077 / D011 | consistent |
| Consumer-neutral bundle | explicit | DEF/EQ/REQ | explicit | 078 / D012 | consistent |
| Mapping/pair/independent validity | preserved | DEF/REQ | explicit | 079 / D013 | consistent |
| Explicit coordinate authority | preserved | DEF/ASM/REQ | explicit | 080 / D014 | consistent |
| Typed non-finite handling | explicit | REQ | explicit | 081 / D015 | consistent |
| Covariance-claim disclosure | explicit | DEF/EQ/ASM/REQ | explicit | 082 / D016 | consistent |
| Actual despiking plus compact normal summaries | explicit | DEF/ASM/REQ | explicit | 083 / D017 | consistent |

No active source retains a role-specific operation partition, a mandatory
normal-run event/donor manifest, an undisclosed covariance-claim rule, silent
zero coercion, or a detection-only definition of selected despiking. Historical
records remain unchanged where they document superseded revision history.

## Mechanical and PDF evidence

- `src/verify_contract.py` passes.
- Inventory is unchanged: 38 definitions, 37 displayed equations, 12
  assumptions, 108 requirements, and 71 predictions.
- Owner ledger is sequential through `SCI-RTC-OWNER-083`: 63 open, one
  conditional, 14 resolved, and five deferred.
- Both Tectonic builds complete without overfull/underfull layout warnings.
- Scientific rationale: 14 pages, US Letter, unencrypted, no forms or
  JavaScript; SHA-256
  `1b2257a141d53d83bb7e8bee0adda0762183a7196b310b23e030a457efc584c7`.
- Engineering conformance: 43 pages, US Letter, unencrypted, no forms or
  JavaScript; SHA-256
  `30ff35fe8d10a7b591e61c53a17d3663e1145329981e132cba7d74465423dcc0`.
- All 57 pages were rendered with Poppler and visually inspected. No clipping,
  overlap, orphan page, malformed table, or unreadable figure was found.
- Extracted PDF text contains the application-context lifecycle,
  consumer-neutral bundle, covariance-disclosure, actual-despiking, compact
  spike-summary, and typed non-finite clauses.

## Disposition

The r0.9 candidate is internally consistent and ready for explicit
scientific-owner freeze disposition. Open numerical/methodological ledger
entries remain open and continue to block only their named operations or
claims. After owner freeze, RTC is ready to enter the RTC--CAL--PTC coherence
review; this report does not perform that cross-package review.

## Post-review owner disposition

After this review passed, the scientific owner stated exactly, “Freeze SCI-RTC
v0.1/r0.9.” The reviewed candidate therefore became frozen scientific
authority on `2026-08-20`; implementation conformity remains unassessed. The
two PDFs were republished with status-only title-page changes, and their frozen
artifact hashes are recorded in `pdf/README.md`. No normative content,
identifier, inventory, or owner-ledger state changed after this consistency
review.
