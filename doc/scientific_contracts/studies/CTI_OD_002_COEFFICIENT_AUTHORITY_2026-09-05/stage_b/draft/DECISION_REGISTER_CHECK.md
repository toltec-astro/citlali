# PDF decision-register, owner-response, and crosswalk check

Draft checked: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.4**  
Check date: **2026-09-06**  
Result: **PASS**

The compact decision register is singly sourced from
`src/common/assumptions.tex` and appears in the complete engineering PDF. It
was explicitly checked against `OWNER_DECISION_LEDGER.md`:

- UFC-REC-001--008 are decided in the PDF register and individually present in
  the ledger. UFC-REC-008 contains the exact admitted requirement 072 and
  090/094/095 provenance without claiming authority for another route.
- UFC-OD-001--014 occur exactly once in each register, remain open, and
  describe equivalent blocked outputs.
- UFC-Q-001 occurs in each register with the same open actual-owner question
  and bounded Registry/profile activation effect.
- UFC-DEF-001--005 remain deferred and separate from recovered and open states.
- Owner support in principle remains review evidence; no proposal is labeled
  adopted or frozen.

| Checked file | SHA-256 |
| --- | --- |
| `OWNER_DECISION_LEDGER.md` | `9bd339c5ea4a75111209ffdfb420106c72a885350d91a34efae872d946be353d` |
| `src/common/assumptions.tex` | `7561dbdac512d0c17e98609ee5dedc72810035d0ea99a727b7a96e18eed88a70` |

`OWNER_REVIEW_1_RESPONSE.md` contains the complete stable inventory
OR1-01--OR1-05, OR1-S1--OR1-S3, and OR1-A1. Each disposition points to
canonical source plus requirements, predictions, or cases where applicable.
Its checked SHA-256 is
`ce3cbe50c7482691722d85451a3bb0879be6951992353f4f93ec2469381cb809`.

The bounded r0.4 consistency correction also passed explicit checks:

- The shared structural row applies corruption only to authoritative identity
  or binding records, including the payload-publication identity/binding
  record.
- Once publication identity and binding are valid, authoritative corrupt,
  digest-failing, or decoder-invalid bound bytes give
  (Q_{m integrity}=mathbf F); unavailable or unexplained integrity
  evidence gives (mathbf U); contradictory integrity evidence gives
  (mathbf C).
- Bound bytes never become structural merely because they are corrupt.
- UFC-REQ-017, UFC-MIX-008, UFC-PRED-009, both crosswalks, the profile record,
  owner ledger, owner response, and blocked-definition record use the same
  partition without changing any proposal state.

The requirement, prediction, and mixed-case checks passed:

- 25 unique canonical requirements and 25 unique requirement-crosswalk rows.
- 16 unique canonical predictions and 16 unique prediction-crosswalk rows.
- 10 unique canonical worked cases and 10 unique case-crosswalk rows.
- `CROSSWALK.md` contains only title/status context and the exact
  three-column schema `Requirement | Scientist-facing source | Engineering interpretation`.
- UFC-REQ-015 maps to the scientist-facing four-axis and finalization
  explanation; UFC-REQ-016 maps to the scientist-facing nonexceptionability
  explanation.

| Checked file | SHA-256 |
| --- | --- |
| `CROSSWALK.md` | `6b3ee8d86b485cbca07ef7db46258042b5db3c243d64e6f2a8767a4758eaf2d5` |
| `PREDICTION_CROSSWALK.md` | `b8dbdcfb477090533ba00ed76993493a59593c2c661d2cbeb76e882b9c19e69a` |
| `src/common/requirements.tex` | `ca78a7a77aea54832b5baf69d11c16d9603ffd06ae3711cec2914ad55158533c` |
| `src/common/edge_cases.tex` | `e47b0c18016193711275c50bed1a2ab02fd98b0d11b904782f0d95c1c4741151` |

