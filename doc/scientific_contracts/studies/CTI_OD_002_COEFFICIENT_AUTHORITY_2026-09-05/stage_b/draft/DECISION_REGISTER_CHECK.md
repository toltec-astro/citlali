# PDF decision-register and crosswalk check

Draft checked: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.1**  
Check date: **2026-09-05**  
Result: **PASS**

The PDF decision register is singly sourced from
`src/common/assumptions.tex` and therefore renders in both views. It was
explicitly checked against `OWNER_DECISION_LEDGER.md`:

- UFC-OD-001 through UFC-OD-014 occur in both the full ledger and the common
  PDF register.
- UFC-Q-001 occurs in both and carries the same open actual-owner question and
  blocked Registry/profile activation.
- The common register’s UFC-REC-001--007 decided range corresponds to the
  seven individually listed recovered decisions in the ledger.
- The common register’s UFC-DEF-001--005 deferred range corresponds to the
  five individually listed deferred items in the ledger.
- All entries remain separated as decided, open, or deferred; no open proposal
  was rendered as adopted.

The checked source hashes were:

| File | SHA-256 |
| --- | --- |
| `OWNER_DECISION_LEDGER.md` | `3c1fb532a23014f0f98e40e6b614ce152a39458bbbaec98dfce3a79afa453c1d` |
| `src/common/assumptions.tex` | `6c85d8cf9118ef2074e0b620665caa0c3ba0add469c3e80b55ab33ca778ea26b` |

The requirement/prediction checks also passed:

- 25 unique canonical requirements and 25 requirement-crosswalk rows.
- 16 unique canonical predictions and 16 prediction-crosswalk rows.
- `CROSSWALK.md` has exactly the requested three mapping columns:
  `Requirement | Scientist-facing source | Engineering interpretation`.

Their checked hashes were:

| File | SHA-256 |
| --- | --- |
| `CROSSWALK.md` | `e41e33cffb28702c28e4210aa38aa91729fe670ded48d8d061fff3c3bff34742` |
| `PREDICTION_CROSSWALK.md` | `c7e23c45cd94076aa896681ead41a160a4a46bfee597d62f9cc069c91b8b7d22` |
| `src/common/requirements.tex` | `6981313de81b51a8b5e25f0057299589f762742b1aa350049a27be8b0cd8470f` |
| `src/common/edge_cases.tex` | `f5f6ef5d4a82334e88f5c754c4ca0e1c19c0ec4184f6c87c875d189ca064ddbf` |

