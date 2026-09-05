# PDF decision-register and crosswalk check

Draft checked: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.2**  
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
- UFC-OD-007 remains open in both records and now explicitly covers the
  disjoint structural-target/payload-fidelity partition. The r0.2 repair is
  recorded as an initial-draft consistency repair, not an owner decision.

The checked source hashes were:

| File | SHA-256 |
| --- | --- |
| `OWNER_DECISION_LEDGER.md` | `3f9b405272bcdf003987b80abe148f99cbd5400fa7be84fff5daaeeef7f358ce` |
| `src/common/assumptions.tex` | `3844877f22a6ba9388a3b21c7cf478d602206774f583d55e4c5448938e067581` |

The requirement/prediction checks also passed:

- 25 unique canonical requirements and 25 requirement-crosswalk rows.
- 16 unique canonical predictions and 16 prediction-crosswalk rows.
- `CROSSWALK.md` has exactly the requested three mapping columns:
  `Requirement | Scientist-facing source | Engineering interpretation`.

Their checked hashes were:

| File | SHA-256 |
| --- | --- |
| `CROSSWALK.md` | `7d2be3cb5d5e4a1fd84e47c1a4b2c93044ff1428dcf82099b59261a75adfb255` |
| `PREDICTION_CROSSWALK.md` | `8189244a0b5da6e7c3c1a8cc6cbff9bb0ae4d27586f5be726d9146f1fc30b83d` |
| `src/common/requirements.tex` | `9355b660002215ba7791677d0c1b1877cea033feb7a9ac439f251e073f1ef932` |
| `src/common/edge_cases.tex` | `fb39a76ac0e7fa57667fd320e7150c16b407544ee8358b17570f86fb1187c4c4` |
