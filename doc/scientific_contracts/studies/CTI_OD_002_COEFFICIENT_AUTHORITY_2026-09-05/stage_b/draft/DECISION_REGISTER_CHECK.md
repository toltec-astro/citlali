# PDF decision-register, owner-response, and crosswalk check

Draft checked: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Check date: **2026-09-06**  
Result: **PASS**

The compact decision register is singly sourced from src/common/assumptions.tex and appears in the complete engineering PDF. It was explicitly checked against OWNER_DECISION_LEDGER.md:

- UFC-REC-001--008 are decided in the PDF register and individually present in the ledger. UFC-REC-008 contains the exact newly admitted requirement 072 and 090/094/095 provenance without claiming authority for another route.
- UFC-OD-001--014 occur exactly once in each register, remain open, and describe equivalent blocked outputs.
- UFC-Q-001 occurs in each register with the same open actual-owner question and the same bounded Registry/profile activation effect.
- UFC-DEF-001--005 remain deferred and separate from recovered and open states.
- Owner support in principle is retained as review evidence in the full ledger; no proposal is labeled adopted or frozen.

| Checked file | SHA-256 |
| --- | --- |
| OWNER_DECISION_LEDGER.md | c6707de6913860668ae853f3ca2fb6e3ed376ead9602679bc2e8cf3757740a93 |
| src/common/assumptions.tex | 8ec6e02297b3bb9f1bc1dd7ea0cad0c9131d418ad4553120e3dbaa157eab521d |

OWNER_REVIEW_1_RESPONSE.md contains the complete stable inventory OR1-01--OR1-05, OR1-S1--OR1-S3, and OR1-A1. Each disposition points to canonical source plus requirements, predictions, or cases where applicable. Its checked SHA-256 is 7042d940a7ad9876c0a7b7b0818496605a8e89dfd9e88446a976e09e01e33c04.

The requirement, prediction, and mixed-case checks also passed:

- 25 unique canonical requirements and 25 unique requirement-crosswalk rows.
- 16 unique canonical predictions and 16 unique prediction-crosswalk rows.
- 10 unique canonical worked cases and 10 unique case-crosswalk rows.
- CROSSWALK.md contains only title/status context and the exact three-column schema Requirement | Scientist-facing source | Engineering interpretation.
- REQ-015 maps to the scientist-facing four-axis and finalization explanation; REQ-016 maps to the scientist-facing nonexceptionability explanation.

| Checked file | SHA-256 |
| --- | --- |
| CROSSWALK.md | a94ed180e5636c14a104e7891344e756b1eea4c693d979c67ed7a2f9937d582c |
| PREDICTION_CROSSWALK.md | 376c393a75ae1aec52fa85ca3b9ad88903fdfe6d39223479b29b9a47fba92af5 |
| src/common/requirements.tex | dd3b069d8a410b59509c09b6a7c1c47d0109d5dd438a32f2974434ad34b210ce |
| src/common/edge_cases.tex | 29fadddecd7c99223a1491b2599494744bd35f7c504f08852b775bd6538c86e8 |
