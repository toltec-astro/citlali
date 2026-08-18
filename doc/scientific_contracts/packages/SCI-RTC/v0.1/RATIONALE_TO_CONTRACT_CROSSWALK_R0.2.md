# SCI-RTC v0.1/r0.2 rationale-to-contract crosswalk

Status: explanatory routing; `src/common/` remains the sole normative core.

| Rationale section | Scientific question | Primary formal authority |
| --- | --- | --- |
| 1. Executive summary and present status | What is RTC and what is presently claimed? | DEF-018, DEF-020; REQ-048--054, REQ-070 |
| 2. Learn–resolve–apply | What is learned, frozen, and executed? | DEF-007, DEF-015--016, DEF-021--025; EQ-021, EQ-028; REQ-010, REQ-031--036, REQ-055--058; PRED-023--024, PRED-027--029 |
| 3. Temporal frequencies and spatial scales | Which astronomical modes meet a filter? | DEF-026; EQ-013--014, EQ-023; REQ-030, REQ-033, REQ-040, REQ-060, REQ-066; PRED-010, PRED-020, PRED-031 |
| 4. Despiking and donors | What is rejected, transferred, and eligible? | DEF-005--006, DEF-011--013; EQ-001--002, EQ-007, EQ-020a--b; REQ-014--020, REQ-043, REQ-064--065; PRED-003--006, PRED-018, PRED-021, PRED-034--035 |
| 5. Notch filtering | What line is removed and what response remains? | DEF-008, DEF-025--026; EQ-009, EQ-024; REQ-021, REQ-023, REQ-040, REQ-058--061, REQ-069; PRED-011, PRED-015, PRED-030 |
| 6. Low/high/band-pass | Which temporal and spatial bands remain? | DEF-026; EQ-008, EQ-013; REQ-021--022, REQ-039--040, REQ-059--062, REQ-069; PRED-007--010, PRED-033 |
| 7. Order, phase, registration | Why these taps and where is the response centered? | EQ-008, EQ-013, EQ-025--027; REQ-022, REQ-040--041, REQ-063, REQ-067, REQ-069; PRED-009--010, PRED-032, PRED-036 |
| 8. Decimation and aliases | Why this factor and prefilter? | DEF-009, DEF-015--016; EQ-011, EQ-014--015, EQ-021, EQ-023; REQ-028--036, REQ-066--067; PRED-019--020, PRED-023--024, PRED-031, PRED-036 |
| 9. Masks, edges, state, non-finites | What happens at excluded or incomplete support? | DEF-014; EQ-009--010, EQ-015; REQ-024--027, REQ-041, REQ-046--047; PRED-012--017, PRED-025 |
| 10. Calibration and Beammap response | What commutes and when may calibration transfer? | DEF-002, DEF-004--005; EQ-001--004; REQ-002--005, REQ-013--016, REQ-065, REQ-068; PRED-002--005, PRED-035, PRED-037 |
| 11. Covariance, support, eligibility | What information and uncertainty reach each output? | DEF-010--013, DEF-017--018; EQ-012, EQ-015--020b, EQ-022; REQ-018--020, REQ-037--052; PRED-006, PRED-018, PRED-021--022 |
| 12. Validation and owner decisions | What evidence can falsify or qualify each choice? | DEF-019--020; ASM-005--012; REQ-051, REQ-054, REQ-057, REQ-069--070; all PRED IDs; OWNER-001--036 |

Every formal ID appears exactly once in the inventory tables of
`CROSSWALK.md`. This file maps the explanatory science-team narrative to that
inventory without adding normative science.
