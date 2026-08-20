# SCI-RTC v0.1/r0.9 Rationale-to-Contract Crosswalk

Status: implementation-blind routing for binding owner Decisions 1--8. The
exact ID-level authority remains `CROSSWALK.md`.

| Owner clarification | Rationale locus | Shared normative authority | Engineering/falsification surface |
| --- | --- | --- | --- |
| Operations admitted across application contexts; availability is not selection or qualification | §§1--2, 10--11 | DEF-002/006; ASM-005; REQ-002/012/105 | REQ-006--012 and REQ-103--105 routing; PRED-002 |
| Distinct one-way application context, resolved plan, and realized record | §§1--2, 10 | DEF-007/023--024; REQ-010/032/035 | Lifecycle/restart checks; PRED-024/029 |
| One consumer-neutral atomic bundle | §§1, 11 | DEF-018; EQ-022; REQ-003/048/105 | Atomic-output and multi-context schema inspection; PRED-002/063 |
| Preserve upstream mapping, exact pair identity, and independent $x/r$ validity | §§1, 4 | DEF-001/030--031; REQ-001/083--086 | Missing/reordered pair and independent-validity fixtures; PRED-047--049 |
| Coordinate operations require explicit authority; no inference or clamping | §§4, 10 | DEF-014; ASM-003; REQ-009/024--025 | Invalid-coordinate matrix; PRED-012--013 |
| Non-finite state is typed and cause-preserving | §10 | REQ-026/046--047 | Cause-specific non-finite injection; PRED-017 |
| Every covariance/uncertainty claim discloses included and excluded components and correlations | §12 | DEF-017; EQ-016--019; ASM-006--007; REQ-042--045/093 | Conditional/component-limited/total claim audit; PRED-021/054 |
| Selected despiking modifies data; normal output uses compact population summaries and optional inert detail | §5 | DEF-006/018; ASM-010; REQ-017--020/050--051 | Accepted-target modification, summary, and detail-toggle checks; PRED-026 |

R0.9 does not alter the Decision 9 routing in
`RATIONALE_TO_CONTRACT_CROSSWALK_R0.8.md`.
