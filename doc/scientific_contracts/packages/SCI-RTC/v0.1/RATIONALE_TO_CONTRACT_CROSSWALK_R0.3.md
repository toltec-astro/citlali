# SCI-RTC v0.1/r0.3 bounded-refinement crosswalk

Status: explanatory routing; the six files under `src/common/` remain the sole
normative authority.

| Rationale locus | r0.3 scientific content | Formal authority |
| --- | --- | --- |
| §2 lifecycle definition | Finite outer cycles; immutable apply; online-adaptive exclusion | DEF-024--025, DEF-027; EQ-028--029; REQ-071--072 |
| §2 cumulative plans | Complete successor plan and one-way cycle state | DEF-028; REQ-073, REQ-081 |
| §2 parent rule | Default original-input replay; explicit cascade alternative | EQ-029; REQ-074--075; PRED-040/042 |
| §5 intended consequences | Residual line, predicted/measured PSD, response, source, covariance, support | REQ-076; PRED-039--040 |
| §5 successor candidate | Hidden-line discovery and artifact discrimination | REQ-077; PRED-041 |
| §5 cumulative admission | Interference improvement and cumulative scientific budgets | REQ-078; PRED-043 |
| §5 finite stopping | Complete-plan stability, maximum-cycle, oscillation and nonconvergence | DEF-029; REQ-079--080; PRED-044--045 |
| §§2/5 provenance | Original input, learning parents, plans, diagnostics, dispositions, stop and final identity | REQ-081 |
| §§2/12 restart | Final selected plan and parent-rule reproducibility | REQ-082; PRED-046 |

All pre-r0.3 authority remains routed by
`RATIONALE_TO_CONTRACT_CROSSWALK_R0.2.md` and inventoried with the appended IDs
in `CROSSWALK.md`.
