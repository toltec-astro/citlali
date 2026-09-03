# Role-specific validity-domain crosswalk — r0.5

Status: normative closure amendment

| Role | Domain | Relation | Optional-role effect on signal |
|---|---|---|---|
| filtered amplitude signal | 'V_signal' | primary producer domain | n/a |
| response role r | 'V_response(r)' | subset of 'V_signal' | unavailable response does not narrow signal unless named use requires it |
| covariance role r | 'V_covariance(r)' | subset of 'V_signal' | unavailable covariance does not narrow signal unless named use requires it |
| NOI child/companion | 'V_NOI' | subset of 'V_signal' | not requested or unavailable does not narrow signal |
| FLT-to-FRUIT handoff | 'V_FRUIT_handoff' | subset of 'V_signal' or bundle predicate | not requested or unavailable does not narrow signal |
| state/lineage companion | role-specific | no broader than 'V_signal' | independently atomic |

PA, SA, SP, CU, NU, RU, and FH retain independent request, applicability,
eligibility, realization, validity, lifecycle, action, and provenance records.
No aggregate verdict may conceal a failed or unavailable required role.
