# Scientific-owner review 1 response

Draft returned: **SCI-PTC-COEFFICIENT-UNIFORM v0.1-draft.1 / r0.3**  
Review round: **1 of at most 3 substantive scientific-owner rounds**  
Status: **all requested corrections addressed as revised open proposals; complete contract approval and freeze remain withheld**

The governing feedback is the byte-exact local
[owner review](owner-review-1-input/OWNER_REVIEW_1_VERBATIM.md). The only
new provenance recovery is the admitted
[PTC rank and publication clauses](owner-review-1-input/PTC_RANK_AND_PUBLICATION_CLAUSES.md).
No implementation or independent-review evidence was used.

| Stable ID | Owner issue | r0.3 disposition | Canonical source and check |
| --- | --- | --- | --- |
| OR1-01 | Producer payload invalidity versus named-consumer denial versus unavailable/conflicting permission | **Addressed.** Only a UQ-05 fidelity false makes the payload producer-invalid. A complete permission record that denies or does not authorize consumer c makes only that named use ineligible. Missing/unresolved permission is U; contradictory permission is C. Payload fidelity and the other consumer remain unchanged. | definitions.tex, Handoff decisions and UQ-06; assumptions.tex, Failure-scope table; UFC-REQ-010, 013, 017; UFC-MIX-005--006; UFC-PRED-013. |
| OR1-02 | Independent scalar/unit/integrity/membership facts and mixed evidence | **Addressed.** UQ-05 now composes four independently T/F/U/C facts. Any independent F wins; otherwise same-fact C remains C; otherwise U remains U; all T is T. A contradiction within one fact is not also treated as a selectable F. Worked cases cover wrong scalar plus unknown membership, membership false plus unknown integrity, and same-fact conflict with and without an independent false. | definitions.tex, Independent payload-fidelity facts; UFC-REQ-014; UFC-MIX-001--004; UFC-PRED-007--008. |
| OR1-03 | Named-profile request versus selected-family predicate; shared-object versus malformed-request scope | **Addressed.** The explicit named-profile request sets R independently. After a selection-record identity is bound, a wrong selected value reaches UQ-01 and is false, missing value is unknown, and same-value conflict is conflict. A wrong q,g pair in one request remains local; a corrupt/misbound shared authoritative q-to-g object affects all references. | definitions.tex, Profile identity/request/applicability; assumptions.tex, Failure-scope table; UFC-REQ-011, 015, 017; UFC-MIX-007--008; UFC-PRED-006, 009, 014. |
| OR1-04 | Economical exact representation, shared evaluations, incremental publication, and analytic normalization | **Addressed as a revised proposal.** One immutable publication generation q is explicitly tagged as a complete segment, scan, or bounded within-scan/segment chunk. Each unit becomes handoff-eligible independently after its required components and links are durable. Authoritative target R_q is structurally distinct from stored compact reference R-hat_g, which is independently fidelity-checked; enumeration remains exact-once checked. Shared immutable facts may be referenced by logical occurrence decisions. Mean-one is analytic and requires no literal reduction or array of stored ones. | equations.tex, Exact support, Compact exact association, and Representation/publication conformance; UFC-REQ-002--004, 008--009; UFC-MIX-009--010; UFC-PRED-004, 009. |
| OR1-05 | Authority for positive-rank restriction | **Addressed by exact admitted provenance.** The ordinary-route paragraph explicitly cites frozen SCI-PTC v0.1/r0.5 requirements 090, 094, and 095; publication completion cites 072. Rank zero, noninteger rank, and unrealizable positive rank fail the admitted first ordinary route and do not become PTC-disabled or another route. The text infers no identity or zero-rank route family. | assumptions.tex, Ordinary-route and rank provenance; equations.tex, Exact support; UFC-REQ-008, 023; UFC-PRED-016; local admitted clause file. |
| OR1-S1 | Family-rule version versus realization generation | **Addressed.** Rule changes create the corresponding family/profile/permission/boundary successor. New observations, publication units, supports, payloads, requests, and evaluations under unchanged rules create immutable realizations rather than family versions. | assumptions.tex, Family rules, realizations, and immutable change; UFC-REQ-024; UFC-PRED-004--005. |
| OR1-S2 | Noncircular UQ-07 staging | **Addressed.** Construction snapshots pre-existing inputs; UQ-07 evaluates those lineage and nonmutation facts before sealing; finalization then records Z. Failure to finalize produces failed/incomplete artifact state and invents no completed UQ-07 result. | assumptions.tex, Noncircular UQ-07 construction and finalization; UQ-07; UFC-REQ-015, 024. |
| OR1-S3 | Collision between PTC product p and spatial p | **Addressed.** a is application generation, q is PTC product/publication generation, g is coefficient realization, i is detector-time occurrence, and p is solely the consumer spatial/map index. | equations.tex, Distinct identities and indices; UFC-REQ-002, 008, 021. |
| OR1-A1 | Distinct jobs for the two rendered views | **Addressed.** The science view foregrounds meaning, support, uncertainty, boundaries, consequences, and owner choices. The engineering view carries the full Registry/profile rules, mixed cases, scopes, representation, lifecycle, requirements, predictions, and evidence mapping. Both are composed transparently from the same six canonical modules through one view flag. | Both entrypoints, common_core.tex, and all six files under src/common/; the requirement crosswalk verifies every engineering requirement has a scientist-facing explanation. |

The owner’s in-principle support for fixed unity, no tunable parameter/default,
separate signal validity and uncertainty, separate consumer permissions, and
the retained NOI limit is recorded in the owner ledger as review evidence.
All fourteen consequential proposals remain open. This response does not
treat support in principle as complete adoption.

No conflict among the admitted rank/publication clauses and the revised
compact incremental proposal was found: atomic completion applies to each
declared required publication unit and does not impose an observation-wide
barrier. No mathematical definition is blocked. UFC-Q-001 remains:

> Is the actual scientific owner Grant Wilson acting for SCI-PTC, as proposed,
> or another named authority?

That question blocks active Registry/profile binding only.

