# Prioritized Repair Backlog

Status: **proposed bounded backlog; no item is authorized by this packet**

This is sequencing advice for owner review.  Scientific authority work,
implementation work, tests, configuration, and application validation must be
separate reviewed units.  No item may edit frozen package science silently.

| Priority | Proposed unit | Blocks | Required outcome | Authority / review gate |
| --- | --- | --- | --- | --- |
| P0 | Stop NOI coefficient promotion | MSP-E031, MSP-T014 | Make all NOI products immutable consumers/companions that cannot mutate PTC/MAP coefficient, support, exposure or validity state; retain legacy diagnostic only under explicit non-authoritative identity if owner permits | Scientific owner confirms compatibility disposition; independent source review and negative mutation tests |
| P0 | Replace MAP original-exposure path | MSP-P003, MSP-P004, MSP-E003, MSP-T001/MSP-T004 | Introduce exact stable-original coordinate/occurrence ledger and unique-original exposure placement distinct from processed signal membership | Exact AST/MAP boundary conformance review before code; focused deduplication tests |
| P0 | Implement equal-observation coadd | MSP-P005, MSP-E004, MSP-T004 | Use observation-row dimensionless `u_op=1`, preserve pixel/sample roles without flattening, atomically admit full bundles, union original occurrences | SCI-MAP r0.7.1 mapping review, exact arithmetic tests, no-mutation rejection tests |
| P0 | Replace JINC predecessor product | MSP-P006, MSP-E005/MSP-E020, MSP-T002/MSP-T012/MSP-T016 | Publish exactly N,C,Q,m with local validity, and C²-time; move diagnostics to non-product information; preserve typed unavailable states and per-array atomicity | Owner-selected JINC coefficient authority first; exact five-role review and schema gate |
| P1 | Close PTC consumer coefficient authority | MSP-E001, MSP-E005, MSP-E017 | Owner selects MAP-, JINC-, and NOI-design-balance-facing families/profiles and missing/QC behavior without aliases or numeric inference | Scientific-owner decision and immutable Registry/source binding |
| P1 | Implement frozen FLT-FIXED product | MSP-P007, MSP-E009--MSP-E011, MSP-E021 | Immutable parent, exact full-footprint fixed operator, response/covariance state propagation, identical NOI-member action, typed unavailable outputs | Separate scientific-to-implementation mapping review; deterministic operator fixtures |
| P1 | Implement immutable template and FLT-MATCHED products | MSP-P008, MSP-P009, MSP-E012--MSP-E014, MSP-E022 | Exact template authority and one selected A/C estimator route; no Wiener-name alias, fallback, detection, or hidden adaptation | Route-specific owner decision where still unavailable; source review and estimator truth tests |
| P1 | Close POINT route/policy/VAL authority | MSP-P013--MSP-P016, MSP-E023--MSP-E029 | Bind each parent family, compatibility/formal-error method, per-array atom, named-use policy and four-axis VAL evaluation | Register complete owner-bound profiles; independent scientific review before implementation |
| P2 | Preserve negative route boundaries in types | MSP-E007/MSP-E008/MSP-E015/MSP-E016/MSP-E032 | Make prohibited MAP/JINC, FIXED/MATCHED, and POINT/detection routes unrepresentable or reject atomically | Conformance review plus compile-time/negative tests |
| P2 | Separate legacy product names and outputs | all legacy rows | Make predecessor identifiers explicit, prevent names/headers from implying frozen package identity, and document replay-only compatibility | Output-schema migration proposal and owner-approved compatibility window |
| P2 | Build current frozen-oracle validation suite | MSP-T001--MSP-T016 | Implement the planning matrix after P0/P1 prerequisites; retain exact input/output/source/profile digests | Independent test review; application/Unity gates only under later work order |

## Recommended sequencing

Authority closure precedes affected implementation.  Within implementation,
the proposed safe order is: coefficient/negative-mutation boundary; original
exposure; MAP observation bundle; equal-observation coadd; five-role JINC;
fixed filter; matched template/filter; NOI identities; POINT/VAL; then the
complete representative-trace suite.  Do not combine mapmaking arithmetic,
filter mathematics, and owner-policy registration in one change.

## Explicit non-backlog items

The forbidden graph edges are not features to implement.  Active FRUIT and
OOF work are not backlog items here.  Unity reproduction, performance tuning,
and production activation are also outside this packet.
