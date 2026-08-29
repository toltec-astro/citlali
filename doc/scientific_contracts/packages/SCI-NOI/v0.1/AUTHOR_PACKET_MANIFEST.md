# SCI-NOI v0.1 — Proposed Implementation-Blind Author Packet

Manifest identity: `SCI-NOI_AUTHOR_PACKET_MANIFEST v0.1/r0.2`

Status: complete, exclusive, and content-bound closure candidate; not approved;
Stage B not launched

The future implementation-blind author may open this manifest and only the 17
exact objects below. No adjacent full package is admitted.

| Item | Exact source and identity | Content SHA-256 | Authority/compatibility state |
| --- | --- | --- | --- |
| 1 | `SCOPE_BRIEF.md` | `6c93bd5c4e502bc00dedb56b39b4f13e94d4735feead51258fffcb4eec8810a8` | Closure candidate; ODQ-101 incorporated; later choices open/unavailable |
| 2 | `AUTHOR_SUPERSESSION_COVER.md` | `f631467ce57442c600f2907d00c8525ba98525fed6b852aacf80ded57a6ce189` | Binding cover; proposed |
| 3 | `5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` | Recovered core; usable only under item 2 |
| 4 | `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` | Recovered core; usable only under item 2 |
| 5 | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `14e9ca1bbd9827a9c19651be6dcaf0e35b5dda65ef5a583bdb6006a7b486544a` | Proposed sanitized ownership extract |
| 6 | `AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md` | `ff2e85e6793c8958befc5fc79d97e552b51e392e17b2c34cf52c3b22c0ba2e88` | Proposed collision-free taxonomy |
| 7 | `SCI-MAP_TO_SCI-NOI_BOUNDARY.md`, `SCI-MAP_TO_SCI-NOI v0.1/r0.1` | `419d4bbed7d2a13d9f983845280ba41b2cae01957196bf83478f9aced31e1cf1` | Exact frozen-source extract; numerical route unavailable |
| 8 | `SCI-JINC_TO_SCI-NOI_BOUNDARY.md`, `SCI-JINC_TO_SCI-NOI v0.1/r0.1` | `a6b34323ae9c09ea06d6530f59164d5a8080ac20113a2f7d40725ad88687fc64` | Exact frozen-source extract; numerical route unavailable |
| 9 | `SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md`, `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.1` | `55cc555efa0a0eb645224da17fe6d526dc7295e7d7626535af5f0073131817bb` | Exact frozen-source extract; both host routes unavailable |
| 10 | `NOI_GEN_PARENT_OPERATOR_GRAPH.md`, `SCI-NOI_GEN_PARENT_OPERATOR_GRAPH v0.1/r0.1` | `d333328d64d096bd8bb944d4a82a412357fee2a8087b124b22e3bdcd9e61c3fe` | ODQ-101 approved; route candidates proposed/unavailable |
| 11 | `ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md`, `SCI-NOI_ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT v0.1/r0.1` | `55acdd80386286e7471d9add79b705afa3d5815bf81f792af57d9cafaf099786` | Complete candidate identity; ODQ-102B/C and 103 open |
| 12 | `FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md`, `SCI-NOI_FINITE_DESIGN_UNC_TABLE v0.1/r0.1` | `c349394d3de36ab8d3044bd12ddb3ee9ea9f73dc6c3d16e8b6b207f53775360d` | Candidate decision surface; numerical UNC unavailable |
| 13 | `STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md`, `SCI-NOI_STD_NUMERATOR_SCALE_CLAIM v0.1/r0.1` | `960e46273ca1653272bd6078bc5715f880909a3978357842a024cf90a0593a98` | Candidate decision surface; numerical STD unavailable |
| 14 | `SCI-NOI_VAL_PROFILE_DRAFTS.md` | `25b48d3e172759b18f98767cf0af736d62ea96d44b82e913687c6f51dbbb69ed` | NOI-owned drafts; unapproved, unregistered, unevaluable |
| 15 | `FILTER_AND_FRUIT_SCOPE.md` | `de02815792bf9bdcf19651abd8163659c79c8ad6c3cbbfaf41210b17973e9f97` | Proposed scope; all filtered routes unavailable |
| 16 | `PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md`, `SCI-NOI_PRODUCT_ROLE_AND_LIFECYCLE v0.1/r0.1` | `9bc64f65f06e0945be63f516298d8bad0a007fb1dc214cecf20110921f83ed48` | Proposed atomic roles; GEN completion ownership explicit |
| 17 | `SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`, `SCI-NOI_OWNER_DECISIONS v0.1/r0.2` | `6081857084055df9116fa78db4e407efaecd311445c00c3c1338815021933e4b` | ODQ-101/104 decided; every dependent later method stays unavailable while open |

Any byte change requires recomputed hashes and renewed exact-byte review. The
three boundary artifacts are exact sanitized extracts; they do not create a
numerical parent. Profile drafts do not become evaluable until owner-approved
bytes are bound by exact versioned SCI-VAL Registry/source successors.

## Complete Prohibited-Input Firewall

The future author must not open:

- `README.md`, `PRIOR_WORK.md`, `INTERNAL_DOSSIER.md`,
  `OWNERSHIP_AND_BOUNDARY_CLASSIFICATION.md`, `DECISION_LOG.md`,
  `SCIENTIFIC_OWNER_DECISION_LEDGER.md`, `STAGE_A_CHANGE_LOG.md`,
  `BYTE_EQUALITY_AND_SOURCE_CLOSURE_REPORT.md`, raw owner directions, or raw
  approval records;
- historical NOI owner/coordinator briefs, audits, findings, repairs,
  re-audits, integration records, cross-audit handoffs, and evidence results;
- current or historical Citlali implementation, source, headers, schemas,
  configuration, product contracts, tests, generated products, accepted runs,
  validation, reductions, Unity, achieved performance, defaults, or historical
  behavior;
- `doc/citlali_noise_estimation_plan.tex`, Convolve material, historical
  counts/defaults, or implementation vocabulary;
- the full frozen MAP, JINC, RTC, CAL, PTC, AST, VAL, BEAM, FLT, SRC/MODE,
  FRUIT, or other package; and
- any unlisted local file, repository, web source, external paper, or
  model-memory substitute.

If this exclusive packet is insufficient, the author returns one precise
scientific question. It may not inspect prohibited material or select a
scientific default.

## Author Controls And Dispatch State

The author must preserve collision-free roles, immutable parents, distinct
route identities, no mixed fixed/relearned ensemble, complete design and
source imprint, target-specific estimation, STD unit `1` and claim ceiling,
atomic lifecycle, and explicit unavailable states. No implementation,
calibration, physical-noise, Gaussian-significance, performance, readiness, or
production claim is authorized.

Stage B has not been launched. Conditional dispatch requires the exact gate in
the Scope Brief. The next walkthrough item is `SCI-NOI-ODQ-102A`.
