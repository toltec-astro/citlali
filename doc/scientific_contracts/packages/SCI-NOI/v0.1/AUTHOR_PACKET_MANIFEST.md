# SCI-NOI v0.1 — Proposed Implementation-Blind Author Packet

Manifest identity: `SCI-NOI_AUTHOR_PACKET_MANIFEST v0.1/r0.3`

Status: complete, exclusive, and content-bound closure candidate; not approved;
Stage B not launched

The future implementation-blind author may open this manifest and only the 17
exact objects below. No adjacent full package is admitted.

| Item | Exact source and identity | Content SHA-256 | Authority/compatibility state |
| --- | --- | --- | --- |
| 1 | `SCOPE_BRIEF.md` | `a4c22e2f83dbe5f48b07962d0553b9947b86cb349064e273e27fb5e56943fa55` | Closure candidate; ODQ-101/102A incorporated; selected route numerically unavailable |
| 2 | `AUTHOR_SUPERSESSION_COVER.md` | `4efd4c471ab2c6a83e1c3f99bd02503209aac1b0a0d4e4de295cee6b6087bdc4` | Binding cover; proposed |
| 3 | `5a027c94ef9fc9c4a6e6cadc84af1c8a550d3508:doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE_R3.tex` | `27263ab3bf29ac8f098463455e540f13e783241a688ef2bc5cb15b1f2a4319da` | Recovered core; usable only under item 2 |
| 4 | `4f1fec36f7802f3b5e8ac067377679946930983c:doc/audits/packages/SCI-NOI-002_INDEPENDENT_CORE.tex` | `36781b766a2f57c9a3bd7e173ee8f1d85cba7f3d08afe2e67a403166f6b6d72d` | Recovered core; usable only under item 2 |
| 5 | `AUTHOR_CONVENTIONS_AND_OWNERSHIP.md` | `2fd5d5eb8435ff8c89d7091d284d10957092977813aaeda1938791d62b17c262` | Proposed sanitized ownership extract; inline MAP application bounded |
| 6 | `AUTHOR_OPERATOR_AND_PRODUCT_TAXONOMY.md` | `d0622531832353d8341bcc141cabd84b40e54b14e2897c12f7c6e68db2ffc696` | Proposed collision-free taxonomy; ordinary route selected |
| 7 | `SCI-MAP_TO_SCI-NOI_BOUNDARY.md`, `SCI-MAP_TO_SCI-NOI v0.1/r0.2` | `42e4449cef69f8f362ccb064c48ec6bb30fcee94a9f285eda4a96e1a25a2458d` | Exact frozen-source extract; realized-MAP route unselected/unavailable |
| 8 | `SCI-JINC_TO_SCI-NOI_BOUNDARY.md`, `SCI-JINC_TO_SCI-NOI v0.1/r0.2` | `3bdf5f620940eae213dfd82edd8e421e343b4e64b4e721d3fdef2530f857cd99` | Exact frozen-source extract; JINC routes unselected/unavailable |
| 9 | `SCI-PTC_TO_SCI-NOI-GEN_BOUNDARY.md`, `SCI-PTC_TO_SCI-NOI-GEN v0.1/r0.2` | `128107ef070293188135012dc079ab61fcf0dc293c3271a839804880c05f610c` | Exact frozen-source extract; MAP host selected but numerically unavailable; JINC host unselected |
| 10 | `NOI_GEN_PARENT_OPERATOR_GRAPH.md`, `SCI-NOI_GEN_PARENT_OPERATOR_GRAPH v0.1/r0.2` | `4d532f2fc966f96db87aaafd0989dfab5ce9fe1d7fc4633188c9029bddeefef0` | ODQ-101/102A approved; selected route conditional/unavailable |
| 11 | `ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT_SPECIFICATION.md`, `SCI-NOI_ENSEMBLE_DESIGN_AND_SOURCE_IMPRINT v0.1/r0.1` | `55acdd80386286e7471d9add79b705afa3d5815bf81f792af57d9cafaf099786` | Complete candidate identity; ODQ-102B/C and 103 open |
| 12 | `FINITE_DESIGN_UNC_ESTIMATOR_AND_COVARIANCE_TABLE.md`, `SCI-NOI_FINITE_DESIGN_UNC_TABLE v0.1/r0.1` | `c349394d3de36ab8d3044bd12ddb3ee9ea9f73dc6c3d16e8b6b207f53775360d` | Candidate decision surface; numerical UNC unavailable |
| 13 | `STD_NUMERATOR_SCALE_AND_CLAIM_TABLE.md`, `SCI-NOI_STD_NUMERATOR_SCALE_CLAIM v0.1/r0.1` | `960e46273ca1653272bd6078bc5715f880909a3978357842a024cf90a0593a98` | Candidate decision surface; numerical STD unavailable |
| 14 | `SCI-NOI_VAL_PROFILE_DRAFTS.md` | `34fb589d1691a43460445e47a600e0be4c4880320e71711df80f0331150463ee` | NOI-owned drafts; selected-route action proposed; unapproved, unregistered, unevaluable |
| 15 | `FILTER_AND_FRUIT_SCOPE.md` | `de02815792bf9bdcf19651abd8163659c79c8ad6c3cbbfaf41210b17973e9f97` | Proposed scope; all filtered routes unavailable |
| 16 | `PRODUCT_ROLE_AND_LIFECYCLE_TABLE.md`, `SCI-NOI_PRODUCT_ROLE_AND_LIFECYCLE v0.1/r0.2` | `3c822403fc038302fc17e59695e4bae81145618f7d6d18d652fd59eb4bc79961` | Proposed atomic roles; inline modifier and NOI realization-map ownership explicit |
| 17 | `SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`, `SCI-NOI_OWNER_DECISIONS v0.1/r0.3` | `d288d177a31f65aa466589713fb2da5fbc24c7cc60229b7cca72fcfa75b1bc90` | ODQ-101/102A/104 decided; ODQ-102B recommendation exact; dependent later methods unavailable while open |

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
the Scope Brief. The next walkthrough item is `SCI-NOI-ODQ-102B`.
