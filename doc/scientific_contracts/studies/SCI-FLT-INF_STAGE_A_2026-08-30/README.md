# SCI-FLT-INF recovery and family-split study

Study identity: `SCI-FLT-INF-STAGE-A-2026-08-30`

Status: recovery-first Stage A holding study; not an approved package,
combined contract, author packet, Stage B launch, scientific authority, or
implementation-conformity finding

## Program adherence and prior-work recovery

This study follows the
[Scientific Contract Library Program](../../README.md), the
[pilot process review](../../PILOT_PROCESS_REVIEW_2026-08-16.md), the
[downstream roadmap](../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md), and
the [prior-work discovery registry](../../PRIOR_WORK_REGISTRY.md). It starts
from exact SCI-FLT Stage A commit
`cd55752e716051383da54356833ef0fac20b083a` and treats frozen SCI-NOI
authority commit `f28d7a2617160febca85c1c40e6f7ba7494e266e` as read-only exact
object access.

Recovery precedes new derivation. [`PRIOR_WORK.md`](PRIOR_WORK.md) records
what was found and classifies it as adopt, cite, abstract, supersede, defer,
exclude, or unavailable. Implementation, configuration, schemas, audit,
repair, validation, and production history remain quarantined in
[`IMPLEMENTATION_INFORMED_DOSSIER.md`](IMPLEMENTATION_INFORMED_DOSSIER.md)
and cannot enter any future implementation-blind author packet.

The active SCI-FLT-FIXED Stage B task, worktree, branch, drafts, and outputs
were outside this study's authority and were not used. The exact 17 approved
SCI-FLT-FIXED Stage A author objects and their manifest are immutable inputs
to the byte-preservation check only; this study changes none of them.

## Purpose

`SCI-FLT-INF` was retained by the SCI-FLT owner scope repair only as a
non-authoritative holding tranche. This study determines whether the candidate
material actually represents one future package or several scientific
identities. It inventories and separates:

- noise-weighted template-amplitude estimation;
- any genuine Wiener or posterior reconstruction method;
- matched or generalized least-squares source-amplitude estimation;
- learned-then-frozen and per-member-relearned state lifecycles;
- data-thresholded spectral mode selection;
- automatic method selection or fallback;
- input-derived edge/background conditioning;
- empirical NOI-based coefficient rescaling and standardized products; and
- source-learned or source-conditioned variants.

The study also records parent, order, response, covariance, support, validity,
failure, and NOI parity consequences. ODQ-001 now selects the historical
path's scientific estimand; the study does not yet select a package name,
rename an existing product, authorize a numerical route, or modify an
algorithm.

## Stage A packet

| Object | Role |
| --- | --- |
| [`SCOPE_BRIEF.md`](SCOPE_BRIEF.md) | sanitized purpose, bounds, and success criteria |
| [`PRIOR_WORK.md`](PRIOR_WORK.md) | recovery record and disposition |
| [`IMPLEMENTATION_INFORMED_DOSSIER.md`](IMPLEMENTATION_INFORMED_DOSSIER.md) | quarantined implementation/config/product inventory |
| [`FAMILY_SPLIT_MATRIX.md`](FAMILY_SPLIT_MATRIX.md) | candidate method/package separation |
| [`OPERATOR_STATE_PRODUCT_TAXONOMY.md`](OPERATOR_STATE_PRODUCT_TAXONOMY.md) | estimand, operator, state, response, product, and lifecycle vocabulary |
| [`CROSS_PACKAGE_AND_NOI_BOUNDARIES.md`](CROSS_PACKAGE_AND_NOI_BOUNDARIES.md) | MAP/JINC/NOI/SRC/VAL/FRUIT boundaries and parity cases |
| [`CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md`](CONTRADICTIONS_AMBIGUITIES_UNAVAILABLE_STATES.md) | explicit unresolved and unavailable states |
| [`PROPOSED_SANITIZED_AUTHOR_INPUTS.md`](PROPOSED_SANITIZED_AUTHOR_INPUTS.md) | material that could be sanitized after owner decisions |
| [`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md) | ordered consequential owner questions |
| [`SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md`](SCIENTIFIC_OWNER_ODQ_001_APPROVAL_2026-08-30.md) | exact matched-template estimand approval |
| [`FROZEN_AUTHORITY_AND_SOURCE_BINDING.md`](FROZEN_AUTHORITY_AND_SOURCE_BINDING.md) | exact base, frozen NOI, historical, and evidence bindings |
| [`STAGE_A_SOURCE_MANIFEST.md`](STAGE_A_SOURCE_MANIFEST.md) | content-bound Stage A study objects; not an author manifest |
| [`verify_stage_a.py`](verify_stage_a.py) | study and protected-byte verifier |

## Principal recovery result

The evidence does not support one undifferentiated inference-filter package.
The scientific owner has now closed `SCI-FLT-INF-ODQ-001`: the historical full
path is an **optimal matched-template amplitude estimator**. It uses the exact
supplied kernel as the expected template response and the declared noise model
to estimate template amplitude versus map position. A point-source-response
kernel yields a matched point-source amplitude field; another scientifically
defined kernel yields the amplitude field of that specified template. The
normalization must be unbiased for a matching signal under the declared
method assumptions.

The method is not a posterior/Wiener sky reconstruction, and ordinary source-
shaped convolution remains a distinct deterministic operation. A genuine
posterior reconstruction would be a separate future package. Adaptive edge
conditioning, data-thresholded mode selection, automatic fallback, empirical
coefficient calibration, and member-specific relearning also remain distinct
scientific identities or lifecycle policies.

The next owner question is package ownership and split under
`SCI-FLT-INF-ODQ-002`. No package naming or Stage B scope is approved.

## Nonclaims and stop rule

This study makes no implementation, representation-fidelity, conformity,
validation, calibration, achieved response/covariance, uncertainty,
significance, performance, readiness, production, Unity, or scientific-freeze
claim. It creates no default and authorizes no method substitution.

Do not create an implementation-blind author packet until the scientific owner
has resolved `SCI-FLT-INF-ODQ-002` and the later package-specific operator,
state, response, covariance, product, and lifecycle gates identified in the
decision ledger. Create separate package-local Stage A records rather than
converting this holding directory into a combined package.
