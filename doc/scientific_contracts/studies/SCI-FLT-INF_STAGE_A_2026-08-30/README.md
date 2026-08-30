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
failure, and NOI parity consequences. It does not select an estimand, rename
an existing product, authorize a numerical route, or modify an algorithm.

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
| [`FROZEN_AUTHORITY_AND_SOURCE_BINDING.md`](FROZEN_AUTHORITY_AND_SOURCE_BINDING.md) | exact base, frozen NOI, historical, and evidence bindings |
| [`STAGE_A_SOURCE_MANIFEST.md`](STAGE_A_SOURCE_MANIFEST.md) | content-bound Stage A study objects; not an author manifest |
| [`verify_stage_a.py`](verify_stage_a.py) | study and protected-byte verifier |

## Principal recovery result

The evidence does not support one undifferentiated inference-filter package.
The current full path is consistent with a spatially varying normalized
template-amplitude estimator, but that is an implementation-informed
inference rather than scientific authority. A genuine posterior/Wiener sky
reconstruction has a different estimand, prior, response, and covariance and
was not recovered as an independently specified active method. Adaptive edge
conditioning, data-thresholded mode selection, automatic fallback, empirical
coefficient calibration, and member-specific relearning are also distinct
scientific identities or lifecycle policies.

The first owner question is therefore the intended estimand of the existing
full path. No package naming or Stage B scope can be made coherent until that
question is answered.

## Nonclaims and stop rule

This study makes no implementation, representation-fidelity, conformity,
validation, calibration, achieved response/covariance, uncertainty,
significance, performance, readiness, production, Unity, or scientific-freeze
claim. It creates no default and authorizes no method substitution.

Do not create an implementation-blind author packet until the scientific owner
has resolved at least `SCI-FLT-INF-ODQ-001` through the package-split and
operator-identity gates identified in the decision ledger. If the owner
selects more than one estimand, create separate package-local Stage A records
rather than converting this holding directory into a combined package.
