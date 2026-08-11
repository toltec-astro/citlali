# SCI-CAL-001 successor-4 bounded technical repair handoff

Date: 2026-08-11

Handoff ID: `SCI-CAL-001-SUCCESSOR-4-REPAIR-DISPATCH-001`

Status: frozen owner-authorized scope; role-separated repair not launched

## Exact authority and proposed base

- Owner acceptance:
  [`SCI-CAL-001_SUCCESSOR_3_OWNER_ACCEPTANCE_2026-08-11.md`](SCI-CAL-001_SUCCESSOR_3_OWNER_ACCEPTANCE_2026-08-11.md),
  SHA-256
  `73bd190349dd2ccd3405baa4cab294deb4e87c576367a54dbc44b3014a52a9e1`.
- Immutable re-audit report:
  [`../SCI-CAL-001_SUCCESSOR_3_REAUDIT_2026-08-11.md`](../SCI-CAL-001_SUCCESSOR_3_REAUDIT_2026-08-11.md),
  SHA-256
  `ee0f8c40e31300fd5c547b45d086a5e97f7be52e45d453016c0ade28c014e59a`.
- Immutable local evidence:
  [`../evidence/SCI-CAL-001_SUCCESSOR_3_LOCAL_EVIDENCE_2026-08-11.yaml`](../evidence/SCI-CAL-001_SUCCESSOR_3_LOCAL_EVIDENCE_2026-08-11.yaml),
  SHA-256
  `7245872f044fc15a7cdb631ea02b70a9746cba1351bc942c1dde8bffa2b25a6f`.
- Machine-readable finding ledger:
  [`../proposals/SCI-CAL-001_SUCCESSOR_4_REPAIR_FINDING_LEDGER_2026-08-11.yaml`](../proposals/SCI-CAL-001_SUCCESSOR_4_REPAIR_FINDING_LEDGER_2026-08-11.yaml).

The exact pushed application base is
`3af6faf996fa002b2647adca8f33991002d49ff1` on
`origin/codex/repair-sci-cal-001-successor-3`:

- parent: `8b1534807f5abe4d80be2fbd45ed3838ed351509`;
- tree: `16130eb6deba3f9d8b5a8f1d1fae126084b63c95`; and
- parent-to-candidate binary-patch SHA-256:
  `4558d541a82b2a1f5c4406825c277a2f7317b6d4c788f4bbaf699a385d471bdf`.

The proposed repair branch is
`codex/repair-sci-cal-001-successor-4`. It was absent locally, in
remote-tracking state, and at live origin when this handoff was frozen. This
document does not create the branch or launch a task.

## Frozen scope

Successor-4 is a technical completion of only F005, F007, F008, and the local
implementation portion of F009. It must preserve F002, F003, F004, and F006
and leave F001 and F010 open and conditioned. The controlled axes remain
`approved`, `nonconformant`, `in_progress`, and `fail_closed`, with verdict
`amend`, until a later independent re-audit and owner disposition say
otherwise.

No new scientific decision, estimator, calibration arithmetic, weighting
scheme, variance definition, mapmaking behavior, response calculation,
covariance product, or scientific-validity claim is authorized.

## Mandatory READY checkpoint before edits

A later role-separated repair task must start at the exact pushed base above
in a fresh clean worktree. Before editing, it must return:

1. exact local and live-origin base, parent, tree, patch digest, branch, and
   clean worktree/index state;
2. confirmation that the successor-4 branch was absent before creation;
3. an exact proposed changed-path allowlist, separated by F005, F007/F008,
   local F009, regression preservation, and status handback;
4. finding-to-file and finding-to-test traceability;
5. the existing authority and consumer for every metadata, product-contract,
   profile, baseline, and fixture path proposed;
6. proof that F002/F003/F004/F006 behavior remains preserved and that F001 and
   F010 are not being implemented or reinterpreted;
7. confirmation that no new product, uncertainty, covariance, response
   estimator, unit, lineage system, RTC/PTC/MAP/BEAM/TEL/ALIGN change, or
   scientific choice is proposed; and
8. a first viable artifact and a second scope checkpoint before writer-wide
   or broad-test execution.

Silence prohibits a capability. Any scientific ambiguity, new persisted
authority, new production product, cross-package redesign, or path outside the
approved checkpoint requires an immediate stop and coordinator/owner review.

## F005 — exactly-once recipient proof

Complete the proof and truthful inventory for the active `validated`
weighting mode as well as `approximate`, `hybrid`, and `full`. The inventory
must state each existing production recipient, coefficient/factor role,
application stage, units, normalization, support, and exactly-once behavior.

Exercise the actual production `noise_variance_I` recipient with nonzero noise
realizations and prove that a valid calibration factor `a` gives
`V' = a^2 V`. Existing inverse-variance recipients retain `W' = W/a^2` under
the already approved F005 semantics. A scalar helper alone is insufficient
evidence for the production recipient.

Do not redesign weighting, variance, mapmaking, or calibration arithmetic and
do not create a new uncertainty or covariance product.

## F007 and F008 — complete applied identity and product joins

Make canonical calibration, response, and package identities
collision-resistant over the complete material state already authorized and
actually applied. At minimum, bind:

- the applied sample-elevation/extinction state or its deterministic full
  identity rather than extrema alone;
- active realized FIR `a_gibbs`;
- applied fixed-notch zero-phase state, center frequencies, and widths;
- applied dynamic line-audit notches;
- selected APT, factor, mapmaker/kernel, filtering, acquisition, and package
  state already required by the accepted CAL contract; and
- exact requested, effective, and realized distinctions.

Dormant disabled configuration must not be labelled realized. Identity
construction must be deterministic, collision-resistant over the declared
state, and stable across writer/readback joins.

Populate `calibration_identity` and `package_identity` through the actual map
FITS, TOD NetCDF, and Beammap ECSV production metadata routes. Focused tests
must reopen each product and resolve it uniquely to the canonical package and
response record. Manual test-only metadata injection is not production-route
evidence.

Preserve exact once-only calibration composition and fail-closed statements
for unavailable empirical response fidelity, total uncertainty, nuisance
covariance, donor-target covariance, and scientific precision/accuracy. Do
not create or imply those stronger claims.

## Local F009 — executable v4/package-copy synchronization

Synchronize executable baseline and audit consumers with
`citlali-raw-timestream-provenance-v4` and validate its canonical-lineage and
package semantics. Preserve v1-v3 compatibility unless an exact contradiction
is found; if so, stop before changing compatibility policy.

Classify `selected_calibration_apt.ecsv` as the required CAL package member in
the executable product contract and in every applicable mode profile and
baseline comparison policy. Add checked-in v4 and package-copy fixtures that
exercise:

- v4 admission and canonical lineage;
- actual production identity population;
- required member classification and comparison behavior;
- writer/reopen package and product joins;
- required-output failure propagation; and
- zero partial generation on failure.

This scope synchronizes the already implemented package surface. It does not
authorize a new package design or output class.

## Required preservation regressions

The successor must retain and rerun the accepted controls for:

- F002 fixed atmosphere-operator contract, nodes, generated header, endpoints,
  monotonicity, and seams without a model-fidelity claim;
- F003 unsupported-unit startup rejection and typed admission failure before
  analysis, mutation, or publication;
- F004 legacy and optional-modern APT lineage/association boundaries without
  inventing authority; and
- F006 approved `mJy/beam` configuration boundary only.

Any regression in those closures stops the repair. F001/F010, exact-SHA Unity,
astronomical standards, ALIGN/AST authority, empirical response fidelity, and
production precision/accuracy remain conditioned external work.

## Local validation and handback

Before handback, run only deterministic local implementation gates:

1. all focused F005, F007/F008, and local F009 fixtures above;
2. actual map FITS, TOD NetCDF, Beammap ECSV, canonical package, and selected
   APT writer/reopen tests;
3. F002/F003/F004/F006 preservation regressions;
4. focused tests and full CTest, recording every disabled or skipped test;
5. `$HOME/tolteca/bin/python tools/config/run_config_preflight.py --require-all`;
6. all applicable authority, baseline, product-contract, mode-profile, and
   interface-synchronization gates; and
7. `git diff --check`, exact changed-path inventory, and clean final state.

Return the exact repair commit, parent, tree, changed paths, patch and artifact
digests, commands/results/skips, finding traceability, and remaining
dependencies. Stop for coordinator review. Do not start a re-audit, prepare or
request Unity work, merge, or push.

## Explicit exclusions and stop rule

No CAL scientific expansion, RTC/PTC/MAP/BEAM/TEL/ALIGN change, production
authorization, downstream launch, Unity access or request, local science
reduction, external contact, merge, or push is authorized. The repair may be
launched only by a separate owner-approved, role-separated task citing this
frozen handoff and its exact post-commit digest.
