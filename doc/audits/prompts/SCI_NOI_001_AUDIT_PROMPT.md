# SCI-NOI-001 conditional-noise-ensemble audit dispatch — 2026-08-05

This is a frozen, phased scientific-contract audit dispatch. It authorizes
only the Phase 0 scope checkpoint below. It does not authorize an audit,
implementation inspection, an auditor-created audit branch/worktree, a
reduction, Unity activity, repair, integration, a push, or another package
launch. The Codex app/coordinator may create the isolated Phase 0 execution
worktree stated below; that infrastructure action has no audit authority.

## Frozen assignment

- Package: `SCI-NOI-001` — Noise and jackknife realization construction and
  propagation.
- Tier: A.
- Canonical repository: `/Users/gwilson/GitHub/citlali-refactor`.
- Frozen application source to assess in later phases:
  `d5015fe716971bf8ea617e8a187311bf5af05185` on
  `origin/codex/refactor-mainline`.
- Phase 0 execution environment: the Codex app/coordinator may create a
  dedicated isolated, clean execution worktree at detached/starting HEAD
  `d5015fe716971bf8ea617e8a187311bf5af05185` from
  `origin/codex/refactor-mainline`. This is environment creation only.
- Required later audit branch: `codex/audit-sci-noi-001`, created only after
  explicit Phase 1 authorization, in that isolated task worktree if safely
  possible.
- Dispatch coordination parent:
  `192e0d9b5e3be4eb20522d3319cae346168c4bce`.
- Frozen inbound manifest:
  `doc/audits/handoffs/SCI-NOI-001/SCI-NOI-001_INBOX_MANIFEST_2026-08-05.yaml`
  (digest recorded in the canonical ledger).
- Pre-core authority handoffs: `SCI-NOI-001-XAUD-001` (MAP) and
  `SCI-NOI-001-XAUD-002` (FLT). There are no post-core evidence handoffs at
  dispatch.

## Quarantined implementation and exposure boundary

The following known package implementation paths are quarantined through the
Phase 1 independent-core freeze. Their enumeration is an exposure boundary,
not a claim of an exhaustive implementation inventory:

- `include/citlali/core/pipeline/timestream_scan_generation.h`
- `include/citlali/core/pipeline/noise_execution_plan.h`
- `include/citlali/core/mapmaking/naive_mm.h`
- `include/citlali/core/mapmaking/jinc_mm.h`
- `include/citlali/core/pipeline/observation_coadd_accumulation.h`

Package-specific tests/diffs and any other implementation source directly
exercising sign construction, RNG/seed, realization mapmaking/coaddition/
filtering/persistence, or fruit-loop subtraction/re-addition are likewise
quarantined until Phase 2. No post-core evidence may be opened in Phase 1.

## Phase profile and stop rule

| Phase | Model/effort | Authority now | Required stop |
| --- | --- | --- | --- |
| 0 | `gpt-5.6-terra`, high | scope checkpoint, repository state, and quarantine verification only in the coordinator-created isolated execution worktree | return before independent derivation or source inspection |
| 1 | `gpt-5.6-sol`, ultra | authorized only for the existing task `019fd40a-cf4d-7ad0-aa9b-9543d5236154` and the independent-core freeze | commit only `SCI-NOI-001_INDEPENDENT_CORE.tex`, report its digest/commit/timestamp, and stop before quarantined source or post-core evidence |
| 2 | `gpt-5.6-sol`, ultra | only after a separate explicit coordinator follow-up | documentation-only source/product/consumer trace and final audit synthesis, then stop |

The Phase 1/2 Ultra trigger is owner-approved and specific: reconcile the
distinct hard investigations of (a) sign-law mathematics and effective
ensemble size; (b) conditioning, RNG, state, eligibility, and provenance;
(c) propagation through observation/coadd/filter operators; (d) deterministic
signal leakage and the fruit-loop residual bias trade; and (e) ensemble
adequacy across consumer classes while preserving the NOI-002 boundary. The
single audit task is the synthesis owner. Freeze/hash/commit the independent
core first; then map every source/product/consumer result to numbered core
equations and state dependency-conditioned conclusions. Ultra ends at each
named Phase 1 or Phase 2 stop boundary. No delegation or parallel subagents
are permitted.

## Exact Phase 1 continuation prompt

Continue only task `019fd40a-cf4d-7ad0-aa9b-9543d5236154` in its verified
isolated execution worktree
`/Users/gwilson/.codex/worktrees/e71f/citlali-refactor`, which is clean at
detached HEAD `d5015fe716971bf8ea617e8a187311bf5af05185`. Use
`gpt-5.6-sol` at Ultra effort, serially, with no delegation or subagents.

Before derivation, reverify the clean worktree and exact starting SHA. Create
`codex/audit-sci-noi-001` in that worktree only if the branch is absent and the
worktree remains clean; do not move, reuse, or alter another branch. Read only
repository-level authorities and frozen pre-core handoffs
`SCI-NOI-001-XAUD-001` and `SCI-NOI-001-XAUD-002`. Do not open any quarantined
implementation/test/diff or post-core evidence.

Create and commit only
`doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE.tex`. Report its path,
SHA-256, commit, UTC timestamp, first permitted Phase 2 inspection boundary,
and clean state, then stop. Do not perform Phase 2, a local reduction, Unity
action, helper/schema/verifier work, repair, push, integration, or another
audit/package launch.

## Exact Phase 0 launch prompt

The Codex app/coordinator may create a dedicated isolated, clean execution
worktree whose detached/starting HEAD is exact
`d5015fe716971bf8ea617e8a187311bf5af05185` from
`origin/codex/refactor-mainline`. This Phase 0 infrastructure creation does
not authorize the auditor to create, switch, or move an audit branch; write
files; derive equations; inspect quarantined implementation; run a reduction;
or perform later-phase work.

In that execution worktree, read the repository `AGENTS.md`, TolTEC
context/authorities, this prompt, and the frozen manifest. Verify and report
the worktree path, detached/starting HEAD, clean state, and the dispatch
commit/prompt/manifest bytes via exact Git objects. Confirm that the listed
implementation paths and all post-core evidence remain quarantined. Return
only the `FRAMEWORK-SCOPE-001` checkpoint: allowed paths and deliverables;
prohibited local reduction; Unity requirement; named read-only checks;
permitted delegation/review; first viable artifact; next return point; and any
mismatch. Stop. Do not derive equations, inspect package implementation
contents, create/switch/move an audit branch, write files, run Citlali, open
Unity, request evidence, or make a scientific conclusion.

## Central scientific question for later authorized phases

Given a fixed realized post-RTC/PTC sample state, what conditional random
ensemble does Citlali's pre-mapmaking sign operation generate, which
physical-noise modes and deterministic-signal imprints does it preserve or
destroy, and for which downstream estimators could that ensemble serve as an
admissible input—without deciding whether those estimators are themselves
valid?

## Owner-approved facts and conditional object

The sign operation is immediately before mapmaking. RTC/PTC have already
operated and are frozen per realization. Realized residuals, correlations,
flags, and imperfections remain in the sample stream. The ensemble is
conditional on realized RTC/PTC state; it need not reproduce raw detector-noise
or RTC/PTC parameter-estimation uncertainty.

The ensemble must be a clearly defined, sufficiently faithful empirical
surrogate for explicitly authorized uses, not a statistically optimal
estimator. Observation membership, scan geometries, valid regions, and coadd
weights vary; derive conditional realized-coadd behavior, not a universal
scalar guarantee. Use **ensemble second-moment imprint**, not
“variance/weight signature,” except when tracing an observed downstream
consumer symptom and explicitly reserving estimator/product validity to
SCI-NOI-002. A distribution-valid, covariance-valid, or variance-scale-valid
label in this package means ensemble adequacy only, never validity of a
downstream estimator or product.

For an observation `o`, conditioning state `Theta_o`, realized sample vector
`x_o = s_o + n_o`, sign operator `D_epsilon`, and realized map operator
`A_o`, the Phase 1 core must define and number:

```text
x_o = s_o + n_o conditional on Theta_o
z_o^(r) = A_o D_epsilon x_o
z_o,res^(r) = A_o D_epsilon[x_o - shat_o]   (conceptual only)
```

The residual expression is a conceptual successor hypothesis only: randomize
the final fruit-loop source-subtracted sample state immediately before source
re-addition. It may reduce positive source leakage but may suppress genuine
noise if the data-derived source model absorbs noise. It is not an authorized
implementation, an assumption of unbiasedness, or a request to inspect/audit
the fruit-loop algorithm before the core freeze.

Deterministic astronomical signal may inflate ensemble second moments and
create a source-shaped downstream anti-weight symptom. This is a strong
falsification diagnostic; its absence is not proof of zero leakage. Count 64
is only a resource-admitted high-count validation tier, not a requirement,
production default, beammap expectation, or new cap.

## Frozen dependency and consumer boundaries

1. **MAP (`SCI-MAP-001`)** is `conditioned`, not open or satisfied. The
   accepted bounded MAP contract at `af0c849` and its application-mainline
   documentation child `d5015fe716971bf8ea617e8a187311bf5af05185` supply signal, nonprecision
   gridding/normalization `weight_I`, kernel, facts/support/validity, and
   centered coefficient-weighted observation coaddition. `weight_I` is not
   variance or precision authority. MAP remains conditioned on PTC/VAL and
   `existing_use_only`.
2. **PTC (`SCI-PTC-001`)** remains open. NOI-001 conditions upon the realized
   cleaned sample state, retained correlations, flags, detector weights, and
   source-mask/selection state at the sign boundary; it does not validate the
   PTC covariance, transfer, or parameter-estimation uncertainty.
3. **VAL (`SCI-VAL-001`)** remains open. NOI-001 conditions upon the realized
   sample/detector eligibility, non-finite, flag, and support state presented
   to mapmaking; it does not establish eligibility correctness or universal
   validity semantics.
4. **MAP-002/JINC (`SCI-MAP-002`)** remains separate. If an active JINC path
   is assessed in Phase 2, condition the conclusion on SCI-MAP-002 and do not
   reopen or close JINC.
5. **FRUIT (`SCI-FRUIT-001`)** is not audited. The residual-state hypothesis is
   conditioned on a future approved fruit-loop contract. After the core
   freeze, inspect only the narrow state/interface necessary to characterize
   the hypothesis.
6. **FLT (`SCI-FLT-001`)** facts are frozen through the pre-core handoff:
   identical realized operator/provenance requirements apply; direct
   pixelwise jackknife variance/S/N remain diagnostics; empirical-estimator
   and product validity remain reserved to NOI-002. Do not reopen FLT.
7. **NOI-002** exclusively owns sample-variance normalization/finite-N
   correction, empirical variance/weight calibration, downstream weight
   formula, S/N/detection/threshold/feedback authority, aperture-uncertainty
   estimator, and production realization-count/default policy. NOI-001 must
   not decide them.

## Later Phase 1 independent-core requirements

Under the current Phase 1 authorization, read only repository-level authorities
and the two frozen pre-core handoffs before deriving the independent core. Do
not open quarantined implementation paths, tests, diffs, or a post-core record. The core must
derive conditional mean/covariance using the sign-correlation law
`Q_epsilon`; coherent-sign cases; balance, duplicate/complement, dependence,
seed, and effective-unique-assignment constraints; physical-noise versus
randomization covariance; compact/resolved/extended/scan-synchronous source
leakage second moments; fixed versus data-derived coadd weights as an interface
fact; cross-observation terms; and propagation through realized mapmaking,
validity, coaddition, and filtering.

It must define RNG/state/cardinality/provenance/reproducibility and whether
the ensemble contains information adequate as a possible input to named later
uses. It must state requested/effective/observation-resolved/realized state;
identity, units, shape, indexing, validity and missing policy; analytic
limits; and proportional tiny deterministic fixtures tied to numbered
equations. No fixture may become a repository helper or costly campaign.

Freeze only
`doc/audits/packages/SCI-NOI-001_INDEPENDENT_CORE.tex`, commit it, report its
SHA-256, commit, timestamp, and planned first source-open event, then stop.

## Later Phase 2 trace and evidence boundaries

Only after explicit Phase 2 authorization may the auditor inspect the narrow
quarantined paths needed to trace exact insertion point, randomized unit,
RNG/seed/balance/eligibility/count, operator, cardinality, persistence,
product, and consumer behavior at
`d5015fe716971bf8ea617e8a187311bf5af05185`. Trace sequential/OpenMP
reachability and every state transition only as necessary for the NOI-001
question. Map each conclusion to numbered independent-core equations.

The final audit may draft an exact-`d5015fe716971bf8ea617e8a187311bf5af05185`
current-ensemble evidence
request. It may describe a current-versus-fruit-residual A/B as a conditional
future design only, because no approved residual-mode implementation SHA
exists. Do not request or launch Unity. Blank apertures/source-free regions
may test ensemble reproduction but may not define a production estimator or
tolerance. Do not invent tolerances. Any costly study remains held for
`FRAMEWORK-NUM-001` admission.

The audit may identify consumers and symptoms and propose bounded records to
NOI-002, FLT, and FRUIT; it must defer all downstream estimator validity to
NOI-002. It may not modify application/test/configuration files, canonical
ledger, canonical handoff registry, audit/repair branches, or production
status; repair, re-audit, integration, push, another audit, and production
authorization are prohibited.
