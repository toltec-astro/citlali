# SCI-FLT v0.1 Scientific-Owner Decision Ledger

Date: `2026-08-30`

Status: Stage A walkthrough ready; all questions open

## Program Adherence And Prior-Work Recovery

These are the bounded scientific choices that recovery could not resolve from
current frozen or approved authority. They are not implementation questions.
No option is selected until the scientific owner answers it explicitly. The
answers will be transcribed into a separate content-bound authorship record;
this ledger itself is not future author input.

## FLT-ODQ-101 — Split The Tranche Before Stage B

**Question.** Should `SCI-FLT` remain one Stage B package, or should fixed
deterministic transformations be separated from inference-bearing methods
before authorship?

**Recommended decision.** Use `SCI-FLT` only as the tranche name. Commission
`SCI-FLT-DET v0.1` first for fully fixed deterministic map-domain
transformations. Hold Wiener, matched/template-amplitude, source-learned, and
data-derived spectral-selection work in `SCI-FLT-INF`, with further method
splits required unless their estimand, prior/learned state, response,
uncertainty, and lifecycle are shown to coincide.

**Alternative.** One SCI-FLT contract with separately typed chapters. This
reduces package count but creates a high risk that a common software mechanism
or “filter” label will hide different estimands and conditioning.

**Consequence.** Approval fixes the scope and identity of the first fresh
Stage B author task. Rejection requires a revised taxonomy and author packet
before Stage B.

**Status:** `open`; first walkthrough question.

## FLT-ODQ-102 — Membership Of The Fixed Deterministic Family

**Question.** What conditions must a method satisfy to enter `SCI-FLT-DET`,
and are fixed convolution and fixed low-pass one contract with distinct method
identities?

**Recommended decision.** Require every coefficient, offset, template/kernel,
domain, edge/padding rule, normalization, support, and missing-data rule to be
fixed before application to the admitted parent random field. Admit fixed
convolution and a scientifically specified fixed low-pass method into one
deterministic package as distinct method identities. Move parent-derived local
normalization, adaptive masks, learned templates, and data-derived offsets to
an inference-bearing or explicitly conditioned family.

**Consequence.** Determines whether a method can use fixed linear/affine
response and covariance propagation, and prevents a learned total method from
being called deterministic merely because its last step is convolution.

**Status:** `open`.

## FLT-ODQ-103 — Parent And Ordering Identities

**Question.** Which initial parents and orders should `SCI-FLT-DET v0.1`
cover: observation MAP, raw coadd MAP, JINC, filter-before-coadd, and/or
filter-after-coadd?

**Recommended decision.** Authorize observation-MAP and raw-coadd-MAP parents
as distinct methods/products. Treat filter-after-coadd and coadd-after-filter
as noninterchangeable unless a later contract establishes exact equivalence.
Keep JINC explicitly unavailable in v0.1 until an applicable numerical parent,
response, support, and covariance boundary is bound; do not import ordinary
MAP rules.

**Consequence.** Fixes parent compatibility, lineage, response, and what a
consumer may compare or combine.

**Status:** `open`.

## FLT-ODQ-104 — Edge, Padding, Missing Data, And Scientific Admission

**Question.** Should numerical fill/padding be excluded from scientific
admission by requiring every admitted output footprint to lie within the
admitted parent-valid domain, or should partial-support/local-renormalization
methods be admitted now?

**Recommended decision.** Reaffirm the scientific core of historical FLT-D001:
fill/padding is a numerical boundary device only; the admitted domain is
conservatively eroded by the exact realized operator footprint so no admitted
output depends on fill, padding, missing, or invalid parent samples. Permit
partial-support, local normalization, inpainting, or stochastic fill only as
future separately named methods with their own response and uncertainty.

**Consequence.** Selects the validity meaning and whether fill covariance is
needed for admitted pixels. Under the recommendation, fill-influenced output
is not scientific and no fill-covariance claim is required for it.

**Status:** `open`.

## FLT-ODQ-105 — Stored Amplitude And Response Products

**Question.** What is the initial deterministic transformed product intended
to estimate, and which response object must accompany it?

**Recommended decision.** Reaffirm the scientific core of historical
FLT-D002: store a transformed map amplitude in declared map units, not an
automatic flux or matched amplitude. Require an identically transformed,
content-bound unit-source response/kernel when available, with exact
normalization, centering, support, signed integral, peak response, pixel solid
angle, and effective beam solid angle identities. Allow later user/consumer
peak or aperture response correction subject to its declared source/background
model and CAL; do not create automatic photometry in `SCI-FLT-DET`.

**Consequence.** Separates smoothing, response correction, and source
estimation and fixes the minimum product pair.

**Status:** `open`.

## FLT-ODQ-106 — Formal Propagation Versus SCI-NOI Uncertainty

**Question.** What uncertainty/covariance products may `SCI-FLT-DET` own, and
how should historical FLT-D003 be dispositioned?

**Recommended decision.** Let FLT define only deterministic propagation of an
available declared parent covariance through the exact fixed operator,
including honest distinction among full covariance, structured covariance,
pointwise variance, and absence. Assign empirical conditional uncertainty,
covariance estimation, inverse conditional scale, and standardized signal to
SCI-NOI under the approved exact-transformation boundary. Supersede the old
placement of a robust empirical scale inside FLT; retain it only if NOI later
defines it as an NOI-owned attachment to the exact FLT product.

**Consequence.** Prevents a weight, denominator, diagonal variance, or empirical
width from silently becoming precision, full covariance, or significance.

**Status:** `open`.

## FLT-ODQ-107 — Inference-Bearing Method Identities And State Lifecycle

**Question.** Should Wiener, matched/template-amplitude, source-learned, and
data-derived spectral-selection methods share one inference-bearing contract,
and what happens when required state is unavailable?

**Recommended decision.** Keep them as separate method packages or separate
Stage A assignments unless a later scope analysis establishes a common
estimand and state model. Require explicit `fixed_external`,
`fixed_parent_bound`, `learn_then_freeze`, `successor_update`, or
`per_member_relearned` identity. Unavailable required state fails or selects a
separately requested and separately identified method; no silent fallback may
retain the requested product identity.

**Consequence.** Fixes source-imprint, bias, response, and NOI conditioning,
and determines whether a current same-label substitute is scientifically a
different product.

**Status:** `open`.

## FLT-ODQ-108 — Downstream Consumer Admission

**Question.** Should `SCI-FLT v0.1` authorize any direct Beammap, Pointing,
OOF, source-fit, or FRUIT use?

**Recommended decision.** No generic direct admission. FLT defines exact
transformed products and response/support/validity identities. SCI-BEAM,
future source/mode packages, and SCI-FRUIT must separately admit the exact
product for a named use. Preserve approved SCI-NOI fixed/relearned boundaries
for every uncertainty attachment and FRUIT generation.

**Consequence.** Prevents product existence from being mistaken for a source
fit, correction, feedback, or iterative-science authorization.

**Status:** `open`.

## Walkthrough Order And Closure Rule

Decisions are taken in numerical order because FLT-ODQ-101 changes the package
and author-task identity. Stage B remains blocked until all decisions relevant
to the selected first package are approved, transcribed verbatim into
[`SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md`](SCIENTIFIC_OWNER_DECISIONS_FOR_AUTHORSHIP.md),
and included in a finalized exact packet manifest.
