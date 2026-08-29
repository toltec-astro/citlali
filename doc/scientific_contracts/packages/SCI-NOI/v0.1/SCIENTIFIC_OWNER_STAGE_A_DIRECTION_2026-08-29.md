# SCI-NOI v0.1 — Scientific-Owner Stage A Direction

Date: `2026-08-29`

Scientific owner: Grant Wilson

Status: approved launch direction for recovery-first Stage A at high reasoning
effort

## Approved Package Shape

Use one SCI-NOI package with two primary scientific families:

1. noise-realization generation; and
2. empirical uncertainty inference from a declared realization ensemble.

Maintain a hard typed boundary between them. Creating a jackknife,
sign-flipped, randomized, or otherwise signal-suppressed ensemble does not by
itself establish variance, covariance, calibrated signal-to-noise, or
statistical significance.

Within the same package, define a third, separately identified operation and
product role: a derived standardized-signal product combining an immutable MAP
or JINC parent with an authorized uncertainty product. That product is not an
uncertainty estimate. Unless justified and validated, its claim is only
“standardized by the stated empirical scale,” not calibrated Gaussian
significance or detection probability.

## Required Recovery

Recover and classify rather than reinvent:

- existing detector-, scan-, subscan-, observation-, and coadd-level
  randomization or jackknife methods;
- balance, grouping, cardinality, seed, and requested/effective/realized
  configuration rules;
- the stage at which randomization occurs;
- fixed and relearned RTC, PTC, AST, MAP, and JINC state;
- variance, RMS, empirical-weight, covariance-summary, and `sig2noise`
  definitions;
- finite-ensemble correction, centering, effective ensemble size, and
  correlated-pseudo-realization treatment;
- source cancellation/leakage, scan-synchronous residuals, filtering effects,
  and support restrictions;
- dependencies from MAP, JINC, deterministic filtering, Wiener filtering,
  Beammap, Pointing, OOF, and FRUIT;
- the earlier SCI-NOI-001/002 reasoning and internal noise derivations; and
- validation only as evidence, never as scientific authority.

Fixed-state and relearned realization generation are distinct possible NOI
methods and shall not be mixed in one ensemble. Fixed-state generation is
conditional on the realized reduction and learned operator state. Relearned
generation attempts to include variation from the learning procedure itself.
Fixed-state generation may be the likely ordinary baseline, but neither method
is selected as ordinary authority by Stage A; that remains an owner decision.

## Required Typed Separations

Keep separately typed:

- sample validity;
- PTC/MAP analysis or gridding coefficients;
- realization signs or randomization assignments;
- realization availability/QC;
- empirical variance or covariance;
- empirical inverse-variance or weight representations; and
- standardized signal.

An empirical NOI weight cannot become a MAP-facing PTC coefficient because
the numerical values are both weight-like. Any future crossing requires
explicit authorization and semantics.

MAP and JINC parents remain immutable. NOI outputs attach as versioned
companions with exact parent and method identity. Full pixel covariance is not
mandatory: diagonal variance, stationary/kernel summaries, structured
covariance, retained empirical ensembles, and unavailable covariance are all
possible honest states when their domains and limitations are explicit.

Persistence is plan-controlled. Individual realizations may be written,
transient, or reduced through mathematically equivalent streaming sufficient
statistics. When not persisted, record the decision and audit/reconstruction
limitation. Dense per-sample provenance and every random sign need not be
stored when immutable parents, identified algorithm/version, seed, and
configuration permit exact regeneration.

## Authorized Deliverable And Stop Rule

Stage A shall return:

- a Scope Brief;
- recovered prior-work and implementation inventory;
- ownership and boundary classification;
- scientific ambiguities or contradictions;
- a proposed initial operator/product taxonomy; and
- a bounded list of genuine owner decisions.

Do not begin Stage B drafting, implementation, or modification of frozen MAP,
JINC, or PTC contracts. Return the Stage A packet for review and begin the
owner-decision walkthrough with the first scientifically consequential open
question.
