# Citlali Downstream Scientific-Contract Roadmap

Date: `2026-08-26`

Updated: `2026-08-28` for the owner-directed SCI-JINC Stage A launch

Status: scientific-owner-approved program sequencing record

This record preserves the agreed progression after the frozen ALIGN-through-PTC
handoff. It is a program plan, not a scientific contract, package freeze,
implementation-conformity finding, validation result, or production-readiness
claim.

## Starting Boundary

The next downstream work begins from the existing package authorities rather
than re-deriving them:

- ALIGN and AST own the realized sample coordinate, frame, time, astrometric
  identity, and coordinate-validity facts;
- RTC and CAL retain their frozen conditioning and calibration authorities;
- PTC owns the transformed timestream handed to ordinary mapmaking and the
  PTC-produced facts accompanying that handoff; and
- VAL remains a registry and evaluator of policy owned by scientific
  producers and consumers. It does not become the author of MAP policy.

The compilation-independent WP-7 clean-room closure lane remains separately
governed. This roadmap neither closes that lane nor makes a finding about
implementation behavior.

## Ordered Contract Tranches

### 1. Close SCI-MAP v0.1

Reconcile the existing ordinary-map contract with the now-frozen PTC-to-MAP
handoff, the current ALIGN/AST coordinate boundary, and the owner decisions in
[`ALIGN_TO_MAP_HORIZONTAL_OWNER_DECISIONS_2026-08-26.md`](ALIGN_TO_MAP_HORIZONTAL_OWNER_DECISIONS_2026-08-26.md).
Preserve MAP's existing local decision ledger. Do not silently decide
projection class, normalization, edge handling, support thresholds, or other
open MAP science while repairing the horizontal package boundary.

### 2. Establish SCI-JINC v0.1

Treat the recovered signed-coefficient JINC estimator as a separate contract.
Its signed estimator, normalization, response, support, and covariance
semantics must not be folded into the ordinary positive-coefficient SCI-MAP
contract.

### 3. Establish SCI-NOI v0.1

Define noise-realization and empirical-uncertainty products, including their
conditioning, independence, ensemble, domain, identity, and attachment rules.
NOI may extend the uncertainty evidence available to later products without
retroactively changing what an earlier MAP bundle claimed.

### 4. Establish filtering contracts

Recover and separate deterministic convolution/low-pass transfer from Wiener
or other inference-bearing filtering. Use distinct contracts when the
estimand, prior, transfer, or uncertainty meaning differs. Do not treat all
filtering as one generic operation.

### 5. Establish source and observing-mode contracts

Develop source-fitting science after the map, response, and uncertainty
interfaces are stable. Develop Pointing and OOF as mode-specific contracts
whose scientific interpretations remain outside ordinary MAP arithmetic, even
where they reuse a conforming gridding operator.

### 6. Establish SCI-FRUIT last

Specify fruit-loop feedback, learning, iteration, convergence, restart, and
lineage only after the single-pass upstream and downstream products have
stable identities. FRUIT must not become a way to conceal unresolved
single-pass ownership.

## Cross-Cutting VAL Lane

Each new producer or consumer owns the policy for its named use. VAL may
register and evaluate the exact versioned profile only after that ownership is
explicit. The MAP source/profile binding therefore proceeds alongside MAP
closure, but VAL does not invent map admission, response-use, uncertainty, or
publication policy.

## Gates Between Tranches

Every tranche follows the Scientific Contract Library Program:

1. recover and classify prior work before commissioning new derivation;
2. prepare an implementation-informed internal dossier and a sanitized Stage A
   packet;
3. obtain scientific-owner approval of scope and exact author inputs;
4. commission a fresh implementation-blind Stage B author only after that
   approval;
5. manager-review the shared formal core and both document views;
6. resolve owner decisions explicitly, preserving normative identifiers;
7. freeze scientific authority separately from implementation conformity,
   validation, achieved performance, and production readiness.

## Anti-Drift And Anti-Repetition Rules

- Begin every new package by linking to the program charter, this roadmap, and
  the applicable predecessor authority.
- Search the prior-work registry and package-local recovery records before
  creating new science. Adopt, abstract, supersede, defer, or exclude the
  result explicitly.
- Do not silently edit a frozen upstream contract to simplify a downstream
  boundary.
- Do not turn an interface repair into a reopening of unrelated local
  scientific questions.
- Keep later response, covariance, or corrected products versioned and bound
  to their exact parents; do not rewrite the claims of the original product.
- Record absence or incompleteness honestly. It limits the claims supported by
  a product but is not automatically a prohibition on later scientific use.

## Immediate Authorized Action

The bounded SCI-MAP reopening named in the original `2026-08-26` roadmap has
completed and SCI-MAP v0.1/r0.7.1 is frozen under its own exact owner record.
The scientific owner's `2026-08-28` successor direction launches bounded
SCI-JINC v0.1 Stage A at
[`packages/SCI-JINC/v0.1/`](packages/SCI-JINC/v0.1/).

This launch authorizes prior-work recovery, implementation-informed scope
investigation, a quarantined dossier, a sanitized Scope Brief, decision
records, and an exact proposed author packet only. Implementation-blind Stage B
authorship remains blocked until the scientific owner approves the exact
Scope Brief and packet. No implementation, conformity, validation, achieved-
performance, readiness, production, Unity, or later-tranche action follows.
