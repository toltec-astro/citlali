# Citlali Downstream Scientific-Contract Roadmap

Date: `2026-08-26`

Updated: `2026-08-29` for SCI-JINC closure and the repaired SCI-NOI Stage A
owner-review gate

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

SCI-MAP v0.1/r0.7.1 and SCI-JINC v0.1/r0.3 are frozen under their own exact
owner records. The scientific owner's `2026-08-29` direction launches bounded
SCI-NOI v0.1 Stage A at
[`packages/SCI-NOI/v0.1/`](packages/SCI-NOI/v0.1/).

The recovery-first launch produced a quarantined dossier and was followed by a
final implementation-blind Stage A scope repair. The repaired candidate now
contains collision-free `NOI-GEN`, `NOI-UNC`, and `NOI-STD` roles; exact
fixed-state/relearned graphs; ensemble-design, source-imprint,
target/estimator/rank/covariance, and STD tables; exact sanitized MAP, JINC,
and conditional pre-MAP PTC boundaries; four NOI-owned VAL profile drafts;
the FLT/Wiener/FRUIT deferral record; one granular decision candidate; and an
exclusive SHA-bound author packet plus closure report.

Implementation-blind Stage B has not been launched. ODQ-101 is approved:
fixed-state conditional-sign is the ordinary conditioning family, relearned
methods are separate, and the two member classes cannot be mixed. That approval
selected no numerical route. ODQ-102A then selected the ordinary
PTC-to-frozen-MAP route: the NOI modifier may be applied inline by MAP at the
numerical boundary, but NOI retains design and realization-product ownership
and the output is not ordinary MAP science. The route remains unavailable at
its coefficient and numerical `coverage_cut` gates. ODQ-102B then fixed one
assignment per stable realized detector/channel for all of that detector's
admitted samples throughout one observation; the same detector in another
observation is a different unit. ODQ-102C then selected network-stratified,
coefficient-balanced randomized signs, with balance evaluated separately
inside each readout network, complement-symmetric marginal `1/2`, no count
balance, and no cross-network cancellation. ODQ-102D, exact balanced
finite-design mechanics, is delegated to the implementation-blind scientific-
contract author; the tolerance-conditioned construction is nonbinding guidance
and creates no advance acceptance. ODQ-104 is explicitly approved: every GEN
method classifies scientifically consequential adjacent state, and each
relearned method identifies consequential rerun/relearn stages and resulting
changed state without an exhaustive implementation-provenance requirement.
ODQ-103 now approves the scientific boundary that randomization intends source
suppression but does not by construction establish source-free maps; exact
terminology is delegated to the Stage-B scientific author. ODQ-105A now fails
GEN closed for every UNC use if any admitted realization fails; rejected design
candidates are not members/failures and completed survivors cannot form a
partial ensemble. ODQ-105B now approves a zero-centered, exact-design-weighted
conditional randomization second moment on the common all-member domain, with
no empirical recentering, `B-1`, or physical-noise interpretation. ODQ-106,
covariance representation and rank, is next.
Conditional Stage B remains blocked on exact
bytes/hashes, granular owner decisions or explicit unavailability, and required
source/profile bindings. The frozen MAP/pre-MAP
and JINC numerical-parent unavailability states remain unchanged. No
implementation, conformity, validation, achieved-performance, readiness,
production, Unity, filtering, source/mode, or FRUIT action follows.
