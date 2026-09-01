# Citlali Downstream Scientific-Contract Roadmap

Date: `2026-08-26`

Updated: `2026-09-01` for SCI-JINC closure, SCI-NOI Stage A/Stage B handoff,
the SCI-FLT-FIXED conditional freeze, SCI-FRUIT Stage A acceptance, and the
owner direction to preserve its empirical-development sequence

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

### 5. Establish SCI-FRUIT

Specify fruit-loop feedback, learning, iteration, convergence, restart, and
lineage after the single-pass MAP/JINC/NOI/filtering authority line has stable
identities. FRUIT must not become a way to conceal unresolved single-pass
ownership, and the changed ordering does not itself admit a map or filtered
product as a feedback-model parent.

### 6. Establish source and observing-mode contracts

Develop source-fitting science after the map, response, uncertainty, filtering,
and FRUIT interfaces have been bounded. Develop Pointing and OOF as
mode-specific contracts whose scientific interpretations remain outside
ordinary MAP arithmetic and FRUIT recurrence, even where they reuse a
conforming gridding operator or consume an exact terminal FRUIT product.

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

At that Stage A checkpoint, implementation-blind Stage B had not been
launched. ODQ-101 is approved:
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
no empirical recentering, `B-1`, or physical-noise interpretation. ODQ-106 now
establishes that pointwise field as the ordinary primary representation without
promoting it to covariance; optional covariance methods remain separately
identified, dense full covariance is not universally required, unknown
covariance is not zero, and exact domain/rank/null/regularization disclosure is
required.
ODQ-107 now authorizes a finite-positive reciprocal of the initial second
moment as an inverse conditional second-moment scale, not inverse variance or
precision. Unavailable inputs remain unavailable rather than numerical zero,
regularization is separate, other inverse/precision/consumer weights remain
distinct, and no PTC/MAP promotion follows. ODQ-108, STD numerator and scale,
now selects the exact immutable normalized MAP signal divided by canonical
`sqrt(V_hat_cond)` on the exact compatible finite-positive intersection. The
unit-`1` output claims conditional-scale standardization only, keeps dependence
explicit, makes no significance claim, and leaves JINC separate.
ODQ-109 now admits plan-selected persisted, compact-regeneration, and streaming-
sufficient-statistic modes with exact mode/reproducibility/sufficiency/audit
identity, no default or silent fallback, and unchanged fail-closed completion.
ODQ-110A now keeps deterministic-transformation selection and definition with
the appropriate upstream/downstream scientific process and requires NOI to
apply exactly that transformation to every admitted compatible randomization
when estimating uncertainty for the exact transformed product. The route
remains unavailable until exact owner authority and parity are bound. ODQ-110B
now treats an owner-frozen Wiener transformation under ODQ-110A, makes any NOI-
informed owner learning/update a new immutable transformation/science/GEN/UNC
generation, and keeps per-member learning separate under ODQ-104. Every
numerical Wiener route remains gated. ODQ-110C now preserves FRUIT ownership,
limits fixed residual/terminal-product uncertainty to frozen-state conditional
meaning, separates NOI-informed successor generations from per-member ODQ-104
replay, and prohibits fixed/replayed mixing. All numerical FRUIT routes remain
gated. ODQ-111 now approves the four NOI-owned VAL profile identities and exact
consumer actions. Paired immutable `2026-08-30` SCI-VAL successors bind the
exact r0.18 sources and register all four profiles without altering prior MAP/
JINC records or numerical availability. All bounded Stage A owner decisions
and the process-only Registry/source prerequisite are complete. Conditional
Stage B was owner-authorized on `2026-08-30` at high reasoning effort from only
the exact 17-object implementation-blind author packet. The frozen MAP/pre-MAP
and JINC numerical-parent unavailability states remain unchanged. No
implementation, conformity, validation, achieved-performance, readiness,
production, Unity, filtering, source/mode, or FRUIT action follows.

The scientific owner's later `2026-08-30` direction launches recovery-first
SCI-FLT v0.1 Stage A at
[`packages/SCI-FLT/v0.1/`](packages/SCI-FLT/v0.1/). Recovery is complete for
owner review: it quarantines the implementation-informed inventory and
separates fixed transformation from Wiener, matched/source-sensitive, and
other inference-bearing families. The final owner scope repair selects
`SCI-FLT-FIXED`, retains `SCI-FLT-INF` only as a non-authoritative holding
tranche, narrows base v0.1 to strict-linear same-grid
`y=J_full L_Theta m`, admits fixed low-pass only as a qualified convolution
subtype, and makes full-footprint-only the sole edge/missing method. All
bounded Stage A scope decisions are resolved. At that checkpoint, the exact
repaired 17-object author candidate was SHA-bound but not releasable until
scientific-owner exact-byte approval and explicit launch; Stage B had not
begun. No algorithm,
frozen package, implementation, conformity,
validation, calibration, achieved performance, readiness, production, Unity,
source/mode, NOI, or FRUIT action follows.

The owner subsequently conditionally froze the exact SCI-FLT-FIXED v0.1
candidate by the record at commit
`7f9307ff4e1cda0f112f2398bb72f52a3f4f01d5`. That action preserves the
candidate's typed unavailable ordinary MAP/JINC parent routes and unregistered
FLT profiles; it does not authorize SCI-FLT-MATCHED, implementation,
conformity, validation, calibration, performance, readiness, production, or
Unity activity.

On `2026-08-31` the scientific owner changed the remaining review order and
authorized recovery-first SCI-FRUIT v0.1 Stage A at
[`packages/SCI-FRUIT/v0.1/`](packages/SCI-FRUIT/v0.1/), based exactly on the
SCI-FLT-FIXED freeze commit above. FRUIT now precedes source fitting and
Pointing/OOF. This is a sequencing and recovery action only: Stage B remains
unauthorized, all four candidate parent routes retain their exact unavailable
or provisional states, and no algorithm, frozen authority, validation,
production, or Unity action follows.

On `2026-09-01` the owner accepted the exact SCI-FRUIT Stage A `r0.8` result,
closing ODQ-001F, and requested an xhigh Stage B launch. Dispatch remains
unexecuted because the accepted claim-layer sequence places separately
authorized empirical development, held-out qualification, an owner-approved
qualified-method record, and a sealed method-specific author packet before
Stage B; none presently exists. The next owner action must either preserve
that sequence and authorize the empirical lane or explicitly supersede it with
a framework-only Stage B scope. The owner subsequently preserved the accepted
sequence and directed preparation of the empirical-lane authorization packet.
The successor `r0.1` packet returns a bounded Gate-0 registration-preparation
candidate; it does not authorize Gate 0, development, qualification, Stage B,
or Unity work. No numerical, implementation, validation, readiness,
production, fallback, or Unity authority follows.
