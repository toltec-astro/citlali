# SCI-MAP v0.1 PTC-Handoff Scope Supersession

Date: `2026-08-26`

Status: proposed sanitized Stage A input; scientific-owner approval required

This cover supersedes only the upstream-handoff and adjacent-ownership
statements in the owner-approved SCI-MAP v0.1 Scope Brief. Every other approved
scope decision and exclusion remains in force.

## Revised Upstream Boundary

Ordinary SCI-MAP v0.1 begins from a realized PTC-transformed Stokes-I
timestream and its exact PTC product identity, membership, coefficient,
availability, cause, response, covariance, and provenance facts. The admitted
signal unit remains `mJy/beam`. The transformed timestream is an intermediate,
not an independent sky estimator.

There is no ordinary direct CAL-to-MAP route. A configuration that makes the
PTC transformation scientifically neutral still realizes a PTC product and
retains its identity. An explicitly disabled PTC request terminates on the
separately authorized upstream route and produces no MAP product. An invalid
PTC rank or otherwise unavailable required PTC result also produces no MAP
product.

## Revised Admission Boundary

PTC product availability is necessary but is not sufficient for contribution
to an ordinary map. MAP owns the policy for the named map use and adds its own
signal, coefficient, projection, boundary, companion, and product predicates.
MAP cannot promote a PTC-unavailable occurrence.

Upstream producers preserve the applicable facts and causes. MAP declares how
they affect its named use. VAL may register and evaluate the exact versioned
MAP-owned profile, but VAL does not author the profile, invent a missing
predicate, or turn a producer fact into universal eligibility.

## Revised Coordinate And Projection Boundary

ALIGN/AST own realized sample sky coordinates and their frame, time,
astrometric identity, validity, causes, uncertainty, and response state. MAP
owns the target map grid and the operation that maps admitted coordinates into
projection coefficients, including the scientific meaning, normalization,
extent, and boundary behavior it actually adopts.

This ownership statement does not choose one-hot versus fractional
projection, a normalization law, boundary-loss treatment, or another
projection class. Those remain open MAP-local decisions.

## Revised Response Boundary

Every MAP product declares the response information it actually carries, its
source and parent identity, meaning, domain, normalization, uncertainty, and
limitations, or a typed unavailable state with cause. Response-dependent
claims require response information sufficient for those claims. Response
unavailability does not by itself invalidate the map or prohibit later
scientific analysis.

A later response estimate, including one obtained from simulation, and any
response-corrected map are new versioned products. They bind to the exact raw
MAP parent, PTC parent, simulation or evidence product, source class, and
domain. They do not rewrite the original MAP product or its claims.

## Revised Uncertainty And Covariance Boundary

MAP reports the uncertainty/covariance information it actually provides, its
meaning, domain, assumptions, representation, omitted terms, and limitations.
Absence of a complete covariance model does not invalidate the map or prohibit
later scientific analysis. A claim that requires unavailable covariance must
remain unsupported rather than treating unknown terms as zero.

A later covariance estimate may be attached as a new versioned product bound
to the exact parent and applicable domain. It does not alter the original MAP
product's claims.

## Preserved Scope And Exclusions

The ordinary positive-coefficient normalized gridding estimator, shared raw
map identity, support vocabulary, conditional equations, provenance,
centered-integer compatible-grid coaddition, and existing owner-decision
ledger remain in scope. JINC, maximum-likelihood mapmaking, NOI construction,
filtering, source fitting, Beammap inference, Pointing/OOF interpretation,
FRUIT recurrence, polarimetry, and measured-R mapmaking remain outside this
revision.

The next author revises only the delta above, preserves the existing 52
requirement IDs and 25 prediction IDs, and appends an identifier only if no
existing obligation can carry a genuinely new rule without changing its
identity.
