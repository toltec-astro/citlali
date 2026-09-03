# ALIGN-to-MAP Horizontal Scientific-Owner Decisions

Date: `2026-08-26`

Scientific owner: Grant Wilson

Status: approved owner input for coordinated downstream contract work

This record preserves decisions made while checking the end-to-end scientific
architecture from ALIGN through PTC and into MAP. It is not itself a package
contract, package freeze, implementation-conformity finding, validation
result, or readiness claim. Each affected package must incorporate the
applicable decision through its own governed revision and traceability.

## Approved Decisions

### A2M-OWNER-D001 — Paired x/r occurrence identity

The paired x/r readout represents one physical occurrence. Whether the
occurrence is an `independent_exposure` is evaluated for the explicitly named
component or product aspect. Replacement or synthesis of x does not make that
x value an independent detector measurement, but it does not rewrite the
origin of r or universally invalidate the physical pair.

### A2M-OWNER-D002 — Mandatory PTC-to-MAP route

Ordinary MAP input always comes from PTC. There is no direct CAL-to-MAP route.
Configuration may make PTC cleaning operationally neutral, but the data still
pass through PTC and carry the PTC handoff identity. This does not alter the
frozen PTC rule that explicitly disabling PTC produces neither a PTC result
nor a MAP result.

### A2M-OWNER-D003 — Necessary handoff and MAP-owned admission

Availability from PTC is necessary but not sufficient for contribution to a
MAP product. MAP cannot rescue a PTC-unavailable occurrence. MAP adds the
admission conditions for its own named map use. Producers preserve facts and
causes; the owner of the named use assigns their consequences. VAL may
register and evaluate that policy but does not author it.

### A2M-OWNER-D004 — Honest response availability, not prohibition

MAP must report whether the response information needed for a stated claim is
available and what that information means. An unavailable response limits the
claims supported by the original MAP product; it does not by itself prohibit
later scientific analysis. A scientist may later use simulations or other
authorized evidence to estimate the response and derive a modified product.

### A2M-OWNER-D005 — Versioned later response products

A later simulation-derived response or response-corrected map is a new,
versioned derivative. It must identify the exact parent MAP product, PTC
product, simulation/evidence product, source class, and applicable domain. It
does not alter the original MAP product's contents, provenance, availability
statements, or scientific claims.

### A2M-OWNER-D006 — Uncertainty and covariance disclosure

MAP must report what uncertainty/covariance information it actually provides,
its meaning, domain, and limitations. Absence of a complete covariance model
does not invalidate the map or prohibit later scientific analysis. Later
covariance estimates may be attached as new versioned products without
altering the original MAP product's claims.

### A2M-OWNER-D007 — Coordinate and projection ownership

ALIGN/AST own the sample sky coordinates, reference frame, time and
astrometric identity, and coordinate-validity facts delivered downstream. MAP
owns its target grid and the operation that turns admitted coordinates into
projection coefficients, including the declared scientific meaning of that
projection.

## Explicit Non-Decision

The discussion did **not** select one-hot versus fractional projection,
projection normalization, boundary-loss treatment, additional projection
classes, or their numerical domains. Those remain MAP-local matters already
represented by the existing decision ledger. This horizontal record must not
be used to answer them by implication.

## Incorporation Rule

The decisions above enter a package only through a content-bound package
revision. Existing normative identifiers are preserved; new identifiers are
append-only where a genuinely new obligation is required. Frozen upstream
packages remain unchanged unless a separately approved successor-authority
process is opened.
