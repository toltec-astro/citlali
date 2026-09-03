# SCI-MAP v0.1 r0.4 Change Log

Date: 2026-08-26

Status: integrated scientific/engineering draft; no implementation
conformance, validation, freeze, or readiness claim.

Revision r0.4 applies the scientific owner's bounded corrective directive to
the canonical MAP sources. It preserves the ordinary estimator, coadd
estimator, threshold equations, 52 requirement IDs, 25 prediction IDs, and
nine open owner-decision IDs.

## Scientific changes

- Fix the sole ordinary route as CAL to PTC to MAP. MAP has no direct CAL
  fallback; an identity-like PTC plan still produces a realized PTC product.
- Separate PTC-owned product identity, availability, producer-local validity,
  transformed signal, and MAP-facing coefficients from MAP-owned projection,
  coefficient admission, boundary, finite-value, support, and use conditions.
- Keep ALIGN/AST authority over coordinates, frames, astrometry, and coordinate
  validity; keep MAP authority over the target grid and `G_pi` semantics.
- Treat VAL as a registry/evaluator for named rules, never as their author.
- Preserve declared failure scopes without inventing a universal
  occurrence/product hierarchy.
- Keep paired PTC x/r detector coordinates upstream and explicitly reject any
  reinterpretation of r as MAP response.
- Require honest response and covariance disclosure. Missing or incomplete
  information neither invalidates the map nor prohibits later analysis; later
  estimates or corrected maps are new versioned products bound to the
  original MAP product and processing identity.
- Require scientifically identifiable MAP policy without prescribing JSON,
  hashes, or sidecar serialization, and avoid a closed-world registry of
  future uses.

## Corrected overreaches from the prior handoff draft

1. Removed x/r reinterpretation.
2. Removed invented availability propagation and failure-scope hierarchy.
3. Removed normative serialization requirements.
4. Removed the proposed exhaustive response-claim/use registry.

The existing OD-003, OD-004, and OD-008 questions are narrowed to their
remaining scientific choices without changing identifiers or silently closing
them. The retracted one-hot question is not reopened.
