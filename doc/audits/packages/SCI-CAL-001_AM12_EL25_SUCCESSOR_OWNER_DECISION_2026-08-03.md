# SCI-CAL-001 AM12/EL25 confirmation successor owner decision — 2026-08-03

Status: owner approved; a bounded successor protocol and delta execution may
proceed only after the required preservation, preflight, review, and readiness
gates pass

Package: `SCI-CAL-001`

Study: `SCI-CAL-001-AM12-EL25-CONFIRMATION-001`

Decision ID: `CAL-ATM-D005`

Authority: project owner

## Decision

The stopped 2026-08-02 confirmation is invalid because of an unregistered,
redundant construction-path guard, not because the frozen opacity coordinate,
atmosphere calculation, or one-percent representation-fidelity criterion
failed. The owner authorizes a new, versioned successor confirmation that
preserves all scientific inputs and changes only that guard's formalization.

The successor must replace the source's absolute `5.0e-17` comparison with a
documented, ULP-aware construction-path diagnostic admitting at most two
binary64 ULPs at the compared finite positive value. It remains diagnostic
only: the exact parsed 225-GHz transmission target, AM return/warning policy,
raw-output identity, passband authority, physical domain, and final numerical
metric remain independently required gates. A universal absolute epsilon is
not authorized.

## Frozen scientific content

The successor must preserve without tuning or broadening:

- all 16 tuples, truth profiles, opacity coordinates, elevation lattice, and
  AM 12.2/model inputs;
- the `toltec-passband-set-v1:sha256:5e6f38…` ECSV authority, integration
  convention, and spectral grid;
- the closed confirmation support, full eligible-sample airmass, `X_ref = 0`,
  q95 exclusion, and fail-closed ALIGN condition;
- `WARN-001`, exact parsed-transmission admission, and the one-percent
  numerical-representation-fidelity gate; and
- the distinction between software correctness, numerical fidelity, and
  observational calibration performance.

No Citlali or TolTECA source change, operator adoption, operational-domain
adoption, CAL repair, Unity activity, re-audit, or production change is
authorized by this decision.

## Required successor sequence

1. Make a byte-preserving durable copy of the stopped evidence collection at
   the recorded destination, verify its per-file and aggregate digests, and
   make the preserved copy read-only before it is considered for reuse.
2. Create a distinct successor protocol, runner, evaluator, schema, and
   tolerance/stop-condition register. Bind the original invalid execution and
   this decision as historical provenance; do not modify either.
3. Run the model-free all-16-case guard preflight, independent review, and
   expensive-execution readiness gate required by `FRAMEWORK-NUM-001` before
   any AM invocation.
4. If and only if those gates pass, independently reparse and admit the 672
   preserved full grids, 1,281 anchor outputs, and 13 scale traces; execute
   only the three unstarted scale searches and four missing 56-elevation grids
   (224 grids total) in a separate delta cache.
5. Evaluate the verified 896-grid union only under the frozen successor
   evaluator. Return a result with complete provenance, exact gate verdict,
   and an independent replay/review.

Any identity, preservation, warning, cache, preflight, readiness, or
scientific-gate failure stops the work without a retry, tolerance expansion,
or partial numerical conclusion.

## State effect

`CAL-ATM-D005` replaces neither the original invalid verdict nor its evidence
record. It authorizes only the bounded successor confirmation sequence above.
CAL remains `implementation_status: nonconformant`,
`validation_status: in_progress`, and `production_status: fail_closed` until
the separate repair and fresh re-audit gates are satisfied.
