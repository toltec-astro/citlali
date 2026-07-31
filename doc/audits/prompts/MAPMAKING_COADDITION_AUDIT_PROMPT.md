# First package audit: mapmaking and coaddition

This is the prepared prompt for the first new scientific-contract audit. Do
not execute it while creating or reviewing the framework.

## Assignment and package boundary

Audit `SCI-MAP-001`, **Shared and naive mapmaking plus observation
coaddition**, as a Tier A package.

At audit launch, verify the current `codex/refactor-mainline` pointer and
replace every launch field below:

- repository: `/Users/gwilson/GitHub/citlali-refactor`
- inventory snapshot: `9aae0e669384c5c0c0dda93debc194d6b8dac787`
- governing source SHA: `TO_SET_CURRENT_FULL_MAINLINE_SHA`
- audit branch: `codex/audit-sci-map-001`
- suggested worktree: `/private/tmp/citlali-audit-sci-map-001`
- coordinator ledger commit: `TO_SET_FRAMEWORK_COMMIT`

The audit includes:

- the shared map-product identity and the ordinary naive map estimator;
- sample-to-pixel projection and weighted accumulation;
- pre-normalized and normalized signal, formal weight, kernel, hits/coverage,
  validity, support, and non-finite behavior;
- observation-map identity and inverse-variance-like coaddition;
- map and coadd units, frames, WCS, metadata, and provenance;
- noise-realization accumulation only far enough to prove whether it uses the
  same map operator; and
- downstream input contracts needed by noise calibration, filters, source
  products, mode fits, Beammap, and fruit-loop feedback.

The audit excludes:

- RTC/PTC algorithm internals;
- final calibration/astrometry/flagging policy, which enter as named abstract
  inputs and dependencies;
- the full noise/jackknife and empirical-calibration estimators
  (`SCI-NOI-001`, `SCI-NOI-002`);
- JINC internal mathematics (`SCI-MAP-002`), except for checking that the
  shared map-product contract is capable of describing it;
- rejected production maximum-likelihood mapmaking;
- post-map `convolve` and Wiener filtering (`SCI-FLT-001`,
  `SCI-FLT-002`); and
- source fitting, mode-specific products, Beammap inference, and fruit-loop
  feedback.

If the independent derivation shows that naive mapmaking and coaddition cannot
be reviewed coherently as one package, preserve the derivation and return
verdict `split` with stable successor-ID proposals. Do not silently narrow the
scope after source inspection.

## Hard boundaries

Read and follow repository `AGENTS.md`, the TolTEC context skill,
`doc/ARCHITECTURE.md`, and `doc/SCIENTIFIC_CONVENTIONS.md`. Verify and
report all relevant worktrees, branches, full SHAs, upstream relationships,
and dirty states before creating the isolated audit worktree.

Do not modify application code, tests, build files, existing production
documentation, the framework ledger, frozen candidates, prior audits,
fruit-loop worktrees, or the Conan lane. Do not push, merge, rebase,
cherry-pick, download, install, use the network, or connect to Unity.

Treat the implementation paths below as quarantined until the independent
mathematical core is frozen and hashed:

- `include/citlali/core/mapmaking/naive_mm.h`
- `include/citlali/core/mapmaking/map.h`
- `src/citlali/core/mapmaking/map.cpp`
- `include/citlali/core/pipeline/mapmaking_execution_plan.h`
- `include/citlali/core/pipeline/mapmaking_provenance*.h`
- `include/citlali/core/pipeline/observation_coadd_accumulation.h`
- mapmaking/coadd output and metadata writers;
- focused mapmaking/coadd tests and implementation diffs.

Repository-level product intentions and scientific conventions may be read
before the freeze. Record any unavoidable prior exposure to implementation
details.

## Independent derivation against abstract inputs

Create
`doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`. Derive the contract
without using formulas from the quarantined implementation.

Use explicit abstract inputs such as:

- processed detector samples \(d_{td}\) with named signal units;
- a sample/detector eligibility operator \(M_{td}\);
- detector/sample weighting or inverse covariance \(W\), whose meaning is an
  upstream assumption rather than a fact to invent here;
- a pointing/projection operator \(P_{ptd}\) with coordinate-frame and pixel
  conventions;
- sample duration or other exposure coefficients;
- calibrated response \(C\) and any beam/kernel representation; and
- observation index, array/network/detector/map identity.

Calibration (`SCI-CAL-001`), pointing/astrometry (`SCI-AST-001`), processed
timestream weight/covariance (`SCI-PTC-001`), and validity/flags
(`SCI-VAL-001`) are allowed to remain abstract dependencies. For each one,
state:

1. the exact fact assumed;
2. whether the map conclusion is conditional on it;
3. a falsifiable interface test; and
4. the downstream restriction until it is closed.

Do not let an unresolved upstream package prevent an independent map estimator
from being written. Do not silently choose the upstream meaning either.

At minimum derive and distinguish:

1. the general linear or generalized least-squares map estimator and its full
   covariance;
2. the exact diagonal/independence approximation, if one is claimed for the
   naive estimator;
3. accumulated numerator, normalization denominator, stored normalized signal,
   and any kernel/response product;
4. statistical inverse variance versus gridding weight, hits, exposure
   coverage, support, and validity;
5. output-pixel covariance induced by projection, shared detector noise, and
   preprocessing;
6. response to constant sky, a delta input, a beam/template source, extended
   modes, variable coverage, and boundaries;
7. missing/flagged/non-finite samples and zero/low/negative weights;
8. units through sample weighting and normalization;
9. map grouping and array/network/detector identity;
10. observation coaddition for unequal coverage and covariance, including the
    estimator whose weights are used;
11. conditions under which coadd weight equals summed observation inverse
    variance and when cross-observation covariance invalidates that result;
12. requested, effective, observation-resolved, and realized map/coadd state;
    and
13. the exact input contract exposed to noise, filtering, source, mode, and
    fruit-loop packages.

If formal map weight cannot be derived without a PTC covariance claim, write
the conditional formula and name `SCI-PTC-001`; do not relabel an
accumulation denominator as inverse variance by convenience.

Freeze and commit the independent bytes before source inspection:

```bash
shasum -a 256 doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex
git add doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex
git commit -m "docs: freeze SCI-MAP-001 independent core"
```

Record the digest, freeze commit, timestamp, and first source-inspection event.
Any later scientific correction is a documented successor to the frozen core.

## Post-freeze implementation audit

After the freeze, inspect the exact governing SHA and trace:

- naive sequential and parallel accumulation and merge paths;
- signal, kernel, weight, coverage, hits, and noise-cube coefficients;
- normalization order and invalid/zero support behavior;
- sample/detector flags, low/non-finite weights, edge/pixel bounds, and map
  grouping;
- observation coadd accumulation, centering/mean subtraction, and final
  weights;
- mapmaking/coadd requested, effective, observation-resolved, and realized
  lifecycle;
- FITS products, WCS, units, metadata, cardinality, and provenance;
- JINC compatibility with the common product vocabulary without reopening its
  mature internals;
- consumers in noise generation/calibration, `convolve`, Wiener, source
  products, Pointing/OOF, Beammap, and fruit loops; and
- existing tests and accepted evidence without treating regression equality as
  the contract.

Trace every relevant product to numbered equations. A package product table
must include at least signal, formal weight, kernel, coverage, hits/validity
when present, noise realizations at the map boundary, and their coadd
counterparts. State exact extension/variable identity and whether each is an
estimator, normalization, response, support, exposure, or diagnostic.

## Falsifiable evidence plan

Require or justify N/A for:

- one-pixel, identity-projection, uniform-weight, unequal-weight, masked,
  non-finite, zero-support, boundary, and multiple-detector deterministic
  fixtures;
- direct small-matrix comparison with the independently formed \(A\), mean,
  and \(A C A^{\mathsf T}\);
- constant, delta, beam-shaped, resolved, and extended injections;
- observation coadds with known unequal variances and controlled
  cross-observation covariance;
- exact sequential/OpenMP equivalence or a pre-registered numerical policy;
- same-SHA Unity point and science cases covering all arrays, variable
  coverage, observation maps, and coadds;
- blank-control map covariance, formal-versus-empirical calibration, and false
  standardized-signal behavior;
- exact products, WCS, units, metadata, provenance, zero unexpected errors,
  and no required-data skips; and
- astronomical amplitude/shape/unit recovery where it adds information beyond
  deterministic injection tests.

Use `doc/audits/templates/UNITY_EVIDENCE_REQUEST_TEMPLATE.md` for the external
request. Grant selects/runs Unity evidence; Codex only specifies and audits the
returned bundle. Pre-register tolerances from analytic precision,
finite-sample behavior, repeatability, or an explicit scientific decision.

## Deliverables

Create and compile:

- `doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`
- `doc/audits/packages/SCI-MAP-001_SCIENTIFIC_CONTRACT_AUDIT.tex`

The final audit must contain:

- exact independence record and numbered estimator/covariance/response
  equations;
- source/equation conformity trace;
- product/consumer matrix;
- findings separated into the five controlled classes;
- dependency records for `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and
  `SCI-VAL-001`;
- local and requested external validation with N/A rationales;
- downstream allowlist, restrictions, and fail-closed policy;
- independent contract, implementation, validation, production statuses and
  verdict; and
- a machine-readable proposed update for the `SCI-MAP-001` ledger record.

The YAML fragment is returned to the coordinator; do not edit the canonical
ledger from the audit branch.

Commit the audit documents only. Report branch, commits, parent/governing SHA,
worktree, clean state, core hash, compile/render evidence, conclusions,
findings, dependencies, statuses, verdict, and exact unsupplied Unity request.

Stop. Do not repair Citlali, audit JINC/noise/filtering next, launch Unity, push,
or integrate.
