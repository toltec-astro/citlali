# SCI-FLT-INF Stage A Scope Brief

Scope identity: `SCI-FLT-INF-STAGE-A-SCOPE v0.1/r0.5`

Status: sanitized owner-review holding study; ODQ-001 estimand, ODQ-002 map-
domain ownership/product role, and ODQ-003 ordinary-MAP parents/grouping
approved; ODQ-004 option development author-delegated; remaining scope not
approved; not an author input

## Program adherence and prior-work recovery

This brief is governed by the
[Scientific Contract Library Program](../../README.md). The recovery record
in [`PRIOR_WORK.md`](PRIOR_WORK.md) was completed before this brief was
drafted. The program roadmap requires deterministic filtering to be separated
from Wiener or other inference-bearing methods whenever estimand, prior,
transfer, or uncertainty meaning differs.

## Assignment

Determine the smallest scientifically coherent future contract packages for
inference-bearing map-domain operations currently or historically grouped
under filtering. Preserve the owner-selected ODQ-001 estimand and, for each
remaining candidate family, identify without selecting:

1. the estimand and scientific claim;
2. the exact parent product and observation/coadd grouping;
3. fixed parameters, noise-model state, learned state, and their provenance;
4. the operator or estimator and its order relative to other transformations;
5. response, transfer, normalization, units, beam meaning, and null space;
6. uncertainty/covariance meaning and whether any denominator is merely a
   normalization coefficient;
7. support, edge, missing/nonfinite, validity, and failure rules;
8. product identities and atomic lifecycle;
9. fixed-state versus successor-generation versus per-member-relearned NOI
   parity; and
10. package ownership, product role, exclusions, and cross-package
    dependencies.

## Candidate families in scope

- the active noise-PSD- and template-dependent full map path;
- a genuine Wiener/posterior reconstruction only as a separate deferred
  family excluded from the selected package;
- matched or generalized least-squares template-amplitude estimation;
- map-, noise-, or externally learned state frozen before use;
- per-member state relearning as a distinct NOI-GEN method;
- data-derived Fourier-mode selection/destriping;
- automatic method selection, fallback, or substitution;
- data-derived edge, support, taper, and background-fill conditioning;
- NOI-derived coefficient calibration and standardized products; and
- source-conditioned or source-learned transformations only as recovered
  families to exclude from the selected package.

## Owner-selected estimand, ownership, and product role

`SCI-FLT-INF-ODQ-001` is approved. Recovery of the historical full path shall
use the scientific identity **optimal matched-template amplitude estimator**.
The estimator uses the supplied kernel as the expected template response and
the declared noise model to estimate the template amplitude at each admitted
map position. The point-source-response kernel is the ordinary point-source
specialization; another scientifically defined kernel defines another
template-amplitude specialization.

The exact normalization must be unbiased for a matching signal's amplitude
under the declared model, support, edge, missing/nonfinite, validity, response,
and other assumptions. The future package must make the optimality criterion
explicit. The method is not a posterior/Wiener reconstruction of the sky, and
source-shaped convolution alone is not the matched estimator. Any genuine
posterior reconstruction remains a separate future method.

`SCI-FLT-INF-ODQ-002` is also approved. The selected method belongs to a
narrow map-domain filtering package, and its published signal product is a
matched-filtered version of the exact admitted input map product or products.
It preserves the applicable map-domain structure and semantics of its parent;
the exact parent, units, response, uncertainty, validity, and bundle facts
remain later decisions.

The package does not perform or require source detection, candidate selection,
catalog construction, peak interpretation, deblending, source fitting, or any
other source-analysis behavior. This Stage A study introduces no source-
estimation package or SRC ownership boundary. A later independent scientific
contract may consume matched-filtered maps if separately authorized.

`SCI-FLT-INF-ODQ-003` admits both exact immutable normalized ordinary-MAP
observation bundles and exact immutable normalized ordinary-MAP coadd bundles.
They are distinct parent/grouping identities. Learning and application are
observation-local for an observation parent and coadd-local for a coadd parent.
No equivalence, commutation, filtered-result coaddition, or cross-observation
combination is approved. JINC, SCI-FLT-FIXED derivatives, and other derived
map parents are excluded from v0.1.

`SCI-FLT-INF-ODQ-004` delegates noise/covariance, spectral-weighting, and
parent-coefficient option development to the future implementation-blind
contract author. The Scientific Rationale and Contract and Engineering
Conformance Specification must present the same bounded option identities and
consequences. Citlali's historical radially symmetrized average map noise PSD
is admitted only as a candidate to examine; it is not a selected default,
covariance authority, or proof of stationarity, isotropy, or optimality. The
owner must dispose of the authored options before freeze or numerical
authorization.

## Required distinctions

The study shall not merge two cases merely because they reuse FFTs, a template,
a noise spectrum, or the same numerical class. Separate identities are
required whenever any of the following differs:

- reconstructed field versus local/scalar template amplitude;
- prior-bearing posterior versus frequentist/GLS estimator;
- declared fixed state versus state learned from the parent or from NOI;
- one frozen state applied to all NOI members versus state relearned per member;
- observation parent versus coadd parent;
- fixed operator versus input-dependent mode/support selection;
- response-preserving transformation versus response-corrected estimator;
- direct map product versus derived coefficient, uncertainty, or standardized
  product; or
- fail-closed unavailability versus a named alternative-method selection.

## Included authorities

- exact program and manager documents at the study base;
- frozen MAP and JINC scientific authorities only through their exact
  established parent/boundary facts;
- frozen SCI-NOI authority at
  `f28d7a2617160febca85c1c40e6f7ba7494e266e`, with exact object bindings;
- immutable SCI-FLT-FIXED Stage A objects only as protected neighboring scope
  and byte-preservation references;
- recovered implementation-independent mathematics after explicit
  sanitization and disposition; and
- implementation/config/schema/history evidence only in the quarantined
  manager dossier.

## Exclusions

- all active SCI-FLT-FIXED Stage B task/worktree/branch/draft/output material;
- any edit to the 17 SCI-FLT-FIXED author objects or their manifest;
- implementation or algorithm changes;
- fresh numerical derivation presented as authority before recovery and owner
  scope approval;
- Unity access or validation;
- FRUIT recurrence, learning, stopping, restart, or source-model science;
- RTC temporal filtering/destriping;
- source detection, candidate selection, catalog construction, peak
  interpretation, deblending, fitting, significance, completeness, purity,
  morphology, or other source-analysis behavior;
- JINC, SCI-FLT-FIXED-derived, or other derived-map parents in v0.1;
- absolute CAL authority, passband/color correction, or cross-band covariance;
- inferred MAP/JINC precision or covariance;
- production defaults, observed behavior, or historical labels promoted into
  science; and
- a combined `SCI-FLT-INF` contract solely for administrative convenience.

## Intended output of this Stage A study

The intended output is a recovery packet and an ordered owner-decision
walkthrough. It may recommend multiple successor package identities and may
declare some families too immature for a Stage B author. It shall not contain
an exclusive author manifest, Stage B assignment, normative requirements,
falsifiable acceptance predictions, implementation mapping, or freeze record.

## Stage A completion test

This study is ready for owner walkthrough when:

- every recovered family has a recorded disposition;
- the existing full path's apparent algebra and actual scientific ambiguity
  are stated separately;
- exact frozen-NOI parity constraints are content-bound;
- inferred, proposed, unavailable, and authoritative statements are visibly
  distinct;
- contradictions and absent authorities are not repaired by invention;
- a package split and ordered decision ledger are present;
- proposed sanitized author material contains no implementation facts; and
- the fixed-filter author packet and manifest verify byte-identical to
  `cd55752e716051383da54356833ef0fac20b083a`.
