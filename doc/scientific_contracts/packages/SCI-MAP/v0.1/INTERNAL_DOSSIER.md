# SCI-MAP — Internal Stage A Dossier

Status: internal scope evidence; never an author reference

Date: `2026-08-16`

This dossier records why ordinary mapmaking and observation coaddition form a
coherent package and where current software and historical audit material
appear to realize or consume their facts. It is separate from the sanitized
[`SCOPE_BRIEF.md`](SCOPE_BRIEF.md). Nothing here establishes the
scientifically correct estimator, coefficient, threshold, response, covariance,
validity rule, or product representation.

## Recovery Strategy

The earlier MAP audits already performed detailed source tracing,
implementation comparison, repair, and re-audit. Stage A did not repeat those
activities. It reused the exact frozen inventory and inspected only the living
architecture, conventions, executable interface records, later decisions, and
topic-ref identities needed to determine scope and authority.

This is the anti-repetition rule in practice: existing scientific derivation is
reused, existing conformity evidence is classified and quarantined, and only
genuine scientific gaps are forwarded.

## Apparent Current Transformation

At a high level, the application appears to combine calibrated and conditioned
detector samples, upstream sample/detector eligibility, per-sample projected
position, a detector or analysis coefficient, map geometry, and a selected
mapmaking method to accumulate observation-map planes. Compatible observation
bundles may then be placed on a common coadd grid and accumulated before
downstream noise estimation, filtering, fitting, or feedback.

That apparent flow motivates the boundary. It does not prescribe the
scientific answer and is not copied into the author-facing brief.

## Reused Implementation-Scope Inventory

The historical MAP-001 inventory identifies these representative areas at the
audited application boundary:

| Area | Representative path class | Scope question exposed |
| --- | --- | --- |
| Ordinary map accumulation | `include/citlali/core/mapmaking/naive_mm.h` and shared map buffers | What are the admitted contribution, numerator, denominator, response, and support objects? |
| Shared map state | `include/citlali/core/mapmaking/map.h`, `src/citlali/core/mapmaking/map.cpp` | Which facts form one coherent map bundle, and which must remain distinct? |
| Map execution planning | `include/citlali/core/pipeline/mapmaking_execution_plan.h` | Which requested choices become effective and observation-resolved? |
| Observation coaddition | `include/citlali/core/pipeline/observation_coadd_accumulation.h` and output helpers | What compatibility and atomic-admission contract precedes numerical coaddition? |
| Product publication | map/coadd FITS and provenance helpers plus `validation/product_contracts.json` | Which identities, units, WCS, support, validity, response, and parentage must survive publication? |
| Configuration | `mapmaking.*` and `coadd.enabled` in the resolved leaf contract | Which settings identify method, geometry, grouping, unit request, support policy, and coadd intent? |
| Downstream consumers | noise, filter, source-fit, mode, Beammap, and fruit-loop paths | What must a complete MAP bundle provide without absorbing those estimators? |

Function bodies, current arithmetic details, tests, and repair diffs are not
reproduced here. The exact historical source traces remain in the audit
corpus.

## Current Apparent Inputs And Outputs

### Apparent inputs

- processed sample-by-detector signal and flags;
- detector, array, network, observation, group, and Stokes identities;
- detector/sample eligibility and non-finite policy;
- sample-to-map coordinate projection, WCS, pixel geometry, and map extent;
- an analysis/gridding coefficient with lifecycle and unit metadata;
- sample duration or another declared exposure measure;
- a sample-domain response/kernel tracer and its parent identity;
- mapmaking method/grouping/unit/support requests; and
- an ordered collection of complete observation-map bundles for coaddition.

### Apparent outputs

- observation and coadd signal maps;
- normalization/coefficient maps;
- kernel or response companions;
- hit, exposure, support, validity, and observation-count planes;
- formal or empirical uncertainty-related products with distinct identities;
- observation/coadd identities, WCS, units, grouping, method, and parentage;
- requested/effective/observation-resolved/realized provenance; and
- required-output and failure state.

These lists describe the apparent interface surface. They do not assert that
every current field is necessary, sufficient, correctly named, or
scientifically authorized.

## Ownership Boundary

| Owner/package | Supplies to or consumes from SCI-MAP | Must remain outside SCI-MAP authority |
| --- | --- | --- |
| SCI-CAL | Calibrated sample quantity, unit, response basis, conditional uncertainty transfer, validity/quality, and lineage | Calibration estimator, source/target association, atmosphere, passband, and unresolved CAL Q01--Q09 |
| ALIGN/AST | Common sample axis, eligible projected position, frame/WCS, pixel-coordinate meaning, and astrometric uncertainty | Alignment, pointing, interpolation, projection choice, and astrometric inference |
| PTC | Processed sample and coefficient/covariance identity with retained correlation limits | Cleaning, weighting estimator, covariance estimation, and precision proof |
| VAL | Sample/detector eligibility, flag precedence, non-finite policy, and raw admissibility | Scientific validity policy production and upstream failure classification |
| SCI-MAP | Ordinary map estimator, complete map-bundle transformation, response propagation, conditional covariance statement, and compatible observation coaddition | Upstream estimator meanings and downstream reinterpretation |
| NOI | Noise-realization generation, empirical covariance/weight calibration, and significance semantics | MAP may propagate a fixed realization through its operator but may not define its distribution |
| FLT | Map-domain filtering, local support, filtered response/covariance, and filtered validity | MAP preserves the immutable raw parent and does not certify filter output |
| SRC/MODE/BEAM | Source, pointing/OOF, and Beammap inference from admitted maps | Detection, fitting, photometry, OOF/Beammap parameters, and consumer-specific validity |
| FRUIT | Map-to-timestream feedback, iteration state, recurrence, and stopping | Iterative response and convergence are not one-pass mapmaking |
| MAP-002 / JINC | Signed deposition, subpixel JINC response, conditioning, coverage, and JINC-specific products | No positive-coefficient MAP-001 predicate is imputed to JINC |
| MAP-003 / OOF transfer | Residual transfer estimation and LMTOOF consumer boundary | Not sample-to-map gridding or ordinary observation coaddition |

## Implementation-Informed Risks To Sanitize

The author-facing scope must address these abstract risks without revealing a
current implementation answer:

- a normalization coefficient can be mislabeled as inverse variance;
- signal and kernel can use different membership, weights, normalization, or
  boundary handling;
- a finite output can be treated as valid despite failed identity or support;
- hits, exposure, support, validity, and observation count can be collapsed;
- coaddition can mutate state before later map-slot incompatibility is found;
- map slots can be joined by position rather than complete scientific
  identity;
- a narrowed or rounded WCS can be mistaken for a lossless admission
  authority;
- centered integer placement can be confused with signal centering or
  reprojection;
- invalid/non-finite contributions can contaminate accumulators when masking
  occurs after arithmetic;
- floating/count overflow or unrepresentable projected indices can leave
  partial state;
- empirical recalibration or filtering can rewrite the raw parent bundle;
- formal standardized signal can be mislabeled as empirical significance;
- a JINC signed coefficient can inherit an ordinary positive-coefficient
  support predicate; and
- a mode-specific OOF transfer product can be mistaken for a shared mapmaker
  response.

## Boundary Matrix

| Stage | Input authority | SCI-MAP transformation | Output/consumer boundary |
| --- | --- | --- | --- |
| Observation map | CAL + ALIGN/AST + PTC + VAL | Admit contributions, accumulate ordinary numerator/normalization/response companions, normalize on declared support, publish conditional covariance/status | Complete observation-map bundle |
| Observation coadd | Prior SCI-MAP observation bundles | Atomically admit exact compatible bundles, centered-integer place, accumulate with declared coefficients and shared membership, normalize | Complete coadd bundle |
| Noise realization | NOI-owned realization plus fixed MAP operator | Apply the same fixed map/coadd operator when the contract permits | Return mapped realization to NOI |
| Filter/input consumer | Immutable raw observation/coadd bundle | No filtering in SCI-MAP | FLT receives raw validity and parent identity |
| Fit/feedback consumers | Admitted raw or downstream bundle | No fit or recurrence in SCI-MAP | SRC/MODE/BEAM/FRUIT apply their own contracts |

## Sanitization Record

The Scope Brief retains only:

- the scientific boundary and package coherence;
- recovered owner-approved decisions;
- legitimate abstract inputs and required outputs;
- stable identities, frames, units, and lifecycle rules;
- producer/transformer/consumer responsibilities;
- conditionality on CAL, ALIGN/AST, PTC, VAL, and NOI; and
- genuinely open scientific questions and falsifiable predictions.

It excludes:

- source paths and current class/function names;
- audit findings, verdicts, and repair instructions;
- exact implementation, test, build, validation, campaign, and Unity results;
- current branch-integration and production status;
- file-layout requirements not already elevated by an owner decision; and
- JINC and OOF-transfer scientific content except the statement that each is a
  separate package.

No part of this dossier may enter the author packet. A genuine scientific
question found here may be transferred only in abstract, sanitized form.
