# SCI-POINT v0.1 Prior-Work Recovery

Status: Stage A recovery record; not scientific authority or an author input

Date: `2026-09-02`

## Recovery Question

What scientific, operational, and evidentiary work already exists for fitting
a bright, approximately central pointing source, and what genuine gaps remain
before an implementation-blind contract can be authored?

The recovery intentionally searched beyond filenames containing `pointing`.
It covered frozen predecessor contracts, historical audit inventory and
handoffs, current Citlali fitting and product surfaces, current TolTECA
pointing-correction resolution, current TolProj pointing/calibration workflow,
and historical validation/status records. Unity was not accessed.

## 1. Governing Authority

| Source | Recovered authority | Disposition |
| --- | --- | --- |
| Scientific Contract Library Program, pilot review, downstream roadmap | Stage A/Stage B process, anti-repetition rule, implementation firewall | adopt |
| Frozen SCI-MAP v0.1/r0.7.1 | immutable map parent meaning, WCS/support/response/covariance disclosure, Pointing arithmetic outside MAP | cite and abstract |
| Frozen SCI-AST v0.1/r0.3 and SCI-ALIGN v0.1/r0.3 | coordinate realization, pointing-support record boundary, frame and uncertainty separation | cite and abstract |
| Frozen SCI-CAL v0.1 | signal/unit/calibration meaning and downstream pointing-derived photometric lineage | cite and abstract |
| Frozen SCI-BEAM v0.1/r0.3 | per-detector Beammap fit ownership; relative centroid/PSF/covariance distinctions; no telescope-pointing claim | adopt boundary; reuse generic fitting concepts only |
| Frozen SCI-VAL v0.1/r0.3 plus successors | registry/evaluation mechanism; named-use owner authors policy | abstract |
| Frozen SCI-FLT-FIXED and SCI-FLT-MATCHED authorities | distinct transformed-map parent meanings and response/covariance state | cite conditionally; later ODQ-003 approves family eligibility but not numerical availability |
| SCI-FRUIT Stage A empirical-lane work | terminal/iteration and parent-lineage concepts; POINT remains downstream | provisional boundary evidence only |
| `doc/SCIENTIFIC_CONVENTIONS.md` | Point/OOF AltAz tangent offsets; fit-field meanings; requested/effective/realized separation | adopt where already governed |

## 2. Historical Scientific Inventory

### `SCI-SRC-001`

The historical inventory assigned generic map-domain detection, Gaussian
fitting, and source tables to `SCI-SRC-001`. Its incoming AST handoff concerns
inverse-TAN coordinate covariance for catalog-like source positions. It does
not audit the Pointing estimator and it expressly leaves Pointing/OOF and
Beammap fits separate.

Disposition: **defer** to later blank-field source work. It is the owner's
excluded concept 3 and is not a predecessor POINT contract.

### `SCI-MODE-001`

The historical inventory grouped Pointing and OOF map fitting, significance,
astrometric products, and shape products into one unlaunched package. No
approved scientific core or owner-approved Pointing/OOF contract was found.
Three incoming handoffs record CAL dependency, AST coordinate/covariance
separation, and OOF/LMTOOF lifecycle concerns.

Disposition: **supersede the grouping**, not the evidence. SCI-POINT is narrow;
OOF inference remains a later separate package. CAL and AST handoffs are
reconciled against their later frozen authorities. The OOF handoff is deferred
intact to SCI-OOF.

## 3. Mature Working Pointing Path

Current Citlali contains a mature Pointing path that:

- constructs observation-local array maps;
- fits one map per array with a six-parameter elliptical Gaussian;
- initializes from a weighted peak, optionally inside a central search region;
- uses a bounded fit region and configured amplitude/FWHM/angle limits;
- uses the map weight field in the fit;
- publishes per-array amplitude, centroid, major/minor width, angle, marginal
  formal errors, legacy amplitude/full-map-RMS dynamic range, and
  amplitude/formal-amplitude-error;
- supports raw-observation and implementation-labelled filtered-observation
  fit stages; and
- writes ECSV fit tables and embeds fit values in corresponding map products.

Current TolTECA reads one or two Pointing tables, averages `x_t` and `y_t`
across table rows, accounts for telescope user/paddle offsets, changes sign
from measured displacement to correction, and emits pointing-correction
records. Current TolProj constructs pointing reductions and consumes Pointing
amplitudes in its pointing-derived flux-scale workflow.

Disposition: this is the **working wheel** to preserve and scientifically
describe, not a source of automatic authority. Exact estimator mechanics,
aggregation, correction construction, failure behavior, and product semantics
remain quarantined until the owner adopts or changes them explicitly.

## 4. Historical Validation And Refactor Evidence

Recovered evidence includes:

- zero-difference compact-profile reproduction against a selected historical
  Pointing YAML;
- typed post-processing/source-fitting cutover with an unchanged numerical
  fitter;
- accepted product/table comparisons and current product-contract schemas;
- Pointing fit metric tests and manager-facing realized-state records; and
- earlier source-aware Pointing/OOF implementation notes.

Disposition: **evidence-only**. These records demonstrate that the wheel exists
and help bound later conformity work. They cannot select scientific estimator,
uncertainty, route-admission, or acceptance policy for Stage B.

## 5. Reusable Scientific Content

The following prior reasoning should be reused rather than rederived:

1. a fitted source centroid in map tangent coordinates is not automatically an
   absolute celestial coordinate or a telescope correction;
2. formal fit covariance, astrometric transformation/correction uncertainty,
   and empirical measurement uncertainty are distinct;
3. effective mapped source response is not automatically the intrinsic beam;
4. fit support, boundary, weights, source model, and parent response are part
   of estimator identity;
5. `sig2noise` is legacy amplitude/full-map-RMS dynamic range, not significance;
6. `fit_sig2noise` is a formal amplitude/error diagnostic and is not empirical
   detection probability under correlated noise; and
7. later selection or application of a pointing correction must preserve exact
   source measurement, parent, sign, basis, support, and producer identity.

## 6. Genuine Gaps

No recovered authority answers all of the following:

- how the Stage B contract states the already approved per-array terminal and
  measurement-only boundaries without importing downstream aggregation or
  correction policy.

ODQ-001 through ODQ-009 now close the package-ending, correction-ownership,
parent-family eligibility, compatibility-estimator, and configurable
search/support/constraint questions as well as per-array atomicity and partial
success, formal-uncertainty publication, and amplitude/effective-shape/QC
roles and named-use policy ownership. Exact numerical boundary binding is a
later availability gate rather than an open scientific family choice. No
bounded owner decision remains; only faithful Stage B authorship and any
author-exposed conflict justify new scientific discussion.

## Recovery Conclusion

There is no need to invent a new general source fitter. There is a mature,
bounded bright-source Pointing fit to formalize. The package must preserve that
scientific center of gravity while separating historical operational coupling
and declining claims that the existing products do not support.
