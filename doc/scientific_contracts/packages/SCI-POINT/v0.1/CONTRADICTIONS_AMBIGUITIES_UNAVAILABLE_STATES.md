# SCI-POINT Contradictions, Ambiguities, And Unavailable States

Status: Stage A issue register; not a finding against implementation

## Recovered Conflicts

### Historical package grouping

`SCI-MODE-001` grouped Pointing and OOF. The owner now selects a narrow
SCI-POINT package and defers OOF science. The old grouping is superseded, while
its CAL/AST/OOF handoff evidence remains classified.

### Generic source fitting versus targeted Pointing

`SCI-SRC-001` covered detection/catalog work; current numerical code shares a
generic fitter. Neither fact makes POINT a blank-field source package. The
scientific intent, source selection, cardinality, and claims differ.

### `x_t`,`y_t` overloading

Current Pointing outputs may be interpreted differently when map axes change,
and historical AST evidence found that means and uncertainties crossed an
inverse-TAN boundary inconsistently. Base v0.1 therefore proposes AltAz
tangent-plane displacements only. Spherical-coordinate output remains
unavailable pending a separately complete method.

### Per-array results versus one telescope correction

Current Pointing output is per array, while recovered TolTECA behavior averages
table rows to form one correction input. The aggregate estimand, admission,
weighting, covariance, and partial-array failure policy belong to the named
pointing-support producer under owner-approved ODQ-001. SCI-POINT does not
ratify the recovered arithmetic mean. Owner-approved ODQ-002 also keeps sign,
telescope-offset composition, correction publication, and application outside
POINT.

### Raw/filtered labels versus scientific parent identity

Current implementation stages are labelled raw and filtered. Frozen MAP,
JINC, FLT-FIXED, and FLT-MATCHED products have distinct scientific meanings.
No label mapping is authorized by Stage A.

### FRUIT lineage versus parent type

The owner requires FRUIT to terminate in a map type. POINT therefore consumes
the exact terminal MAP, JINC, FLT-FIXED, or FLT-MATCHED type and separately
binds FRUIT terminal/generation ancestry. It does not invent a generic FRUIT
map type or accept an intermediate iteration.

### Formal uncertainty versus empirical uncertainty

Current output carries marginal formal errors; historical AST evidence and
frozen conventions require separation from coordinate/correction uncertainty.
NOI empirical uncertainty may later attach as a companion. Missing joint or
empirical covariance is not zero covariance and does not erase the fit result.

### Effective width versus beam

A Gaussian width fitted on a processed map depends on the map response and
source morphology. It cannot silently become an intrinsic telescope or
detector beam. SCI-BEAM retains Beammap authority. ODQ-008 nevertheless makes
the fitted centroid, amplitude, widths, angle, and fit state legitimate
telescope/observing-condition quality-control metrics. That QC role does not
remove their processed-map dependence or establish a unique cause for any
deviation.

### Named-use eligibility versus one universal flag

Fit completeness, displacement use, telescope/observing QC use, and
photometric-transfer amplitude use have different owners and facts. ODQ-009
therefore prohibits one universal good/bad status. VAL evaluates each exact
profile independently and does not author or compose the policies.

## Typed Unavailable States

| ID | Unavailable quantity or claim | Release condition |
| --- | --- | --- |
| `SCI-POINT-UNAV-001` | SCI-POINT-owned observation-level aggregate displacement | outside base v0.1 by approved ODQ-001; only a future successor owner decision could add it |
| `SCI-POINT-UNAV-002` | POINT-owned correction candidate | outside base v0.1 by approved ODQ-002; only a future successor owner decision could add it |
| `SCI-POINT-UNAV-003` | any numerical observation-local MAP/JINC/FLT parent route | exact predecessor authority, numerical product, required state, and POINT compatibility binding; family eligibility is approved by ODQ-003, and FRUIT when present remains lineage rather than a route type |
| `SCI-POINT-UNAV-004` | numerical `POINT-FIT/ELLIPTICAL-GAUSSIAN-COMPATIBILITY@1` route beyond Stage A method selection | complete Stage B binding of the approved model and requested/effective/realized search/support/constraint policy plus exact numerical parent binding |
| `SCI-POINT-UNAV-005` | POINT-owned aggregate or whole-observation success from partial array results | outside base v0.1 under ODQ-001 and ODQ-006; downstream producer owns any partial-set admission and publication policy |
| `SCI-POINT-UNAV-006` | full joint parameter covariance or uncertainty calibration/coverage | a separately authorized exact representation and evidence; ODQ-007 permits honest unavailability and preserves the fit result |
| `SCI-POINT-UNAV-007` | absolute flux or calibration accuracy | exact CAL/TolProj authorization and evidence; never POINT alone |
| `SCI-POINT-UNAV-008` | intrinsic beam inference from Pointing width | separate authorized method; SCI-BEAM boundary preserved |
| `SCI-POINT-UNAV-009` | statistical significance/detection probability | complete probabilistic method and validation, not legacy/formal ratio |
| `SCI-POINT-UNAV-010` | applied correction for another observation | exact producer selection and AST application authority |
| `SCI-POINT-UNAV-011` | OOF or blank-field source result | future separate package authority |
| `SCI-POINT-UNAV-012` | coadd or intermediate-FRUIT POINT parent | outside base v0.1 by owner ODQ-003A/003B direction |
| `SCI-POINT-UNAV-013` | universal cross-use POINT eligibility or base-v0.1 aggregate profile | prohibited by ODQ-009; each named-use owner defines its own policy and VAL only registers/evaluates |

Unavailable means the named claim is not established. It does not require
discarding the underlying map or per-array fit, and it does not prohibit later
versioned companion products.
