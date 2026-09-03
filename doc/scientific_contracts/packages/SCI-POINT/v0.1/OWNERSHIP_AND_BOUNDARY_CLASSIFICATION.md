# SCI-POINT Ownership And Boundary Classification

Status: proposed Stage A boundary; owner approval required

| Scientific fact or action | Proposed owner | POINT relationship |
| --- | --- | --- |
| Ordinary positive-coefficient map | SCI-MAP | owner-approved eligible immutable route; exact binding still required |
| Signed-coefficient JINC map | SCI-JINC | owner-approved eligible distinct immutable route |
| Fixed-convolution transformed map | SCI-FLT-FIXED | owner-approved eligible distinct route; no silent equivalence |
| Matched-template amplitude map | SCI-FLT-MATCHED | owner-approved eligible distinct route with different unit/response meaning |
| Terminal product created through FRUIT | exact MAP/JINC/FLT product owner plus SCI-FRUIT lineage authority | consume under terminal map type with complete FRUIT ancestry; not a separate POINT parent family |
| Empirical map uncertainty | SCI-NOI | optional companion, never inferred from fit weights alone |
| Parent WCS, support, validity, response, covariance state | parent producer | consumed unchanged |
| Known-source identity and expected position | observation plan / source authority | required external input |
| Per-array bright-source fit method | SCI-POINT | owned |
| Per-array fitted displacement, amplitude, effective shape, formal uncertainty, fit state | SCI-POINT | owned outputs; centroid is the pointing measurement and fitted parameters/state are authorized telescope/observing-condition QC metrics under ODQ-008 |
| Observation-level cross-array aggregate | pointing-support producer under approved ODQ-001 | external downstream product; not SCI-POINT v0.1 |
| Measurement-to-correction sign and telescope-offset composition | pointing-support producer under approved ODQ-002 | external downstream responsibility; not POINT-owned |
| Selection of pre/post pointing observations | pointing-support producer | downstream external responsibility |
| Interpolation and application within native support | SCI-AST under selected producer record | downstream external responsibility |
| Absolute photometric reference and pointing-derived flux-scale transfer | SCI-CAL / TolProj | downstream use of honest POINT amplitude |
| Per-detector Beammap fit, PSF, sensitivity, APT | SCI-BEAM | explicitly excluded |
| OOF focus/surface inference and observation association | future SCI-OOF | explicitly excluded |
| Blank-field detection, deblending, catalog fitting | future source package | explicitly deferred |
| Per-array POINT fit-result completeness policy | SCI-POINT | authored by POINT; VAL only registers/evaluates |
| Displacement admission for correction construction | pointing-support producer | consumer-owned named-use policy; VAL only registers/evaluates |
| Fitted-parameter telescope/observing QC admission and action | named QC process | consumer-owned named-use policy; VAL only registers/evaluates |
| Amplitude admission for photometric transfer | SCI-CAL / TolProj | consumer-owned named-use policy; VAL only registers/evaluates |
| Implementation conformity and validation | later separately governed work | not Stage A/Stage B scientific authority |

## Correction Boundary

The proposed architecture is:

```text
exact map parent
    -> POINT per-array source measurement
    -> pointing-support producer aggregates when required and selects/composes correction record
    -> AST applies producer-selected record within its exact support
```

A correction consumer may use a POINT measurement only when it declares the
exact aggregation, sign, basis, user/paddle-offset treatment, selection,
support, covariance, parent, and version. It may not rewrite the POINT product
or imply that POINT applied the correction.

The same immutable POINT result may have different outcomes under the four
named-use policies above. No policy owner or VAL may promote one use's outcome
to universal eligibility.

## Shared Numerical Code

The current Pointing and Beammap paths share numerical fitting machinery.
That is an engineering reuse fact, not joint scientific ownership. POINT fits
one known source per array-map product. BEAM fits detector maps to infer
per-detector effective PSF/calibration/sensitivity/APT quantities. The unit,
grouping, parent, response, and claims differ.
