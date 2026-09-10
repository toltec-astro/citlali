# Historical control and S/N recovery

2026-09-10. Read-only manager evidence for the [reference record](REFERENCE_RECORD.md).
No artifact was executed and no scientific map, timestream, noise value or fit
was opened for this recovery. Exact paths, hashes and source revisions are in
[SOURCE_IDENTITIES.json](SOURCE_IDENTITIES.json).

## Two distinct identities

The required source anchor remains
`SCI-FRUIT-HISTORICAL-RECURRENCE@f70701ad`, commit
`f70701ad488444f3e2528c6bbe3e798863c9e301`, tree
`2009a1397bd67d615a1d6e9a8419e18fc794a81e`.
Source and configuration templates establish the historical algorithm, not a
reproducible executed control. An exact matching executable, dependency/compiler
environment, raw/APT/calibration identities, executed configuration and resulting
product tuple are still unavailable. Floating dependency references are not an
environment lock. No build or reconstruction of that environment was attempted.

The existing operational control is separately identified as
`SCI-FRUIT-C31-OPERATIONAL-CONTROL-POINT-152389-R0.1`:

| Binding | Recovered value |
| --- | --- |
| Source recorded for the operational artifact | `c31a60a0b74a7149d03d542966d6e35b77b8091c` |
| Recorded version | `v4.0.0-3753-gc31a60a0` |
| Preserved binary | `citlali-snapshots/9d3ce4f31260bbdf3f630dbaa09e0c18eb92b3ebcc8be15eb146024acd6f7f65/citlali` under the local POINT refactor project's `.tolproj` directory |
| Freshly verified size / SHA-256 | 12,320,912 bytes / `9d3ce4f31260bbdf3f630dbaa09e0c18eb92b3ebcc8be15eb146024acd6f7f65` |
| Metadata root | `/Users/gwilson/work_toltec/local_data/citlali-validation/v2/point/refactor/reduced/redu02` |
| Freshly verified metadata | `citlali_merged_config.yaml`, `config_source_manifest.yaml`, `runtime_provenance.yaml`, `citlali_o152389_0_2_c1.yaml`; all match the earlier recorded hashes |

This is a refactor-era artifact with existing empirical history. Its hash does
not prove equivalence to f70701ad. The earlier development evidence remains
scoped evidence; recovering its metadata does not select 152389 for another run.
An owner decision to use a same-build compatibility control would need to name
that different control and its narrower claims explicitly.

## Actual C31 settings recovered from metadata

These are historical settings, **not proposed-method defaults**:

| Area | Recorded configuration / provenance |
| --- | --- |
| FRUIT | Enabled; no seed path; max_iters=3; save_all=true; obsnum/raw scope; upper comparison; sig2noise=100; per-array flux limits [12,18,10]; legacy centering false; interpolation auto; no weight recomputation after addback |
| Other admission settings | Adaptive peak fraction 0 and local S/N floor 0, radius 18 arcsec / 1.5 FWHM. This records fields, not a claim that an adaptive branch actually admitted any sample |
| PTC | Standard PCA enabled, network grouping, eigenvalue cut 5 per array; adaptive/MP/null-model paths disabled. Historical `clean.tau=0.0` is not an adopted numerical rank tolerance under frozen SCI-PTC |
| Weights / learning | Validated weighting and weight validation enabled. Learning enabled with learns=2 and apply_start=2, including detector exclusions. This is neither uniform occurrence weighting nor the proposed fixed-population method |
| MAP | JINC, altaz, 1 arcsec pixels, coverage_cut=0.1, rmax=1.5, subpixel factor 1; JINC parameter triples [1.1,1.67,2], [1.1,2.17,2], [1.1,3.17,2] |
| Noise | Five noise maps configured, randomize_dets=false, noise products and empirical weights enabled, realization writing disabled. These settings supply no approved NOI law or independence claim |
| Runtime | Requested/effective OpenMP 6, Eigen 1, FFTW planning 1, OpenMP parallel policy, pointing reduction |
| Source order | One CLI source with precedence 0. The manifest says prior TolTECA ordered sources were not provided; full upstream authoring order is not recovered |

This metadata does not recover the complete dependency environment, random
state, raw/APT/calibration content identities or all actual realized decisions.
Unity-looking paths in provenance are historical strings; no Unity access took
place. Reduction products remain in place and were not copied or modified.

## What historical S/N actually means

The f70701ad `map_to_tod` code in the preserved manager source first samples or
projects the signal according to its interpolation route, then applies the
gate at a detector/sample occurrence. Bounds and current flags matter. Its
S/N denominator is an available positive finite per-array `MEDRMS` scalar
(with the code's map-index fallback), not MAP normalization or a per-pixel
inverse variance. An unavailable/bad scalar disables that S/N branch in the
historical code. That fallback is not accepted for the new required policy.

When enabled, the three sign conventions compare as follows. Here s is the
projected signal, R the scalar, t the S/N cut, and f the flux cut.

| Direction | S/N branch | Flux branch |
| --- | --- | --- |
| upper | s/R ≥ t | s ≥ f |
| lower | s/R ≤ t | s ≤ f |
| both | abs(s/R) ≥ abs(t) | abs(s) ≥ abs(f) |

An optional adaptive branch has its own threshold and support. Admission is
the **OR** of the enabled S/N, flux and adaptive branches. Adaptive support is
not a support gate on the entire union. The C31 upper/100/flux settings above
are configuration evidence; no code-equivalence result between C31 and
f70701ad is asserted here.

The f70701ad `MapBuffer::calc_median_rms` source computes, for each noise map
realization, the square root of the mean squared values over the weight-based
coverage region, then takes the median across realizations. It is a
zero-centered spatial RMS, not a mean-subtracted sample standard deviation.
An empty region gives zero in that code. Its meaning depends on the exact
coverage population, noise generator, normalization and realization law.
This recovery does not establish a probability, precision or unbiased noise
estimate for a new FRUIT route.

The ordinary-MAP proposal instead constructs a pixel decision before
containing-pixel projection of the accepted model. Even if the scalar-score
form is retained, this changes the decision domain. Sign, threshold/ties,
noise evidence, support and current-generation compatibility must be explicit.
The historical control keeps its actual flux/adaptive branches; the new
reference's accepted no-extra-flux-floor choice does not erase them. Together
with mapmaker, weighting and learning differences, these prevent attributing
an eventual comparison solely to a mapmaker or selector change.

## Disposition

Keep the required historical control unchanged and its missing executable tuple
unavailable. Keep the numerical S/N slot unavailable until its exact compatible
scale and interpretation are approved. Do not fill it with gamma, Q, a new
map-scatter estimator, legacy threshold 100 or an unaccepted NOI method.
No additional weighting or UID 4460 study is requested. No new data collection,
observation selection or numerical experiment follows from this recovery.
