# OOF preparation and readiness — 2026-10-01

**Engineering preparation passes; the proposed matched comparison and joint LM inversion are not launch-ready.** No observational reductions, feedback applications or inversions were started. POINT remains assessed/closed and its frozen evidence is unchanged. The source candidate is a development candidate, not production activation or canonical integration.

## Validation and exact candidate

Implemented source **0e63e6459112c0ddaeb284694612fd7d09b3d637** on
`codex/rtc-point-transfer-20260928`, based on
`6d9ad5e5c74f0b08687baba2ebeb1f25822d8cf3`. This report/status closure changes
only documentation and evidence. The tested source and binary hashes stay
explicit; a future observational freeze must rebuild/review its exact candidate.
Local AppleClang/Homebrew build; GSL2.8 installed to compile the existing LMTOOF
reader fixture. No representative Spack/Unity run was attempted or claimed.

- Native build and CTest:1322 passed, one pre-existing disabled lifecycle test.
- Baseline tools:207 passed. POINT tools:214 passed. OOF fixtures:11 passed.
- Full required config preflight and native POINT contract:passed.
- Three real OOF metadata preparations and four negative identity/policy probes:passed;
  zero unexpected error-level messages in successful native preparations.
- Three historical reader roundtrips:passed. Unequal-support joint-LM sequence:
  correctly rejected, not labeled solver-ready.
- Initial POINT mock-fixture failure is retained with its diagnosis; no observation
  was attempted or retried. Independent review covers scientific/behavioral,
  architecture/ownership and repository/evidence separately. Final exact closure-SHA
  review is stored under the campaign's `review/` directory.

## Proposed first comparison

152385/0/1, 152386/0/1 and 152387/0/1, each kept separate; source1146+399; requested physical M2Z relative to middle −2/0/+2 mm. The middle observation is not asserted to be best focus. a1100 networks1–4, cold M0, zero feedback, no source-admission prerequisite. Proposed maximum is three R plus three S roles, not six authorized starts or a reuse of POINT's unused budget.

| Binding | R proposal | S proposal |
|---|---|---|
| Numerical owners | Retained historical Citlali treatment | Shared native RTC/CAL/PTC/AST/MAP |
| Population | Same networks1–4, pending canonical subset consumer | Networks1–4, all2069 raw channel identities |
| Telescope/APT/Tune | Same explicitly selected historical telescope and matched148670 APT-v2/Tune152384/0/1 | Same input identities; native admission recorded |
| PTC | Historical standard per-network PCA3 | Observed-entry ALS rank3, cap500, tolerance1e-5 |
| Filtering/segmentation | Historical16Hz/32terms, downsample2, preserved unforced segmentation | Existing a1100 F1/235arcsec/s filter, forced10s processing segments/context32, native clocks/gaps retained, full-stretch RTC diagnostic policy |
| Map | Historical1arcsec AltAz JINC, validated weights, coveragecut0.5 | Ordinary2arcsec FK5, existing uniform coefficient family, full native SCI_SUPPORT; coverage0.5 remains informational |
| Source assumptions | Historical30arcsec source-mask/20arcsec protection and Gaussian diagnostic are explicit reference differences | No POINT40arcsec/core/Gaussian/ring/threshold gate; no source protection mask |
| FRUIT | Disabled in the actual historical config | Disabled; execution returns after native M0 |

The recipes and proposed R YAMLs are in `inputs/`. R filenames deliberately say `PROPOSAL_NOT_RUNNABLE`. These are a declared operational comparison with controlled input population, not otherwise identical pipelines or an isolated PTC algorithm test. Full-array historical maps are contextual evidence only; they cannot replace matched four-network R products. No compact-source sensitivity metric or fitted-width improvement claim is implied.

## Engineering and calibration/support evidence

`EVIDENCE.json` binds exact-source gates, full native metadata receipts, dependency/generated-input hashes and saved-interface results. Native `terminal: prepare` reads metadata, timestamps, AST/TEL and canonical APT/Tune bindings; it returns before KIDs I/Q solving, RTC treatment, sample CAL, PTC or MAP. Positive acquisitions must have Oof/Lissajous headers. Foreign scan, duplicated channel, foreign APT binding and a POINT-only probe are rejected. A separate fixture rejects OOF feedback before environment validation or execution. No new numerical owner or Engine state was introduced.

All three use2069 identities: nw1=494, nw2=510, nw3=533, nw4=532. Each manifest carries the exact observation/scan and raw/KMP identities; selected output_uid/raw_source_uid/relation_uid, matched disposition, flxscale and focal-plane x_t/y_t are recorded per channel. flxscale is mJy/beam/xs and geometry is arcsec. There are1790 finite positive factors in each selected population,188 null flags and91 nonzero flags. Metadata admission is not realized calibrated support: matched disposition, factor availability, native validity and atmospheric support still govern CAL. R and S effective sample calibration/masks must be compared after execution; shared APT bytes do not prove equal admitted samples.

The configuration selects historical **recomputed** TEL files. Their sourceRA/sourceDec/parallactic angle differ from the originals; all other compared numeric fields match. Both versions and hashes are retained. Recompute provenance was not recovered from headers or the bounded code/log search. Selected parallactic angle changes by about0.34–0.36deg within each acquisition. Binding historical bytes makes the candidate reproducible, but does not establish their astrometric correctness. FK5/equinoxJ2000 source semantics must be justified for these derived streams before interpreting new S maps.

Preserve each acquisition's full native support and unsupported regions, and report common support separately; never define a source by a required core or morphology. Historical saved maps already have finite-positive weights on only89440/89377/88880 pixels (the coverage mask selects exactly the same population). The adapter cannot recover unsupported parent pixels. Saved weights/OOF_RMS are not an empirical independent covariance product.

## Existing LMTOOF interface findings

The checked adapter performs lossless axis reversal/integer embedding into512×512 squares with positive1arcsec axes; it preserves every declared supported pixel, sign and amplitude and pads only withNaN. It neither interpolates nor recenters, subtracts a background, rescales a peak or applies pointing feedback. Actual C++ DataGrid roundtrips match every value and NaN; axis zero/spacing and focus/wavelength headers agree. FK5, incompatible units/scales, half-pixel centers and skew are rejected. The LMTOOF source snapshot is hash-bound; the local directory has no Git metadata. Only a reader/focus fixture was built, using the existing sources; no fit was run or sibling source changed.

- **Frame:** The historical AltAz x is cross-elevation tangent offset (cosine-elevation projection plus pointing corrections), y is elevation offset. The reader consumes those physical signed axes, not arbitrary image orientation. Existing S MAP publishes FK5. A single image rotation cannot exactly undo time-varying parallactic rotation; use a native AST-owned sample-level AltAz attachment if joint inversion becomes part of the comparison. Header relabeling is rejected.
- **Wavelength:** Saved OOF_W is the nominal0.0011m, not a measured bandpass. Actual `glm_fit_data_files` selects instrument1100/filter1: seven frequencies233–323GHz and stored power weights, reference276.39GHz. Passing0.0011m on its command line does not make that path monochromatic. RTC's filter-design frequency is a different operand.
- **Focus and sign:** Preserve requested M2Z values−3566.596717269,−1566.596717269,+433.403282731 microns, without best-focus subtraction or sign flip. The existing Subref returns−(cos_primary+cos_secondary), and FFTGrid uses−2π path/(10^6λ). The reader/derivative fixtures verify the implemented convention, not independent physical optical-sign truth. The LM route zeros M2X/Y in its model call and enables per-map tilt fits; saved M2X/Y are zero for these maps. No pointing correction is applied by this adapter.
- **Normalization/model:** Preserve mJy/beam values and relative brightness across focus. LM's common gain multiplies voltage; modeled power scales as gain². Do not report it as a direct linear intensity factor. The C++ model reconstructs DishParams from instrumentID and ignores FITS pupil/taper overrides: radius22.0m stop, inner1.65m, outer25m, taper3dB. The Python path advertises22.5m stop (writer inner1.625m); historical files omit OOF_RS. The22.0m adapter operand is explicitly an interface-test value, not an adopted optical model.
- **Support defect:** Existing OOFLM allocates `nmaps * first_valid_count` and writes each map at that fixed stride. Unequal counts here leave623 uninitialized residual entries; other orderings can overrun. New sequence admission rejects this, retaining the original support. Correct variable-length offsets in the existing solver require a bounded LMTOOF work order. Cropping all maps to equal counts is not an engineering repair.
- **Uncertainty:** The actual LM executable sets all residual weights to1; nominal_sigma is unused. The generic OOF scalar-sigma chi-square path is different. No map precision/covariance is transferred, and these formal fit errors cannot establish scientific uncertainty or sensitivity.

## Readiness boundaries and owner decisions

Engineering prerequisites still open: an explicit canonical population view in the retained R consumer/executor; a sample-level AltAz saved attachment if solver comparison is included; and the existing LMTOOF residual indexing repair. The exact historical binary is Linux sourcec31a60a0, SHA9d3ce4f31260bbdf3f630dbaa09e0c18eb92b3ebcc8be15eb146024acd6f7f65. Local retained source0f52e1a42 binary SHAfeed416327b1f6a97b22d336c78ebdade13f498cde4e9c9f419a18e7665030e1 is a different baseline. Both source versions require complete canonical APT network coverage. Removing seven raw paths or silently emitting a legacy ECSV would violate the current contract. No legacy fallback was restored in the successor.

The genuine scientific choices are whether the first milestone remains the handoff's FK5 M0/support characterization with inversion deferred, or adds a native AltAz projection for immediate model comparison; authority/provenance for the recomputed telescope coordinates; and, before fitting, the exact bandpass/pupil model, focus/tilt freedoms, response/normalization and uncertainty interpretation. Recommend retaining the stated M0 boundary experiment and postponing optical inference until those solver prerequisites are separately validated. Distributed-signal admission/uncertainty and an endpoint/work budget are required only before future OOF feedback; they are not invented as gates on cold M0.

LMTOOF sibling modification is not included in this Citlali preparation work order: the applied [TolTEC skill](/Users/gwilson/.codex/plugins/cache/personal/toltec-codex/0.1.0+codex.20260730133632/skills/toltec-context/SKILL.md) explicitly says, “Never infer permission to edit a sibling repository.” This is a distinct solver repair scope, not a request for approval of routine Citlali engineering.

## Storage reservation

Recounted prior owned campaign evidence: **20,072,023,234 bytes (20.072GB)**.
Selected native cells:16,264,393 /16,315,092 /16,280,961. With the existing conservative
304 bytes/cell plus2.7GB retained-role allowance, active S reservation is at most
**7,659,787,968 bytes (7.660GB)**. Serial maximum-six-role schedule
R152385,S152385,R152386,S152386,R152387,S152387 reserves five completed roles at2.7GB,
plus the active last S and1.1GB report/metadata, reaching
**42,331,811,202 bytes (42.332GB)** including prior evidence.
That is **12,331,811,202 bytes (12.332GB)** above the30GB ceiling.
R active/retained cost must still be bound, so this is not a final launch reservation.
Current free disk at preparation was612,923,097,088 bytes; keep a separate2GB
free reserve. No download required. No evidence, raw, Tune, APT or historical product
was deleted. New preparation metadata is covered by the report reserve; active/released
scratch must be recounted at the eventual freeze. These are conservative storage
reservations, not observed reduction RSS or runtime predictions.

## Launch and manual resume

There is **no valid observational launch command yet**. The saved `preflight.py` is metadata/fixture-only and is not a launch command. Before launch, finish the applicable prerequisites, select an exact R executable, freeze exact candidate/binary/config/input/plan/support bindings, resolve resource admission, and obtain the final exact-SHA review. Do not replace the comparison with an unmatched historical map or launch S alone under a label claiming matched R/S.

The future launch procedure is one owner-run foreground batch with explicit thread caps (two network workers, one numeric thread), serial focus states/roles, immutable output roots and an attempt ledger. Stop on the first failed role; no automatic retry, replacement, feedback, inversion or resume. After it returns, the owner sends **“done”** here; Codex then reads receipts/logs/support/calibration/storage, assesses saved products and reports. A failed attempt remains charged and preserved; repair then a separately frozen manual-resume command is required. No background watcher or scheduled continuation was created.
