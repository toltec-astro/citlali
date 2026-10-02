# a2000 OOF M0 saved-product assessment — 2026-10-02

**Compatible historical a2000 maps were reused. Three S maps completed, retaining the focus-dependent source structure and full central32″ support. Improvement for operational OOF analysis is not established.** Native S has more small-scale field variation and deeper negative excursions, is not yet an AltAz consumer input, and has a MAP acquisition-identity publication defect. This bounded assessment stops here; no extra reduction, feedback or optical inference is authorized by unused R slots. POINT remains closed.

## Completed batch and useful structure

All three S roles completed from exact source `f79db1faa8a85c6d84bc0dd387a18b8924b956e3`, with three charged starts, zero R starts, zero retries and zero feedback. Successful logs contain no unexpected error-level messages. All36 ALS segment fits converged in2–47 iterations, below500, at rank3/tolerance1e-5. This is local supplemental AppleClang/Homebrew evidence, not new Unity/Spack or production acceptance.

| Observation / relative focus | R peak in operational32″ window | S peak in native32″ window | S CAL available/scheduled | Runtime | Sampled peakRSS |
|---|---:|---:|---:|---:|---:|
|152385 /−2mm|127.00|148.11|70.23%|34.59s|2.046GB|
|152386 /0mm|559.85|564.85|72.89%|39.29s|1.572GB|
|152387 /+2mm|214.70|245.77|73.57%|32.84s|2.061GB|

Peaks are **pixel values in mJy/nominal beam, not fitted amplitudes or flux recovery**. The middle acquisition is brightest and most compact; both offset-focus maps are broader and asymmetric. This ordering is comparable to R. Higher peaks do not establish better OOF data. The separate AltAz/FK5 frames, pixel scales, gridding, weighting, source treatment and segmentation prevent an isolated ALS-versus-PCA claim or exact geometric/pointing comparison.

## Operational windows and contamination

Historical operational preprocessing remains frozen and source/hash checked: LMTOOF yields600 supported samples from each625-pixel selected window; ML yields1024 supported samples from each32² window, with independent peak normalization. Existing default/example configuration is explicit; the exact prior152385 consumer invocation was not recovered. No finite-but-unsupported observations occur in those R windows. The general LM unequal-count defect is not encountered by these R inputs, but remains unrepaired; the writer/reader2″ half-cell discrepancy remains.

S's existing AltAz admission check rejects its FK5 maps as expected. Therefore no S optical/ML output, post-LM valid-count certification, same-sky common-support subtraction, or aligned correlation is claimed. The S diagnostic windows use existing map-center Cutout2D indexing, 16² at2″ for32″ and50² for100″; no interpolation, rotation, Gaussian source recentering, background subtraction or fitted amplitude scaling is introduced. Exact pixel ranges/crop slices are recorded: center rounding differs by half a pixel between odd/even grids. These are unshifted FK5 diagnostics, **not a substitute for LM's source-centered AltAz window**. The existing data-dependent LM centering rule is unchanged for R and unexecuted for incompatible S inputs.

S32″ support is256/256 in every focus state. S100″ support is2500/2500,2500/2500,2495/2500. The five missing152387 pixels lie near square corners: (−50,38),(−32,42),(−46,42),(−40,46),(−46,50)arcsec; three are internal to the rectangular cutout. None intersects the central32″ window or the20–50″ radial annulus. Parent supports remain10752/11013/10205 native2″ pixels; R's98806/99165/97276 native1″ supported pixels are recorded separately. Differences in outer coverage alone are not a veto.

| Observation | R native100″ minimum /1st percentile | S native100″ minimum /1st percentile | R→S20–50″ annular median |
|---|---:|---:|---:|
|152385|−20.92 /−12.23|−90.97 /−45.13|−1.82→−2.61|
|152386|−29.45 /−21.23|−78.53 /−51.08|−5.73→−6.63|
|152387|−16.91 /−13.89|−88.92 /−44.56|+1.57→+2.57|

Values are mJy/nominal beam in separate native100″ map-center windows; R here is full-resolution parent data, not LM's interpolated4″ grid. S shows visibly grainier positive and negative structure. Broad bowls in the middle-focus map persist in both routes; deeper S negative tails occur within the relevant angular span, not just unused map edges. In152385 an outside-window positive feature reaches414.53mJy/beam at native(86,−70)arcsec; retain it as context, not a replacement central peak or automatic veto. Native negative-tail/texture quality is **degraded as a delivered-map descriptive metric**; different smoothing and support prevent treating this as a sensitivity loss or causal PTC verdict. No source-free region is certified, so annular statistics measure field structure/contamination, not independent noise, covariance or sensitivity gain.

## Calibration, support and identity

All977 raw identities per acquisition remain bound to original inputs, Tune152384/0/1 and canonical matched148670 APT-v2. R saved stats and S CAL match **all853 finite positive flxscales exactly**, ratio1.0 detector by detector. This excludes an APT scalar mismatch, not all effective calibration differences. S uses flxscale once and target atmosphere once, no parent/rescale reapplication, tau225=.018 from one observation-associated reading, and per-sample elevation. Within-observation opacity variability is unmeasured. R was extinction-enabled, but exact R per-sample factors/support cannot be reconstructed from the saved maps alone; no claim of identical effective sample calibration is made.

S CAL availability is2,696,463/3,839,610;2,807,927/3,852,311;2,828,439/3,844,495 post-F2 detector samples. Detectors with usable CAL samples are777,826,822. All CAL-available entries survive PTC into MAP; MAP admits those same counts, with zero invalid AST coordinates. The denominator is scheduled post-downsampling slots, not raw cells or sky-footprint retention. Causes, detector factors and learned fit diagnostics are retained. Native normalization equals contributing hits (dimensionless uniform coefficients); it is not precision, and cannot be equated with R weights or coverage seconds. Original ALIGN-grid footprint retention remains unavailable.

**MAP publication defect:** `map_saved_execution.h` at executed source lines181 and266 constructs `obs:<number>_0_2`. This propagates into the PTC-to-MAP identity, plan and bundle. The request, canonical APT admission, CAL/AST offset identity and MAP telescope bindings all select scan1; the map loader explicitly calls the OOF scan1 path at lines122–128. Thus the evidence supports actual scan1 data processing with an incorrect constructed identity, not a scan2 raw-data reduction. Because this affects typed provenance it is a conformance failure, not merely a cosmetic label. Original products/receipts are unchanged. No numerical rerun was used to assert invariance; repair and any re-publication need a separately bounded continuation, with no automatic new S start. The preflight and launch gates missed this emitted-string invariant.

## Cost, evidence and disposition

Numerical wall times total106.72s; peak sampled aggregate process-treeRSS2.061GB. The roles retain0.542/0.538/0.531GB (1.610GB total); shared derived storage at assessment is21.710GB, below30GB. The runner already released439,213,552bytes of recorded conditioned/centered duplicates after successful completion. No additional cleanup was performed. These actual disk counts are distinct from the29.330GB conservative launch reservation and RAM usage.

Figures and machine measurements are in
`~/work_toltec/local_data/citlali-validation/successor-oof-a2000-m0-20261001/report/`:
`central-100arcsec.png`, `normalized-32arcsec.png`, `full-context.png`, and `MEASUREMENTS.json`.
Every comparison uses common display scales and labeled native axes, without rotation or paired R−S subtraction. Raw, unnormalized parent maps and operational windows remain hash-bound. Reproduce the saved analysis only with `assess_saved.py <fresh-directory-under-campaign-root>`; it cannot reduce observations. The original owner launch is consumed and must not be reused.

Overall: **not established** for improved operational OOF input. Broad focus structure and central support are preserved; delivered native texture/negative tails are worse, exact cross-frame consumer assessment remains unavailable, and provenance conformance fails. No optical solution, uncertainty, transfer-function or absolute flux claim follows. Scientific/behavioral disposition: conditional saved-map characterization, failed MAP identity publication. Architectural disposition: unchanged shared native ownership; reporting only. Repository/evidence disposition: original numerical source, frozen inputs and receipts preserved; report/status changes only, no push, merge, production activation or sibling edit.
