# Minimal connected POINT–SCIENCE–POINT implementation plan

30 October 2026. **Proposed implementation scope; no reductions or source changes.**

Deliver one usable connected pilot through TolProj and operational TolTECA v2,
then extend it to the already selected two-observation coadd. Keep wing recovery
and temporal-change research paused. Optimize neither the old reference match nor
an isolated injection score; establish correct products, calibration transfer and
usable SCIENCE results with their measured limitations.

## 1. Fixed target and combined starting point

Local project: `/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`.

| Increment | Observations, all a1100/a1400/a2000 |
|---|---|
| First connected sequence | POINT 152389 → SCIENCE 152390 → POINT 152391 |
| Complete selected pilot | Extend through SCIENCE 152392 → POINT 152393; coadd 152390 + 152392, NGC4449 |
| Independent POINT check | 152418; never substitute it for the actual SCIENCE brackets |

All are subobs 0/scan 2. Local input links resolve, matched APTs and telescope
metadata exist, and the archived reference contains individual maps and a real
coadd. No Unity collection is needed to begin implementation. Successor POINT
152391/152393 and successor SCIENCE/coadds have not been produced.

Pin development to combined closure **b090da0458271c618568802fde1cf5c39f5dd5d2**,
whose application source is merge **c83375dc030095821999a0e0ff435f13bf82b914**.
Its parents are pointing 797d013f115e403a30663eb92254082e8e742826 and
engineering 5c46201057d6b6d1048aac2949a64e9a34e9bad3. Closure changes only records.
Do not implement on the older pointing-only code and recombine the same work.
The latest shoulder audit is evidence only; it need not change that source base.

The combined receipt reports 1,388 enabled native tests, 207 baseline checks,
300 POINT Python tests and complete config preflight passing. One inherited
native test remains disabled. This is AppleClang/Homebrew supplemental evidence.
Supported Mac Spack qualification remains pending (44 dependency installations),
as do hosted CI, Unity qualification and mainline acceptance. The engineering
lane owns those issues; this plan neither repeats cleanup nor waives its gates.
Mainline remains e32c4584dec07617d6caa857d39bf13047b6c4b0.

After implementation approval, use one owned application increment based on this
exact combined version, or an explicitly verified accepted descendant. Do not
edit the other active worktree, move mainline, or create another numerical probe.
An exact content review must accompany any changed base. This planning turn
creates no branch and no implementation worktree.

## 2. Reuse first; no parallel pipeline

| Responsibility | Keep its existing owner and machinery |
|---|---|
| Project/cohort selection, directories, validation, guarded submission | TolProj; ordinary project-relative pointing discovery already works |
| Numbered YAML merge, input preparation, telescope pointing-correction construction, launch | Operational TolTECA v2; no TolTECA source change proposed |
| Final low-level config validation, native execution and required output failures | Citlali; use the combined input services and resource/publication owners |
| POINT fit/measurement and its eligibility | Existing scientific POINT owner; one-way compatibility publication only |
| Reference flux lookup and bracket correction estimation | Existing TolProj calibration step |
| Matched/calibrated compact-v2 APT verification and child issuance | Existing Citlali APT guardian/issuer; no mutable APT copy or alternate library |
| Sky association, map formation and coadd arithmetic | AST/MAP typed owners; no scientific calculations in orchestration or Engine |

TolProj source inspected: 6309a5a852a0412cee867992aff01341a56b0ebf.
TolTECA v2 source inspected: 2791e6a1e6349ad1d3ac549a648f41cbc51b98c7
(local `origin/main`, not the checked-out v3 development tree).
TolProj modifications below are a separately scoped part of future implementation
approval; sibling repositories remain read-only during this planning task.

## 3. Four bounded implementation batches

### Batch 1 — Make the POINT handoff truthful and consumable

**Preserve the working OG corrections.** The subsequent
[OG compatibility audit](../validation/pointing_og_compatibility_2026-10-30/REPORT.md)
verifies tangent-az/elevation axes, arcsecond units, signs, User/Paddle composition,
single downstream negation and saved-FITS orientation. These are recovered existing
behavior, not open scientific-design choices. The actual TolTECA resolver exactly
reproduces the saved SCIENCE bracket corrections. Add the demonstrated single-PPT
positive-date rejection, legacy observation-span inputs and fractional-second epoch
conversion to the bounded compatibility work. The current new CAL conversion differs
from OG integer-second truncation by less than0.001″ for these saved brackets, but
needs explicit same-time tests/disposition rather than an assertion of exact parity.
Do not redesign TolTECA or apply a second cosine/sign/correction. Complete the
PPT→TolTECA→native correction fixture, including previously applied POINT offsets.

**Result:** a qualified three-array POINT table is discoverable in the ordinary
project; an unqualified result cannot silently enter calibration or astrometry.

- Extend `tools/timestream_successor/configured_point_products.py`, input metadata
  carriage in `configured_point_inputs.py`, and the existing CLI publication call.
  Keep fitting/error calculations with their scientific owner. Emit exactly one
  `ppt_*.ecsv` under `pointings/reduced/<reduction>/<obs>/raw` only when the requested
  complete product qualifies. Preserve diagnostic JSON/maps when it does not.
- Carry exact integer array and observation IDs, date/MJD, source, units, fit/map
  and method identity, named-use eligibility, and telescope User/Paddle offsets.
  Preserve measured nominal-beam amplitude meaning. Convert sigma to FWHM only
  for fields whose existing consumer requires FWHM.
- Bind the measured centroid to the effective POINT astrometry, including any
  already applied correction. Do not blindly copy it into `x_t,y_t`. Fixture-test
  the existing TolTECA composition of POINT/SCIENCE User/Paddle offsets and one
  sign reversal. POINT measures displacement; TolTECA constructs the correction.
- In TolProj `steps/calibrate_flxscale.py` and
  `steps/setup_science_reductions.py`, reject false/missing successor eligibility,
  fractional or duplicate array IDs, incompatible amplitude/error units, wrong
  observation bindings and unavailable required measurements. Apply protection
  to both calibration and pointing-calobj setup. Recognize genuine legacy PPTs
  explicitly; absence of successor metadata is not an automatic legacy exemption.
- Retain `--pointing-reduction`, existing reference-flux cache, factor calculation,
  compact-v2 issuance and setup. No new path option or calibration formula.

**Scientific boundary:** current `sigma_A` is a frozen-M0 conditional reference
uncertainty for diagnostic source acceptance. It is not the final adaptive-map
`amp_err`. Reuse the existing Gaussian-plus-constant fitter and guard; obtain a
bounded calibration-use decision before publishing eligible results. Section 5
states the unresolved decision without commissioning a new uncertainty study.
Metadata plumbing and rejection tests can proceed while it is resolved.

**Done locally:** complete/partial-array fixtures, all eight prior consumer probes,
unit conversions, unavailable cases, nonzero signed astrometry and offset fixtures,
legacy compatibility, atomic publication and required-I/O failure tests pass.
No fixture may turn an unavailable error into zero or into the old `sigma_A` label.
The restricted pilot option in Section 5 requires explicit named-use handling;
it is not an exemption inferred from a missing value.

### Batch 2 — Connect configured celestial SCIENCE and a real raw coadd

**Result:** both selected SCIENCE observations can traverse the existing native
stages with their own calibration/pointing, form celestial maps on one declared
2″ grid, and contribute to one correctly normalized coadd.

- Add a small explicit SCIENCE configuration/execution path alongside configured
  POINT. Reuse common input binding, RTC/CAL/PTC execution, weighted map arithmetic,
  installed-runtime machinery and output owners. Share only demonstrated common
  code. Do not simply remove the POINT type check: the current adapter also fixes
  AltAz coordinates, disables coadds/noise products and embeds POINT selection.
- Preserve requested/effective/realized config distinctions. The authored SCIENCE
  YAML requests RA/Dec JINC, legacy cleaning/FRUIT and noise/filter products; it is
  not a ready successor recipe. Produce an explicit field-by-field disposition
  for the selected successor profile. Unsupported required settings fail preflight,
  rather than silently becoming POINT defaults or legacy fallback execution.
- Bind each SCIENCE observation's exact APT and prepared pointing correction from
  the final TolTECA low-level YAML. Exercise single and bracketed MJD support and
  ensure telescope corrections are applied once. AST owns sample-to-sky positions.
- Give the two observations compatible celestial frame/reference/WCS/grid facts
  before mapping. Preserve independent support, membership, units and observation
  IDs. Reuse `AstRtcCoordinates::realize_v2`, `evaluate_ast_rtc_schedule(..., fk5_tan)`
  and `AstRtcTanGrid::centered` with `ForwardMapOperator`; the current saved-map
  and weighted POINT execution assumes AltAz and needs explicit frame carriage.
  Reuse existing coordinate and map owners; do not introduce a Python-only
  scientific mapmaker or interpolate finished maps merely to make grids fit.
- Preserve inverse-noise detector weighting. Separately bind a SCIENCE-appropriate
  source/training-support policy: POINT's source-free radii and required bright
  central-source fit are not general faint/extended-SCIENCE rules. Do not require
  POINT-EXPECTED success for NGC4449, or infer that all SCIENCE data are source-free.
- Implement the existing frozen ordinary MAP equal-observation coadd rule on
  complete compatible logical bundles with exact integer grid embedding and
  pixelwise admitted-observation counts. Detector noise weighting remains in each
  observation map. Do not copy legacy `weight_I × signal` coadd arithmetic or label
  the empirical normalization as precision. Bind the actual coefficient family,
  response/covariance availability and named-use admission; generic contract
  acceptance alone does not qualify an individual weighted bundle.

**Done locally:** synthetic shared-sky/pointing-sign tests; unequal detector weights
with equal-observation coadd tests; one/missing/duplicate observations; differing
support, all-invalid pixels, inconsistent units/frames/calibration, fractional-grid
misalignment, overflow and failed-output tests. Sequential streamed accumulation
must match the declared operator and preserve all observation contributions.

An explicitly selected **raw-map connection check** can be an intermediate
engineering milestone. It is not completion of the current configured SCIENCE
product set. Its effective config must explicitly disable unavailable requests;
any later owner-run use must identify that narrower purpose and its limitations.

### Batch 3 — Complete the configured products and calibration selection

**Result:** the normal project can use the explicitly selected pilot products
without misleading units, noise meaning or accidental reference-calibration reuse.

**Recommended minimum:** an explicit successor base-map pilot profile: calibrated
celestial raw maps, valid support/normalization and a genuine raw coadd. Defer noise
realizations and filtering for this connected pilot only, subject to Grant's
explicit product-scope decision. Keep the authored richer recipe preserved and
its full downstream-release requirements open. This avoids making general noise
and filter conformance an unexamined prerequisite to the first connected sequence.

- For any products retained as required, reuse existing SCI-NOI and map-filter owners where their input contracts are
  satisfied; add bounded adapters at the actual missing boundaries. Do not
  synthesize uncertainty from the detector normalization or promote a nominal
  Gaussian to a measured processing response. No noise/filter algorithm redesign.
- Qualify the requested noise realizations, empirical summaries, kernel/response
  inputs and Wiener-filtered coadds, or obtain an explicit product-scope change
  before calling a reduced set usable for its named base-map consumers. Source finding
  and retained detector timestreams remain disabled as authored.
- Retain immutable raw products and parents while publishing filtered products.
  Use existing reader conventions where semantics match; required reader semantics
  that cannot be satisfied are blockers, not occasions to fabricate aliases.
- Keep A, B and C separate. In B, both archived POINT correction sources and archived
  calibrated APTs are intentional. In C, both originate from the qualified
  successor POINT set. Record selected files and digests per SCIENCE observation.
- Preserve the setup distinction: `project.yaml` currently has `flxscale_done:false`,
  while authored B YAML explicitly selects archived calibrated APTs. Resolve the
  intended selection through existing explicit setup options/state in isolated
  case configurations; inspect the final merge, not just an authored file. Never
  mutate the reference or let setup silently replace calibrated children by matched
  parents. Confirm each child preserves detector correspondence and changes only
  the intended finite `flxscale` cells by the recorded per-array factor.

- Test actual TolTECA v2 observation discovery and a concrete coadd-reader/index
  path. Its observation reader discovers numbered observation directories and
  raw per-array FITS, not `coadded/`; filename compatibility alone is insufficient.
  Publish an explicit coadd locator consumed by the intended local analysis without
  editing TolTECA or impersonating an observation. Preserve unavailable exposure
  and covariance states; do not fabricate detector-seconds from downsample counts.

**Done locally:** actual downstream reader fixtures accept the promised product
layout and units; ineligible/partial products fail; B/C manifests cannot cross-select
parents; calibration request/report/child bindings reproduce; required products
are planned before execution and failures propagate to the shell.

### Batch 4 — Freeze one candidate and prepare the owner's connected run

**Result:** exact, copyable shell instructions and a short evidence checklist,
prepared only after the implementation and applicable scientific decisions pass.

First close the single bracketed sequence, then the second SCIENCE observation and
coadd. Use existing saved POINT results only if their runtime/method/config and
publication qualification match the final candidate; otherwise prepare the needed
consistent bracket set explicitly. Do not silently combine old and new methods.
152418 is the independent POINT check. Hold SCIENCE 152392 out of pilot recipe
choices until the first 152389–152390–152391 sequence is assessed; apply the frozen
recipe to 152392 as the independent SCIENCE check, then coadd. Record that holdout
boundary before runs; do not retune on 152392 and still call it held out. This is an
independent observation check, not an assumption of zero inter-observation covariance.

| Case | Processing and calibration |
|---|---|
| A | Preserved historical reference 152390/152392 and recorded calibration; no rerun |
| B | Successor SCIENCE 152390/152392, compatible archived calibration and pointing |
| C | Same successor SCIENCE settings/observations; calibration and pointing from qualified successor 152389/152391/152393 |

B can expose the SCIENCE connection while C's calibration-use gate remains open;
B must still satisfy its own source-policy, mapping and product gates. Because C
changes both flux calibration and pointing, record both differences; do not call
B→C a flux-only controlled experiment. Native inputs, all three arrays, individual
maps and the real two-observation coadd must be demonstrable.

Use TolProj validation/submission and TolTECA preparation/launch. Include executable
and Python-runtime hashes, effective configuration, selected APT/PPT identities,
resource bounds, fresh output locations and expected products. Audit generated
scripts for cleanup and helper dependencies; preserve references and frozen trials.
Grant alone triggers reductions. Stop on failure; no automatic retries or extra
iterations. Quiet mode remains default, with compact detector-drop counts/reasons,
weight summaries, fit validity, calibration identities and per-pass endpoint drift.

## 4. Required-product proposal from this actual project

| Product | Required meaning and status |
|---|---|
| Three-array POINT maps and diagnostics | Required for every bracket; retain rejected/unavailable outcomes and optical-state context |
| Discoverable PPT for each qualifying bracket | Qualified measurement/units/metadata, including truthful uncertainty and displacement; not merely diagnostic fit JSON |
| Reference-flux report and immutable calibrated APT children | Required for C; exact parent, factors and observation selection; separate from B archives |
| Individual SCIENCE signal maps (2 observations × 3 arrays) | Required; celestial WCS, units, validity/support and honestly labelled normalization/coverage |
| Raw coadd (3 arrays) | Required; both real observations, contributor counts/support, compatible grid and declared operator |
| Noise products | Authored recipe enables 10 realizations plus summaries. Recommend explicitly deferring in the base-map pilot; required if retaining S/N feedback or a noise-dependent analysis. No normalization substitute |
| Filtered coadd (3 arrays) and its required noise/response inputs | Authored recipe enables Wiener/lowpass-only products. Recommend explicit pilot deferral; retain as a later requirement for consumers that use them |
| Individual filtered maps | Not present in the archived reference; not assumed required solely because coadd filtering is enabled |
| Source catalog and retained raw/cleaned TOD | Disabled in current project; exclude from the minimum |
| Compact completion/quality record | Required file inventory, config/runtime/APT/PPT bindings, array outcome and rejection/weight summary; no detector-sample tracing by default |

The immediate named downstream use is observation/coadd discovery, FITS WCS/units
and validity-aware display/measurement, with cross-observation position/brightness
comparison. Full sensitivity, significance, aperture-error or filtered-source
analysis is not implied. Name any additional collaborator consumer before dropping
a product it requires; the currently recovered observation discovery does not
require filtered/noise products.

Exact HDU/sidecar requirements must be checked with the existing downstream readers
in Batch 3. A change from the authored JINC/noise/filter recipe is an explicit pilot
profile decision, not permission to discard requested products silently.

## 5. Existing criteria versus decisions still needed

| Existing criterion/evidence | How it applies |
|---|---|
| Exact IDs, required product presence, valid units, complete requested arrays and safe missing-data behavior | Hard interface gates; no tuning |
| Existing flux factor bounds 0.2–5; bracket disagreement 20% | Preserve factor bounds and warning semantics; 20% is a warning, not invented rejection or proof of calibration soundness |
| Current POINT diagnostic S/N >10 and existing position/width guards | Keep diagnostic meaning; not calibrated flux-error or absolute false-alarm certification |
| Historical SCIENCE equivalence: raw RMS-relative 1e-8; filtered 0.015; PTC weights 1e-9; median absolute 5e-5 and fractional 0.001; other diagnostics 1e-7 | Preserve the named V2 profile for unchanged-method regression. These are not acceptance thresholds for intentional successor algorithm differences |
| Recent compact response, broad deficit and seed-memory measurements | Retain as scoped evidence and known limitations; do not demand legacy amplitude reproduction or use compact response to certify extended-source photometry |

Grant's additional decisions should be limited to actual gaps:

1. **Calibration use of POINT amplitude and uncertainty.** Recommendation for the
   A/B/C pilot: permit a tightly scoped diagnostic calibration transfer with typed
   unavailable amplitude uncertainty, while final photometric qualification remains
   pending. TolProj uses `amp_err` in reported bracket/factor uncertainties, not in
   the central factor `1/mean(amp/reference)`, its bounds or disagreement warning.
   If approved, preserve that arithmetic, propagate unavailability to *all* dependent
   uncertainties, and bind cause/method/pilot-use scope in the digest-covered JSON
   report and its calibrated child. Ordinary use must still reject unavailable
   errors. Do not set generic `flxscale_done=true` or let automatic ordinary setup
   select a diagnostic child; require explicit exact pilot-use selection. Verify
   this restriction through both APT admission and setup. This adds no claim of
   total calibration uncertainty, significance or production photometry and does
   not waive amplitude-transfer or astrometry qualification. Alternatively, require
   a content-bound final-map error method before C; then B can proceed without C.
   Neither option is selected by this plan. Keep M1-induced optical loss distinct
   from detector faults; do not substitute diagnostic frozen-C0 `sigma_A`.
2. **The SCIENCE recipe.** Approve the explicit mode-specific source/training-support
   and feedback policy and the mapping/product disposition against existing
   authority. Keep 2″ sampling and detector noise weighting. The initial 152390 RTC
   no-spatial-mask permission does not automatically cover 152392, PTC/noise learning
   or an extended galaxy's source protection. Reuse recovered policy where applicable;
   scope only the remaining genuine gap. Recommend an explicit ordinary base-map
   profile with noise/filter products deferred for the first connected comparison,
   rather than silently executing the legacy JINC/S/N recipe. With empirical noise
   absent, do not retain the legacy S/N2.5 feedback cut. A no-feedback pass can test
   wiring only; it cannot be presented as an adopted SCIENCE recovery policy.
   Preserve the current raw recipe and full product goals for later qualified use.
   Do not optimize POINT wings here.
3. **Scientific acceptance of the connected outputs.** Set needed pointing/flux-transfer
   and SCIENCE usability bounds before examining new results, using existing criteria
   where applicable. Identify the downstream analysis and independent SCIENCE check
   that those bounds serve. No threshold is selected to make a candidate pass.

The first two decisions block their respective scientific-use paths, not safe
metadata/fixture implementation. A novel method, new target set, new threshold or
increased observation budget returns for a bounded owner decision.

## 6. Checks, scope and completion discipline

Planning uses the TolTEC context routing, engineering/successor/review governance,
combined acceptance/current-state records, architecture/scientific conventions,
APT-v2 and existing product/config authorities. Recovery dispositions and exact
sources are in [PRIOR_WORK.md](../validation/point_science_connection_plan_2026-10-30/PRIOR_WORK.md).
This is an implementation plan, not a new scientific-contract package. If a genuine
new method is required, use the existing charter/prior-work process only for that gap.

Implementation must preserve combined build/resource/input/packaging changes and
extend runtime/wheel/CI inventories for any added modules. Run focused POINT/TolProj/
SCIENCE/coadd fixtures; the ordinary native aggregate and baseline tests for touched
native paths; Python tests/lint/type checks and installed-runtime smoke; and full
configuration preflight. Keep experimental and required-data gates honest. Obtain
one focused independent exact-source review of each coherent implementation
milestone, not repeated reviews for helper edits. All native executables must match
the final Git identity. Supported-environment and owner integration decisions stay
separate from supplemental local tests and scientific acceptance.

No branch integration, pushes, dependency installations, Unity operations, new
reduction, source fit, calibration issuance, policy change or sibling edit is part
of this planning task. The future POINT diagnostic limit is M0+M1–M5, prepared
coherently; it is not an instruction to rewrite frozen M10 trials or silently
replace the SCIENCE recipe's existing four-loop setting. General application
1″/M10 defaults are not the selected 2″ pilot settings.

The work is complete when the exact project interface produces and safely consumes
the qualified POINT/calibration/SCIENCE chain on all arrays, demonstrates the actual
coadd and intended downstream analysis, and reports unresolved scientific limits.
A successful raw-map wiring check, a clean build or a matching peak alone is not
that completion. Future wing research resumes only if the connected evidence
identifies it as a concrete blocker.
