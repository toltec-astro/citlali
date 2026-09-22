# POINT 152389 balanced reconstruction comparisons

## Work order and preflight

Owner-requested bounded extension to the empirical companion: reconcile actual
scan IDs/time ranges, construct every complementary balanced partition from
saved accepted contributions, report local coverage and detector populations,
and retain identical-footprint source and individual-scan measurements wherever
supported. Label all differences first-pass reconstruction differences. Relative
cleaning attenuation is unresolved; no retuning, forced amplitude agreement,
new source-response experiment, uncertainty/significance conversion or FRUIT
change. Original products stay immutable.

Base: 4e8dcc37438559a44f7cab72b706a2ae1d50f6c0; existing branch
codex/map-successor-first-forward-001 and isolated worktree
/private/tmp/citlali-map-successor-first-forward-001. Tracked tree initially
clean, original build symlink untracked; shared ALIGN checkout untouched.
Continuation of this bounded companion module-probe slot, no new branch.

Read AGENTS, ENGINEERING_GOVERNANCE, TIMESTREAM_SUCCESSOR_GOVERNANCE,
REVIEW_AND_CONFORMANCE and accepted ledger bindings; current status,
SCIENTIFIC_CONVENTIONS and ARCHITECTURE. Tier 2 exact-SHA independent review
required. Scientific authority is the current explicit owner request plus
preserved saved-descriptive-v1 definitions documented in POINT_EMPIRICAL.md;
no generic SCI-POINT numerical method is supplied. Existing MAP plan/PTC
publications own admitted samples, unit coefficients, WCS and calibration.
The separate diagnostic command owns grouping, coverage reporting, guarded
measurement and new output lifecycle. No core stage or Engine changes.

The saved processing request explicitly forces duration chunks of 10 seconds;
the realized manifests contain six IDs 0..5, separately from raw telescope
ScanNum=2. Resolve their intervals from bound native-time and processing records,
not file-name timestamps. Ask for the owner's four-scan reference if it cannot
be recovered; never select four of six by convenience. The request explicitly
permits all ten balanced three-versus-three comparisons when six are real.

Source footprints are fixed from the original full-map description: complete
40-arcsec disk around AST center for Gaussian+constant fitting; fixed-template
radius three measured major FWHM and aperture radii one/1.5/two measured major
FWHM around the saved centroid. Own-subset coverage cut remains the parent's
0.5 policy applied to that subset's normalization. No intersection refit or
footprint shrinking. Report source/core, template, full fit, peripheral fit
background and apertures separately. Peripheral region is diagnostic context
for the jointly fitted constant, not a new annulus subtraction estimator.
Coverage sufficiency here means exact pixel support only; it does not establish
equal precision, response or scientific adequacy beyond that condition.

Products: six individual processing-chunk maps plus every balanced subset map,
N/Q/hits/support, all complementary raw differences and common support,
per-footprint normalization distributions/ratios, detector/network contributions
and exclusions, identical-method measurements, scan timing/identity table,
figures, exact inputs/source hashes and separate runtime. No new detector-time
history copies: retain only bounded per-region counts keyed to original detector
occurrences; input masks remain authoritative.

Gates: four/six partition enumeration and complement closure; normalization
sum versus invalid mean-of-maps adversary; own-subset support and source-domain
holes; exact native slot/time identities; exclusions/population accounting;
regression against existing parity split; successor Python and required config
checks, preserved baseline gates; actual three-array comparison, FITS/hash
verification, visual review, and fresh-context independent exact-SHA review.
No native build triggered because no native code/executable is changed/used.
Python prescribed venv, one math thread, Agg backend. No Unity, push,
integration, activation or cleanup authorized/performed.

## Reconciliation and development reassessment

The exact processing-request hash is
`ecb419910fad561399580efc301eed2701a6d959b088120c47cc3ab8254d49a4`.
Its effective config forces duration chunks of 10 seconds. All eleven supplied
networks realize IDs 0..5. The telescope has raw ObsNum152389/SubObsNum0/ScanNum2,
60-second requested integration, and ScanPos=0 across all 3200 telescope rows.
No four-scan grouping is identified in those inputs. The older local POINT
config uses non-forced 5-second chunking; it is not the saved successor request.
An optional question for the owner's four-scan definition remains unanswered;
the explicit six-scan alternative in this work order authorizes all ten current
balanced comparisons without inventing a four-element selection.

Native science row ranges and UTC sample-center bounds are retained separately
per network. The first development membership check incorrectly compared RTC
slots with every native science row; a2000 correctly uses RTC factor2. Repaired
the diagnostic to project declared native intervals onto the existing factor/
phase lattice before requiring exact equality; added factor/phase and disjoint
interval tests. No input data, schedule or upstream operation was changed.
An initial import referred to a verifier's private nested helper; the companion
now owns its small explicit digest check. Failed development runs remain scratch
evidence, never represented as accepted results.

Development coverage results: all individual 40-arcsec source fits unavailable;
8/1/0 complementary fit comparisons and 10/8/3 template comparisons supported
for a1100/a1400/a2000. New missing-pixel heatmaps expose one-pixel gaps that a
near-unity support-fraction image could conceal. Complete support is a necessary
measurement condition, not an assertion of response equivalence or matched noise.

Before runtime binding: 214 successor tests with 111 subtests pass; required
config preflight passes. New focused coverage/partition/replay tests number10.
Source and measurement definitions remain unchanged except the explicitly
requested complete-domain availability gate on these new comparisons. Final
runtime must bind a committed source; fresh-context review follows.


### Independent-review repair

Initial runtime candidate `af9faee61251a050493ce1df475a63490191f17b`
passed numerical/coverage reproduction, but review found that native receipt/time
self-consistency alone did not bind timing to the saved MAP input ledger. Actual
inputs match that ledger. Added attempt→bundle/request→manifest→native timing
verification and exact ordered native interval matching against bound PTC segment
records, including unavailable segments and factor2 endpoint changes. Regression
tests reject paired receipt/time replacement, interval edits and reversed timing.
All 215 successor tests and 111 subtests pass after repair. Initial artifacts are
retained; the repaired committed run will be published separately as comparisons-r2.


## Final runtime and conformance evidence

Repaired runtime source `12e1fc80a489ed898dd78357d5b46c4ecd24a058`, tree
`419d3227ddec3a8d30ff9883bfad01b96ada27ff`; unchanged task base above.
Products are under
`/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-balanced-pairings-20260922`.
`RESULTS.md` is the compact owner report and `comparisons-r2/REPORT.md` contains
all individual and paired values, local coverage/populations and FITS/figure links.
Initial `comparisons` remains preserved evidence; `comparisons-r2` is final.

Exact identities:

- Root FILES_SHA256SUMS: `4cf91735b38ce78bb6e509d8e409e8062d33ef642d135640d33aece49b233d55`.
- Final product FILES_SHA256SUMS: `174eb603fb07926ed99d442bf3438e675f9318f1b3197dbd601d867896ba3422`.
- final-source.json: `85dffbbc90528e72adda5a4a252355a94aa05aca37a6768b4eac732c83ce1d73`.
- final-source.json records every changed runtime-source path digest; product
  provenance records environment/library versions, exact command and code hashes.

Across all networks, native science sample centers span 2026-02-19
06:34:47.581–06:35:47.022 UTC. Per-network boundaries differ by milliseconds;
absolute epoch uncertainty remains unquantified. The runtime realizes six forced
10-second chunks. The separate four-scan reference remains unidentified; all ten
balanced partitions are produced without discarding or inventing identities.

| Array | Complete 40″ paired fits / 10 | Complete paired templates / 10 | Complete individual 40″ fits / 6 |
|---|---:|---:|---:|
| a1100 | 8 | 10 | 0 |
| a1400 | 1 | 8 | 0 |
| a2000 | 0 | 3 | 0 |

The alternating 024 versus135 fitted differences are −85.57/−133.50/unavailable
mJy/nominal beam. Template differences are −63.18/−66.63/unavailable. Source-core
contributing detectors A/B/common are 737/741/449,366/327/213,324/313/238.
These are local admitted populations, not whole-network availability counts.
Complete support alone does not assert comparable depth, background, or response.
All source-shaped residuals remain first-pass reconstruction differences with
inherited cleaning attenuation unresolved. The existing nw3 chunk5 ALS failure
and unavailable raw nw6/nw10 remain explicit; no additional network is removed.

Final companion wall73.197 s, CPU43.287 s, peakRSS950910976 bytes (906.9 MiB),
previously accessed inputs. Initial companion wall39.884 s/CPU37.380 s; the guarded
rerun also reads/verifies the saved input ledger. These are measured run timings,
not a performance benchmark. No RTC/PTC/native execution or new learning occurs.
All139 product-manifest entries and483 HDUs in111 FITS files verify. Every numeric
FITS plane, group/comparison measurement and scan time equals the initial run.
Allnine PNGs are bit-identical and visually inspected. Parent MAP hashes unchanged.

Focused repaired tests:11 pass (independently reproduced). Full successor:
215 pass/111 subtests. Baseline:207 pass/137 subtests. Required config preflight:
pass. Evidence logs reside in root/evidence; postrepair successor and verification
are explicitly suffixed r2. No unexpected error-level output in final gates/run.
Native CTests/build and representative Spack/Unity are not triggered by this
Python-only saved-product diagnostic; earlier native evidence is not claimed as
new validation. This work adds no source-recovery-through-PTC validation.

The independent reviewer reproduced all78 own-subset support masks, all30
complementary normalizations, allavailable saved-profile fits and exact local
population sums. Original parity maps have exactly identical Q and support;
signal differences from summation reassociation are only2.84e−13–6.82e−13.
Review found and verified the bounded timing-binding repair above. Final exact
SHA disposition follows in the closure record.

Changed implementation paths: point_pairings.py,point_pairings_io.py,
run_point_pairings.py,test_point_pairings.py,POINT_EMPIRICAL.md, under
`tools/timestream_successor`; this handoff and living status are evidence docs.
Scientific source measurements, WCS, nominal beams, accepted coefficients and
upstream algorithm owners remain unchanged. The diagnostic owns only grouping,
local support availability, comparisons and separate artifact lifecycle. No
Engine/core growth. No integration,push,activation,production qualification or
cleanup. Worktree tracked files are committed; the pre-existing untracked build
symlink remains. Stop at owner assessment of these diagnostics; a four-scan
interpretation or source-response experiment would require a separately named
next step, not an implicit continuation here.
