# Citlali Refactor Status

## 2026-08-24 Bounded Native Scientific Provenance

The owner rejected the proposed compressed NetCDF encoding of exhaustive
native detector-sample lineage. Runtime sample state, canonical scientific
provenance, and diagnostic tracing now have separate contracts under ADR 0013.
The native-required Science route publishes the bounded
`citlali-native-cohort-product-provenance-v3` contract: authoritative
observation/APT/input/config/software identities, one observation-wide record
per APT detector exclusion, named causes at their natural scope, reconciled
scan populations, detector-scope weight/map identities, and final product
checksums. It neither serializes nor hashes a complete detector-by-sample
revision history. The historical v2 contract remains independently readable,
and execution repair `9c3b71e79` remains its own reviewable commit.

Detailed sample tracing is now an explicit diagnostic API, disabled by
default, with required scan/network/detector/row selection and a hard record
bound. Its products declare themselves noncanonical and carry no retention or
completion requirement. Ordinary publication never calls this API.

Native publication is fail-closed at the index boundary. Iteration completion
does not publish the reduction-root index for a native-required run. The
observation index follows validation and atomic publication of the bounded
sidecar, and the final checksum-bearing root index follows every required
reduction sidecar. Live log/profile files are classified as mutable
operational artifacts and are explicitly excluded from the immutable checksum
set rather than carrying stale checksums. The v3 sidecar records an explicit
`validated_complete` publication status, and the reduction auditor requires
it.

The frozen local 152390 native-gap campaign completed all 124 scans and all
three array maps with no error- or critical-level records. The 8,168,140-byte
bounded sidecar contains 1,006 observation-wide APT exclusions and 124 scan
summaries, declares `detector_sample_expansion: false`, and replaces the
previous roughly 835-million-record requirement. All 23 independently checked
hash bindings across the three product indices passed. The data-pathology
review is separate from completion: 77 network-step masks occurred across 39
scans, and 204 local-residual despike guard summaries reported 1,207 rejected
detector occurrences, including a maximum added fraction of 0.9598 against the
0.1 cap. These are observed-data events, not software completion failures.

All 817 runnable CTests pass with the one established disabled test not run;
the focused bounded-provenance/publication tests, Python provenance auditors,
and required config preflight also pass. The localized production JINC config
differs from the downloaded Unity merge in local paths and one explicit
scientific correction: empirical noise products are enabled because JINC
fruit-loop S/N selection requests them. The downloaded merge requested ten
noise realizations and positive S/N selection while disabling those products,
which is invalid and fails the typed validator.

The exact production-JINC replay exposed an orchestration-parity gap in the
compact-v2 native-required route rather than a sequence of independent JINC
defects. That route bypasses mature Lali orchestration and initially admitted
only a restricted RTC/PTC subset. A first separately gated execution layer now
supports exact run-local extinction FCF aggregation, native kernel carriage
through RTC and PTC, and gap-bounded full-cohort RTC execution for line audit,
Alt/Az destriping, cross-network coincidence, and detector inverse-variance
cuts. The full-cohort adapter preserves exact common-slot identity, accepts
interleaved input detector columns through stable internal grouping, scatters
back to original identities, and runs before downsampling. The production
replay now reaches the next intentional boundary, reduction learning.

The second execution layer now admits the established PTC second-pass only on
complete network cohorts within each native gap-bounded segment. Numerical
results carry an append-only runtime exclusion mask through PTC and into map
weighting; neither existing exclusions nor gaps may be removed. Post-clean
detector-outlier exclusions are applied afterward. Canonical provenance
reconciles the final mask actually consumed by mapmaking, but reduces new
second-pass and post-clean exclusions to named detector intervals and counts;
it does not publish the runtime mask. Focused tests cover complete-network
admission, rejection of incomplete groupings and mask rollback, final map
eligibility, and bounded interval causes.

The third execution layer restores ordered reduction-learning behavior without
publishing an exhaustive detector-sample ledger. Earlier-iteration RTC masks
are applied on absolute common-slot coordinates before each native RTC body;
earlier-iteration PTC masks are applied on concatenated, gap-bounded output-row
coordinates before each established PTC body. Learned detector and network
exclusions run at their configured pre-RTC, pre-PTC, and pre-map boundaries.
The exact applied pre-map detector set, rather than a difference inferred from
already-flagged samples, owns detector-weight zeroing. Bounded canonical
lineage distinguishes learned RTC flag bits, learned PTC detector intervals,
and learned pre-map detector causes with named authorities and reconciled
counts. Canonical UIDs outside the legacy learning integer domain are rejected
instead of narrowing into a different detector identity.

Focused activated-learning tests prove strict earlier-iteration selection,
both coordinate systems across a native gap, append-only PTC masks, exact
pre-map weight exclusion even when samples were already flagged, all three
bounded cause authorities, and oversized-UID non-aliasing. The CLI builds, all
817 runnable
CTests pass, and the required config preflight passes 129 Python tests, four
mode kits, eight compatibility cases, and every authority audit. Native parity
for noise realizations and fruit-loop feedback remains required before the
clean-commit JINC replay and one exact-SHA Unity confirmation.

## 2026-08-24 Build-Time Git-Provenance Refresh Repair

The Stage 7 Unity workflow exposed that a successful build after changing Git
commits could compile current source while retaining an older embedded Citlali
revision. Tula's Git-version helper queried Git only during CMake configuration;
its generated-header target depended on the already-existing header and had no
input edge that changed when HEAD moved. A build-only invocation therefore had
no reason to refresh `citlali_config/gitversion.h`, making `citlali --version`
and reduction provenance potentially stale even when compiled implementation
files were current.

The active Citlali graph now adds an always-run build dependency ahead of the
existing generated-header target. It resolves the current short revision and
`git describe` identity, rewrites the header atomically only when that identity
changes, and otherwise leaves its contents and timestamp untouched. A Git
identity change therefore recompiles the version-dependent objects in the same
build invocation, while an unchanged checkout retains a true no-op build. A
missing Git executable or unreadable checkout fails the provenance refresh
instead of silently reusing an unverifiable identity.

The build-system regression creates a temporary Git repository and executable,
builds its first commit, advances HEAD, and proves that a second build without
reconfiguration embeds the new revision. It also proves that a following no-op
build does not rewrite the generated header. The real Citlali target reproduces
both outcomes: the first identity refresh recompiles and relinks the dependent
objects, and the next build performs no compilation or link. The complete
local surface passes all 800 runnable CTests with the one established disabled
test not run, all 205 baseline-tool tests, and the full required config gate:
129 unit tests, all four mode kits, and all eight compatibility cases. A final
owner-run Unity confirmation remains pending.

## 2026-08-24 Native Explicit-MJD Pointing-Support Repair

The owner-run Stage 7 NGC4449 science reduction for observation 152390
successfully admitted its observation-matched, pointing-flux-calibrated APT v2
and reached native RA/Dec geometry, then failed because the native pointing
offset model used the narrower common science telescope grid as the support for
two pointing-derived offset values. Exact detector timestamps may legitimately
extend just beyond that common grid while remaining inside the explicit MJD
bracket supplied by pointing observations 152389 and 152391, so the strict
no-extrapolation guard rejected a supported target.

Two-value native pointing offsets now use their positive, increasing explicit
MJD pair as the interpolation support, converted through the established
MJD-to-Unix utility. Only configurations without an explicit MJD pair retain
the observation-span support. Constant offsets are unchanged, and the native
model continues to reject evaluation outside the applicable calibration or
observation support; this repair does not permit extrapolation.

Focused regressions prove that exact native timestamps outside the common grid
but inside the explicit calibration bracket are accepted, that targets outside
the calibration bracket remain rejected, and that the no-explicit-MJD fallback
retains common-grid support. Both the 10-test native-carrier suite and the
16-test astrometry/native suite pass. The CLI builds, all 799 runnable CTests
pass with the one established disabled test not run, all 205 baseline-tool
tests pass, and the full required config gate passes 129 unit tests, all four
mode kits, and all eight compatibility cases. The validation and science-change
ledgers, profile registry, and Phase 5 readiness report also validate. Owner-run
Unity replay of the Stage 7 science reduction remains pending.

## 2026-08-24 Canonical APT v2 Pointing-Flux Calibration

Citlali now owns issuance and admission of immutable
`citlali-observation-fluxcal-apt-v2` children from an ordinary fresh
matched-v2 parent. The public `issue-fluxcal-apt-v2` operation verifies the
parent, a closed TolProj request, the request/report digests, exact observation
and cohort bracket identity, and one positive finite binary64 correction for
each TolTEC array. Publication is no-replace and receipt-last, followed by an
independent filesystem reread. TolProj remains the correction-estimation and
orchestration owner; it does not write canonical APT bytes.

The calibrated child retains the matched detector relation and changes only
positive finite `flxscale` values through typed `field-deviation` exceptions.
Nulls remain null and finite zeros remain bitwise unchanged. The authority
reference binds the matched-parent semantic identity, request digest, report
digest, and all three exact factors; each exception repeats its applicable
factor so admission proves the output by multiplication instead of recovering
a potentially rounded ratio.

A local end-to-end smoke used the downloaded Stage 7 NGC4449 152390 matched-v2
bundle: all 5,518 observation rows reread successfully, with 813 null scales
and 193 zero scales preserved and all 4,512 positive scales transformed with
zero bitwise mismatches against the requested three-array factors. The focused
seven-test canonical-v2 C++ suite and the 32 focused TolProj integration tests
pass. The complete Citlali surface passes all 797 runnable CTests with the one
established disabled test not run, all 205 baseline-tool tests, and the full
required config gate: 129 unit tests, all four mode kits, and all eight
compatibility cases. The complete TolProj surface passes 222 tests with one
established skip and the full Ruff check.

## 2026-08-24 Flagged Non-Finite PCA Masking Repair

The canonical Stage 7 two-observation Pointing replay with empirical map
weighting enabled exposed a processed-timestream masking defect rather than a
JINC or weighting defect. The matched-v2 APT correctly represents unavailable
detector fields as typed nulls, and the corresponding detector/sample flags
correctly exclude those payloads. RTC retained finite active samples, but the
ordinary PCA paths used multiplication by a zero/one mask. Under IEEE
arithmetic, a flagged `NaN * 0` remains NaN, so nullable flagged detector
payloads contaminated the PCA projection and made every PTC detector sample
non-finite. JINC then correctly admitted no contributors, and the empirical
global-nonprecision scale failed closed as unavailable.

All ordinary and adaptive PCA covariance, projection, and correction paths now
use explicit selection of a finite zero for excluded samples. PTC mean
subtraction applies the same rule to detector and kernel means. Flagged payloads
remain outside the numerical contract and retain their original representation
in the cleaned result; unflagged finite values and the established PCA
algorithm are otherwise unchanged. A focused regression alternates NaN and
infinity in flagged cells and proves that the eigensystem and every unflagged
cleaned sample are invariant to those excluded payloads.

The complete downloaded Stage 7 local replay succeeds with the canonical
empirical-weight configuration for both 152389 and 152391. Every unflagged PTC
sample is finite (13,333,916 and 13,467,298 samples respectively); the JINC
maps contain 307,096 and 300,853 supported pixels with maximum contributor
counts of 56,082 and 55,715; every array has a positive formal map-weight sum
and a finite empirical scale. Both observations produce valid Pointing fits,
the reduction exits successfully, and its log contains no error- or
critical-level entries. The complete local C++ surface passes all 797 runnable
CTests with the one established disabled test not run. All 205 baseline-tool
tests pass. The full config preflight passes 129 unit tests, all four mode kits,
and all eight compatibility cases.

The owner then supplied the completed Unity Stage 7 Pointing reduction from
job 63565989. The log records Citlali `v4.0.0-3693-g16dda3011`, KIDs
`04088da`, and the expected Citlali executable SHA-256
`35940cc8b96a0084a61ed350376bb274d871f41a6900fc85eac2220bef610173`.
The six-thread OpenMP reduction completed in 172.493 logged seconds with no
error- or critical-level records, marked `status.pointings_done` true, and
published both observations plus all required reduction sidecars. Its two PTC
products reproduce the local finite-eligibility counts exactly, and its JINC
support/contributor counts and empirical scales agree with the local replay.
All six Pointing-fit rows are bit-for-bit identical to the local single-thread
replay across every ECSV column.

Intake exposed validation-tool drift rather than a product failure. The general
auditor still admitted only mapmaking provenance v1/v2 and coadd provenance v1,
although the reviewed application contracts emit mapmaking v3 and coadd v2.
The preparing Phase-5 Pointing profile also still referenced the immutable
Phase-4 product contract, whose legacy FITS unit spellings are `sec` and `N/A`;
the current JINC Pointing products correctly use FITS-standard `s` and `1`.
The auditor now admits and semantically cross-checks the successor schemas, and
the preparing profile uses a new `phase5-point-products-v2` successor contract
without changing the Phase-4 snapshot. The supplied Unity reduction passes all
required provenance checks and all 27/27 classified product checks. This is
owner-supplied Unity campaign evidence, not an accepted-ledger promotion; the
Phase-5 profile remains preparing pending the coordinated validation decision.

## 2026-08-24 Stage 7 Pointing Output Completion Repairs

The downloaded Stage 7 two-observation Pointing project now completes a local
single-iteration replay after disabling empirical-weight application for this
bounded output/lifecycle check. Both 152389 and 152391 independently load their
own telescope file and matched-v2 APT, run through mapmaking, write all three
array FITS products, publish raw-timestream lineage, and participate in final
reduction provenance. This confirms the earlier telescope observation-state
replacement repair across the complete two-observation lifecycle; it is not a
claim that the separately configured empirical global-nonprecision scale gate
has passed.

The replay exposed two independent output defects after mapmaking. Invalid
Gaussian fits correctly produce non-finite optional pointing-table values and
`fit_valid=0`, but the FITS adapter attempted to serialize those values as
floating-point header cards. It now omits only non-finite optional
`POINTING.*` fit-value cards while retaining the ECSV table values and explicit
fit-validity card. The existing fruit-loop map reader already treats those
value cards as optional and requires a valid fit before consuming them. The
replay then exposed a NetCDF API mismatch in final noise-product validation:
querying a missing `comment` attribute throws rather than returning a null
attribute. Validation now enumerates attributes, remains strict for the three
noise-contract variables, and ignores unrelated variables without comments.

The owner-local Pointing compatibility fixture moved under the campaign's `v1`
directory, so the compatibility manifest now follows that existing location.
The complete local C++ surface passes all 796 runnable CTests with the one
established disabled test not run. All 203 baseline-tool tests pass. The full
config preflight passes 129 unit tests, all four mode kits, and all eight local
compatibility cases. The bounded Stage 7 replay exits successfully and writes
the final noise, post-processing, and Pointing provenance sidecars.

## 2026-08-24 Distinct OOF Reduction Identity

OOF is now a distinct public low-level reduction identity instead of being
encoded as `runtime.reduction_type: pointing`. The runtime enum, parser,
canonical TolTECA v1/v2 mode kits, compact expander, reverse converter,
baseline auditor, and provenance-facing configuration all carry `oof`
directly. Pointing and OOF remain one numerical execution family and both
dispatch to the existing `PointingTodProc`; this change does not fork or
duplicate their mature RTC, PTC, mapmaking, fitting, or output algorithms.
Exact identity remains available for future mode-specific contracts.
The changed canonical kits are versioned as successor `phase4.1-v1.1` and
`phase4.1-v2.2` artifacts so TolProj cannot confuse them with an installed
pre-change bundle.

Complete matched-v2 APT authority is admitted equivalently for Science,
Pointing, and OOF, and partial native authority is rejected for all three.
The current native numerical activation candidate remains Science-only.
Pointing and OOF therefore publish admitted matched-v2 calibration values but
continue through their established legacy numerical path until each mode has
an approved native product and kernel-support contract. Existing low-level OOF
files that say `pointing` remain valid legacy Pointing requests; new canonical
OOF kits say `oof` and select the same Pointing processor explicitly.

The complete local C++ surface passes all 795 runnable CTests with the one
established disabled test not run. All 203 baseline-tool tests pass. The full
config preflight passes 129 unit tests, all four v1 mode kits, all config
authority/boundary gates, and the six locally available OOF, Beammap, and
Science compatibility cases. OOF's two historical comparisons admit exactly
the one named `runtime.reduction_type` identity migration; all numerical
leaves remain exact. The two Pointing compatibility cases retain the recorded
missing owner-local fixture and are not represented as passed. The independent
v2 mode-kit validator also passes all four modes with the revised OOF policy
digest. The coordinated TolProj vendor update remains the closeout gate for
this candidate.

## 2026-08-24 Telescope Observation-State Replacement Repair

The owner-run Stage 7 two-observation Pointing reduction passed compact-v2
admission and the repaired constant-offset evaluation. During initial geometry
for the second observation, 152391, native telescope capture rejected
`alt_phys` under its finite/shape contract. The reduction-lifetime
`Telescope` compatibility object retained derived tangent-plane fields created
while preparing 152389. The next telescope load overwrote configured raw
fields but did not clear the observation-owned data/header maps, so the stale
152389 `alt_phys` entered the 152391 native snapshot with the earlier
observation's cardinality and values. The same behavior could also retain an
optional raw field or header omitted by a later telescope file.

The telescope file boundary now clears only the observation-owned data and
header containers immediately before every load. Reduction-wide telescope
configuration remains intact. The next observation therefore constructs its
raw trajectory solely from its own file, and missing optional inputs remain
missing instead of inheriting prior values. A pipeline regression seeds stale
raw, derived, and optional-header state and proves that all are replaced before
the next loader runs. This changes no telescope values within an observation,
interpolation, pointing offsets, RTC/PTC, mapmaking, or product contract.

Local verification builds the complete CLI, passes all eight focused telescope
pipeline cases and all 68 SCI-ALIGN cases, and passes all 793 runnable CTests
with the established single disabled test not run. All 203 baseline-tool tests
pass and both ledgers are valid. The config preflight passes all 127 unit tests,
all four mode kits, and the six locally available OOF, Beammap, and Science
compatibility cases. Its two Pointing cases retain the recorded missing
owner-local `point/refactor/70_reduce.yaml` fixture and are not represented as
passed.

## 2026-08-23 Native Constant Pointing-Offset Support Repair

The owner-run Stage 7 Pointing reduction using repaired matched-v2 APTs passed
canonical admission and reached native telescope/offset construction, where it
failed with `native pointing-offset target is outside support`. TolProj's
Pointing kit supplies one finite azimuth value and one finite altitude value,
both zero. That is the established constant-observation offset model, but the
native evaluator incorrectly subjected it to the support bounds required only
for two-value interpolation. Exact delivered detector timestamps may extend
slightly beyond the legacy common-slot grid while remaining inside the raw
telescope trajectory, so a valid constant model was rejected.

The native evaluator now applies the no-extrapolation support guard only to a
genuine two-value interpolation model. A one-value model remains constant at
every finite exact native target time; raw telescope interpolation retains its
independent strict measured-support guard. Focused coverage proves constant
azimuth/altitude values on both sides of nominal common support and proves a
two-value model still rejects the same request. This changes no reconstructed
timestamp, telescope interpolation, offset value, detector relation, RTC/PTC
kernel, mapmaker, or product contract. The complete 792-test runnable CTest
surface passes with the established single disabled test not run, and all 203
baseline-tool tests plus 137 subtests pass. The config preflight passes all
127 unit tests, all four mode kits, and the six locally available OOF,
Beammap, and Science compatibility cases. Its two Pointing cases retain the
recorded external owner-fixture gap and are not represented as passed.

## 2026-08-23 Canonical APT v2 Typed-Null Issuance Repairs

The first owner-run Stage 7 Pointing reduction using project-local matched-v2
bundles exposed a producer/consumer contract inconsistency. The observation
issuer correctly made every copied Beammap-baseline field nullable and marked
it `copy-seed-or-null`, but retained the baseline field's
`missing_policy: reject`. General matched-bundle verification did not check
that policy, so publication and TolProj validation succeeded; the stricter
typed detector-relation admission then correctly rejected the bundle because
an unmatched detector requires an explicit typed null.

The issuer now records `missing_policy: typed-null` for every copied baseline
field, and general matched-bundle validation independently requires the same
policy. The public protocol regression now issues and rereads a TolProj-style
bundle, admits its typed detector relation, proves a matched baseline `flag`
is retained, proves an unmatched `flag` is null, and rejects a copied-field
policy changed back to `reject`. This changes no target/seed choice, detector
identity, finite copied value, Pointing numerical algorithm, RTC/PTC,
mapmaking, or Beammap baseline bytes.

The first owner-run regeneration with that repair exposed the companion value
conversion defect. A selected 137389 seed carried
`cal_amp_over_fit_amp` as the baseline's permitted `nan-token`; the issuer
declared the matched field `typed-null` but copied the raw NaN, so its own row
validator correctly rejected publication. Matched-v2 issuance now normalizes
both baseline nulls and permitted baseline NaNs to the one declared typed-null
representation. Finite selected-seed values remain exact, and independent
matched-bundle verification reconstructs the same normalization rule. The
public protocol regression includes this production-shaped matched-seed NaN
case and completes a filesystem reread plus typed detector-relation admission.

The owner removed the three defective project-local Stage 7 bundles rather
than retaining them as evidence, so TolProj may regenerate the original labels
after deployment. The verified observation 137389 Beammap baseline remains
reusable and unchanged. Local verification builds the affected test targets,
passes all 791 runnable CTests with the established single disabled test, and
passes all 203 baseline-tool tests plus 137 subtests. The earlier config
preflight result for this branch remains 127 unit tests, all four mode kits,
and all six locally available OOF, Beammap, and Science compact-compatibility
cases. Its two Pointing cases remain unavailable because the owner-local
`point/refactor/70_reduce.yaml` fixture is absent, the same recorded external
input gap as the parent candidate; therefore the local `--require-all` config
gate is not represented as complete.

## 2026-08-23 Legacy Pointing APT Admission Repair

The first owner-run Stage 7 Pointing setup exposed a compatibility regression
at the APT format boundary. TolProj correctly supplied the established
observation-matched Pointing ECSV, but Citlali routed every APT locator through
compact-v2 root-manifest admission. The strict guardian correctly rejected the
legacy filename because it was not an absolute `manifest.ecsv` locator.

Citlali now dispatches explicitly on the fixed compact-v2 root name:
`manifest.ecsv` receives strict v2 verification with no fallback, while every
established non-manifest APT ECSV uses the restored legacy loader. The load
remains transactional, and a rejected manifest cannot be reinterpreted as a
legacy table. This changes no APT values, matching, detector filtering,
Pointing numerics, ALIGN, RTC/PTC, JINC, or TolProj behavior. Stage 7 may use
the unchanged legacy Pointing products for observations 152389 and 152391 and
the explicit matched-v2 science product for observation 152390.

Local verification passes all 647 runnable C++ tests with the established
single disabled test and builds the complete CLI. The required config gate
passes all 127 unit tests and all four mode kits; its six locally available
OOF, Beammap, and Science compact-compatibility cases pass. The two Pointing
cases remain unavailable because the owner-local
`point/refactor/70_reduce.yaml` fixture is absent, the same recorded external
input gap as the parent candidate.

## 2026-08-23 TolProj-Orchestrated Observation APT v2 Candidate

The controlled `issue-observation-apt-v2` producer boundary is implemented as
an integration candidate. TolProj owns project/cohort selection, exact raw/KMP
source discovery and byte binding, and realization of the established tone
matcher. Its compact SHA-pinned match request contains exact target KMP facts,
source/application ranks, matcher evidence, and selected target/seed UIDs.
Citlali remains the sole authority for target/relation validation, canonical
identities, selected-seed value copying, typed unmatched nulls, receipt-last
publication, and filesystem reread.

The operation requires one fresh verified Beammap baseline and publishes one
fresh observation-matched bundle without replacing any existing destination.
Digest disagreement and post-request source-byte tampering fail before
publication. `canonicalize-target-v2` and `migrate-v1-to-v2` remain disabled.
This candidate does not change the Beammap producer, matcher numerics, ALIGN,
JINC, RTC/PTC, detector membership, the shared APT library, or legacy APT
selection. Stage 7 remains unaccepted until the owner runs the 152390 campaign
and supplies the matched-product and science-reduction evidence.

Local candidate verification passes the complete 646-test C++ binary with the
established single disabled test, all 203 baseline-tool tests plus 137
subtests, and all 127 config unit tests with all four mode kits. The six
locally available OOF, Beammap, and Science compact-compatibility cases pass.
The two Pointing cases remain unavailable because the owner-local
`point/refactor/70_reduce.yaml` fixture is absent; this is the previously
recorded external-input gap.

## 2026-08-23 Compact-v2 Product-Index Contamination Repair

The first owner-produced Beammap compact-v2 baseline exposed a publication
boundary defect before Stage 7 Science activation. The Beammap producer wrote
the receipt-complete `.apt-v2` bundle correctly, but the later generic
reduction product-index pass recursed into that bundle and added an unbound
`index.yaml`. Public `describe-baseline-v2` verification correctly rejected
the resulting directory as containing an extra member. The adjacent
`manifest.ecsv.sha256` is the canonical five-line receipt, not a GNU
`sha256sum -c` input file.

Generic product indexing now treats every `.apt-v2` directory as an opaque
product namespace. Its parent index may list the bundle, but the indexer does
not recurse into it or write any member inside it. A regression test freezes
the bundle membership across final product-index publication. This repair
does not change APT values, Beammap fitting, native alignment, RTC/PTC, or
mapmaking behavior.

The focused SCI-ALIGN publication cases and all 67 SCI-ALIGN cases pass. All
789 runnable CTests pass with the established single disabled test not run,
and all 203 baseline-tool tests pass. The config gate passes all 127 unit
tests, all four mode kits, and all six locally available Beammap, Science, and
OOF compact cases. Its two Pointing compatibility cases remain unexecuted
because the owner-local `point/refactor/70_reduce.yaml` fixture is absent;
this is an external-input gap, not a code failure.

An already produced field bundle can be salvaged without rerunning the
Beammap by retaining the injected `index.yaml` as evidence outside the
`.apt-v2` directory, leaving every canonical member untouched, and then
passing the public compact-v2 verifier. Successful verification is still
required before the baseline is eligible for the separately gated
observation-matched producer work; Stage 7 remains unaccepted.

## 2026-08-21 Compact-v2 Native ALIGN Consumer Reconstruction Plan

The orphaned native-consumer behavior in historical commits `fd3627fc7` and
`9d9d55a54` has been reduced to a bounded compact-v2 reconstruction plan. The
historical patches remain ineligible for cherry-pick: together they mix a
superseded canonical APT v1 authority with 48 application paths spanning
alignment, KIDs ingress, RTC/PTC, pointing, mapmaking, and product provenance.

The [reconstruction plan](COMPACT_V2_NATIVE_ALIGN_CONSUMER_RECONSTRUCTION_PLAN_2026-08-21.md)
defines compact-v2 detector identity, native-time and gap invariants,
transactional gather/scatter, numerical-preservation boundaries, and seven
separately gated implementation stages. It also promotes the Beammap
correction to an explicit mode contract: detector/automatic Beammap remains
the raw/APT producer; the existing non-detector calibration-table lane remains
unchanged; and neither may acquire an observation-matched consumer lineage
merely because native timing or pointing is present.

Independent review of exact plan commit `82b086856f891873167760534b64a0811840f3cb`
returned `revise` on 2026-08-22. The revised plan distinguishes both Beammap
calibration lanes, names baseline-governed `flag` and its authorized typed
missing state, freezes exact signed-counter continuity and gap association,
and assigns immutable observation, mutable scan/chunk, and output publication
owners. See the [initial review record](../handoff/COMPACT_V2_NATIVE_ALIGN_PLAN_INDEPENDENT_REVIEW_2026-08-22.md).

Independent re-review then returned `accept` with no blockers for exact plan
commit `a3f2bf465a26048b24017ebd50876c4a2684b1b8`, tree
`3ef26b7f05413dd3a48139fb0be3fd0586a59a2b`. The
[acceptance record](../handoff/COMPACT_V2_NATIVE_ALIGN_PLAN_ACCEPTANCE_2026-08-22.md)
opens Stage 1 only: immutable verified compact-v2 bundle-to-detector-column
relation and atomic `Calib::get_apt` publication, without runtime consumer
activation. Later stages retain their separate stop gates.

Stage 1 is implemented at exact commit
`da9e1deac139f4904d059822e8518259838e45c0`, tree
`41897db2ef8468d408153c7da2110812b880c36e`. The immutable relation binds the
verified matched bundle and relation component to presentation-ranked legacy
detector columns while retaining exact target, raw network/channel, rank,
disposition, selected-seed, and baseline-governed `flag` facts. The numeric
APT compatibility view and typed relation publish through one candidate
`Calib` transaction; rejection leaves live APT and derived grouping state
unchanged.

Science and Pointing loads retain this relation. Detector/automatic Beammap
continues through its raw producer lane. Existing non-detector Beammap loading
keeps its numeric calibration-table view but explicitly discards native-
consumer lineage. No runtime consumer is activated. The
[Stage 1 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE1_2026-08-22.md)
records focused rejection coverage, public-header isolation, 736/736 runnable
CTests, the established single disabled test, 203/203 baseline-tool tests, the
complete 127-test required config preflight, and valid ledgers. That accepted
snapshot opened the separately gated Stage 2 carrier work.

Stage 2 is implemented at exact commit
`838f50249ac07bd90308f90f49397d3a38c4cd4a`, tree
`f38ef2fe1e1cacc676a250a21474dd1877208114`. Immutable observation-owned
carriers now preserve per-network delivered timestamps, exact signed packet
counters, counter discontinuities, scan-bounded contiguous runs, and the
relational association of native rows to existing compatibility slots.
Association pins the established one-candidate `std::round` behavior, the
inclusive realized-`dt/2` edge, injectivity, and exact legacy presence-mask
parity; it never synthesizes a missing detector sample.

Raw telescope trajectories and the existing one- or two-value pointing-offset
model are evaluated at each network's exact native detector times. The
pointing carrier is admitted only when every network, row interval, and sample
identity exactly matches its immutable alignment handle. Observation
publication constructs and validates the complete alignment/pointing pair
before one pointer swap, so rejected absent, stale, foreign-scope, duplicate,
nonfinite, fractional, out-of-range, or cross-handle input leaves the accepted
pair pointer-identical.

The [Stage 2 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE2_2026-08-22.md)
records 7/7 focused carrier cases, 22/22 complete SCI-ALIGN cases, 743/743
runnable CTests, the established single disabled test, 203/203 baseline-tool
tests, the complete required config gate, the CLI build, and valid ledgers.
Current common-time compatibility products and all numerical routes are
unchanged. Detector values, RTC/PTC, naive/JINC mapmaking, products, and runtime
activation remain outside this stage; Stage 3 has not begun and retains its
separate stop gate. No Unity run is required for this carrier-only stage.

Stage 3 is implemented at exact commit
`6008ec6330e7058c7c87f3a6a7e568165763f35b`, tree
`de5edb31e645bf662e4dfe82daf890f4ba38863f`. The scan admission boundary now
joins every raw KIDs source and channel to exactly one compact-v2 detector
column while retaining exact raw source, network, channel, detector-column,
output-UID, disposition, and baseline `flag` identity. The immutable mapping
distinguishes mapped-valid, mapped-invalid, and absent cells across complete
and partial cohorts. Original native flag bits are retained, and a nonfinite
value or nonzero original flag is mapped-invalid rather than being converted
into an absent or synthesized sample.

The existing measured matrices remain the value owners: admitted network
inputs and the measured mapping retain shared handles, so Stage 3 introduces
no second O(rows x detectors) value copy. A scan/chunk transaction constructs
the complete mapping, a fresh per-sample revision ledger, and a fresh monotonic
operation sequence before one owner swap. Rejected admission leaves the live
lifecycle pointer-identical; commit, rollback, or boundary destruction clears
all scan-owned mapping, ledger, and sequence state.

The [Stage 3 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE3_2026-08-22.md)
records 6/6 focused ingress cases, 28/28 complete SCI-ALIGN cases, 749/749
runnable CTests, the established single disabled test, 203/203 baseline-tool
tests, the complete required config gate, exact-commit CLI identity, and valid
ledgers. Runtime routing, RTC/PTC, naive/JINC mapmaking, products, and numerical
kernels remain unchanged and inactive. The native-required processing mode
still cannot enter RTC. No Unity run is required for Stage 3; before Stage 4,
the required small owner-reproducible native-gap fixture must be frozen
locally.

That pre-Stage-4 fixture gate is satisfied at exact commit
`6d65b151eb836e2bbd5f5f1d3bf381427800528a`, tree
`406d1784d64d6fb41e567aab083798324f871634`. The frozen
`urn:citlali:sci-align:native-gap:v1` YAML is raw-byte pinned by SHA-256
`a4dfdfe4b45638952f57f5f258badfab84f5d6ce1d022abfefc47a9e84091701`
and has a reusable test-only loader. It contains two interleaved detector
networks, exact zero/large/max-int64 output UIDs, measured values, original
flag bits, five relational slots, and one delivered network-7 packet gap with
the exact `701 -> 703` counter discontinuity. The gap remains one explicit
absent cell; it is not allocated a value or native identity.

The fixture freezes complete-cohort slot intervals `[0,2)` and `[3,5)`, exact
packet-contiguous network runs, and a factor-2 Stage 4 oracle for run-local
selected anchors and bitwise-OR original flag support. Four focused fixture
tests reject any unreviewed byte change and prove that the current Stage 2
alignment contracts materialize exactly that topology. The
[fixture-gate handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE4_FIXTURE_GATE_2026-08-22.md)
records 32/32 complete SCI-ALIGN cases, 753/753 runnable CTests, the established
single disabled test, 203/203 baseline-tool tests, the complete required
config gate, exact-commit CLI identity, and valid ledgers. This test/data-only
commit does not call RTC, alter a numerical route, or activate runtime
processing. The fixture prerequisite is closed and Stage 4 RTC adapter work
may begin under its existing separate stop gate.

Stage 4 RTC contiguous-run dispatch is implemented at exact commit
`23a5cabe9fa6ec6579c91ec7c7a344339d06c993`, tree
`bd2c74f4dde97e3eb9f99b845d2f4741ee458d76`. The adapter derives maximal
complete temporal segments from the Stage 3 measured scan, dispatches each
network's detector partition one packet-contiguous run at a time, and invokes
the existing downsampler separately for every run so its stride anchor resets
at a discontinuity. Each output records ordered exact native support, selected
anchor, detector partition, common slots, and the bitwise OR of actual input
flags. Delivered flagged rows remain temporally present; nonfinite measured
values, cross-run windows, numerical-body shape drift, removed input flag bits,
and nonfinite outputs fail closed.

Five focused cases pass separately at OpenMP thread counts 1, 2, 4, and 8;
the complete SCI-ALIGN executable passes 37/37 cases; and the public header
compiles in isolation. All 758 runnable CTests pass with the established
single disabled test not run, all 203 baseline-tool tests pass, and the full
required config gate passes 127/127 unit tests, all four mode kits, and 8/8
compact-compatibility cases with zero skips or gaps. Both ledgers remain valid,
the frozen fixture digest is unchanged, and the exact implementation-boundary
CLI is `v4.0.0-3672-g23a5cabe9` with binary SHA-256
`88bc483ca7fe9a3ee8a26be73e1505cf6504e3be8391a737be9f0358d412c89a`.

The [Stage 4 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE4_RTC_DISPATCH_2026-08-22.md)
records the complete evidence. This stage does not call `RTCProc::run`, change
an RTC numerical kernel, enter PTC/PCA or mapmaking, publish products, or
activate a production route. RTC product writing remains disabled and the
native-required mode still cannot enter production RTC. No Unity run is
required at this boundary. Stage 5 PTC/PCA cohort gather and transactional
scatter may begin as a separate commit, with its stop gate before mapmaking.

Stage 5 PTC/PCA cohort gather and transactional scatter is implemented at
exact commit `35a61eaaf91722ac7167bfb90d1029f09b4d1df2`, tree
`11b7c62726f3678fb54ef5f0513935dc0d0e0383`. The adapter accepts only a
complete Stage 4 RTC dispatch that exactly matches the admitted scan's run
inventory, raw inputs, detector partitions, run boundaries, ordered support,
and anchors, with internally consistent recorded ORed flags. It builds a
separate detector-level cohort for every contiguous Stage 4 segment, so PCA
cannot bridge the frozen packet gap.

Ordinary `all`, `nw`, `array`, and per-detector groups use exact typed relation
membership, including noncontiguous presentation-ranked columns. Enabled
`corr_nw` delegates subgroup selection to its established grouping body and
preserves ungrouped columns as pass-through. Unsupported group identities and
second-pass/windowed requests fail closed. Excluded cells use checked finite
private placeholders; the established optional-mode compatibility authority
runs before a grouping or cleaner body; and both PCA-invalid and pass-through
cells preserve the exact finite RTC value entering PTC.

The scan/chunk ledger now stages the complete scatter before one swap of
sparse current values and dense revisions. Foreign, stale, duplicate,
identity-changing, shape-changing, or nonfinite results leave both unchanged
and may be corrected and retried under the same issued operation. A successful
scatter advances each affected anchor revision exactly once and records a
monotonic committed operation inside the existing scan/chunk owner.

Six focused Stage 5 cases pass separately at OpenMP thread counts 1, 2, 4,
and 8; the complete SCI-ALIGN executable passes 43/43 cases; and the public
header compiles in isolation. All 764 runnable CTests pass with the established
single disabled test not run, all 203 baseline-tool tests pass, and the full
required config gate passes 127/127 unit tests, all four mode kits, and 8/8
compact-compatibility cases with zero skips or gaps. Both ledgers and all
boundary audits remain valid, and the frozen fixture digest is unchanged.

The [Stage 5 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE5_PTC_COHORTS_2026-08-22.md)
records the complete evidence. This stage does not alter an established RTC
or PTC numerical kernel, enter naive or JINC mapmaking, publish products, add
public `Engine` state, or activate a production route. No Unity run is required
at this boundary. Stage 6 native science pointing and map projection may begin
as a separate commit, with its stop gate before output-lineage claims or mode
activation.

Stage 6 native science pointing and map projection is implemented at exact
commit `d09234f37b2eda851f35106d994be2620e2468bc`, tree
`12137d9701025ac6c3faa6e40eb6f44e1c56ba39`. The immutable projection is
constructible only from the exact Stage 3 measured-scan handle retained by a
successfully committed Stage 5 operation. It freezes every rectangular
consumer cell's exact network-native identity, post-PTC revision and value,
validity state, native latitude/longitude pair, detector binding, and current
map index. Missing cells are never materialized; invalid cells remain present
but flagged and cannot project.

Before either mapmaker can mutate a destination, the adapter revalidates exact
sample values and flags, pixel axes, resolved map grouping, map indices, typed
output UID/array/flag identity, and detector offsets. Stale, foreign,
incomplete, duplicate, unequal, nonfinite, unresolved, or synthetic
candidates fail closed. The opt-in naive and sequential/parallel JINC entry
points then execute their existing accumulation bodies with only the pointing
source selected from the admitted native snapshot. JINC's established unique
map-owner preflight remains in force.

Six focused Stage 6 cases pass, including bit-exact identical-time equivalence
with naive checksum `8052882556844240840`, JINC checksum
`4269599267376700904`, and exact JINC repetition at OpenMP thread counts 1, 2,
4, and 8. The complete SCI-ALIGN executable passes 49/49 cases; all 770
runnable CTests pass with the established single disabled test not run; all
203 baseline-tool tests pass; and the full required config gate passes 127/127
unit tests, all four mode kits, and 8/8 compact-compatibility cases with zero
skips or gaps. Both ledgers and the session-exit audit remain valid, and the
frozen fixture digest is unchanged.

The [Stage 6 handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE6_SCIENCE_PROJECTION_2026-08-22.md)
records the complete evidence. No ordinary runtime route, output lineage,
product publication, or numerical kernel is changed or activated. No Unity
run is required at this boundary.

Stage 7 is now implemented as a local activation candidate at exact commit
`36f6ada25d06f2236dfcd279d53c6afc40298cb1`, tree
`40099545347326aed03df7be22bcc7cfe74e0e7d`. An ordinary Science observation
activates the native-required route only with one complete verified compact-v2
matched relation and its exact native alignment/pointing carriers. That route
is exclusive: it does not run the legacy common-grid RTC/PTC pass first, and
it does not advertise or create legacy RTC/PTC TOD or diagnostic products.
The established RTC/PTC numerical bodies run on admitted native runs/cohorts,
and the established naive or JINC body projects measured cells through the
Stage 6 native pointing snapshot.

The observation-owned `citlali-native-cohort-product-provenance-v2` lineage
binds compact bundle/relation, raw manifest, alignment, pointing, scan
operation, RTC support, PTC groups, revision transitions, detector weights,
eligible map inputs, and product occurrence. JINC occurrences additionally
bind the processing configuration and actual native scan trace. Required
lineage is committed with the scan, and final product-index replacement is
deterministic and atomic after required-product existence checks.

Mode routing remains deliberately asymmetric. Pointing fails closed when
native authority is present because the current low-level identity cannot
distinguish Pointing from OOF. OOF remains inactive. Detector/automatic
Beammap remains the raw/APT producer; non-detector Beammap remains the existing
calibration-table consumer; neither can request matched-consumer lineage.

The candidate domain also fails closed on operations that lack native support
lineage: polarimetry, extinction, RTC kernels, cross-network RTC observers,
line audit, AltAz destriping, nonzero raw/processed detector outlier cuts,
learning, noise maps, TOD output, source-mask radii, PTC second pass, fruit
loops, weight validation, duplicate-tone exclusions, and outer scan context.
Only disabled, naive, or JINC mapmaking is admitted. These restrictions must
be made explicit in the first Unity Science config; they are not silently
discarded.

All 66 SCI-ALIGN cases pass at OpenMP thread counts 1, 2, 4, and 8. All 788
runnable CTests pass with the established single disabled test not run; all
203 baseline-tool tests pass; the full required config gate passes 127/127
unit tests, all four mode kits, and 8/8 compact cases; and both ledgers,
Phase 5 readiness validation, and the 733-dependency session-exit audit pass.
The frozen raw execution census remains exactly 48 records with digest
`efd347b41857542b770de90c9c383a254fbb5a4890988f3b1da43f27de4bcf9f`,
zero review-required entries, and no drift. The exact CLI is
`v4.0.0-3678-g36f6ada25` with local binary SHA-256
`95c53af60db30a353b6bfd8e2badcbb368a16d112fb04f876218183cdab84a7a`.

The [Stage 7 candidate handoff](../handoff/COMPACT_V2_NATIVE_ALIGN_STAGE7_ACTIVATION_CANDIDATE_2026-08-22.md)
records the implementation, bounded config domain, complete local evidence,
and owner checklist. Campaign 1 preparation is frozen at exact package commit
`39138fc24aa762ceb3dda8a471ffe1747f359d1c`, tree
`9191007a1f50f611cba7ee0487d345e00de2edcc`, under
[`validation/campaigns/SCI-ALIGN-STAGE7-UNITY-001/campaign-1-native-gap/`](../validation/campaigns/SCI-ALIGN-STAGE7-UNITY-001/campaign-1-native-gap/README.md).
The campaign-only final overlay selects naive projection, disables every
unsupported operation and the nonzero-context Science FIR, and retains the
supported network-local RTC, PCA, and weighting bodies. Its preflight passes
four focused positive/fail-closed tests and emits the explicit merged config,
numbered-source inventory, override origins, and policy digest. The full
required config gate remains green: 127/127 unit tests, four mode kits, 8/8
compact cases, and zero skips or gaps. Bundle/raw admission, exact native
carriers, zero duplicate tones, and realized zero outer context remain
owner-observation checks and are not prefilled.

Stage 7 is not accepted or production-ready. Campaign 1 is prepared but not
launched. Its next gate is owner selection of one conforming small matched-v2
Science observation, explicit deployed merged-config review, and owner-run
Unity execution. The remaining campaign sequence is identical-time/no-gap
legacy comparison, same-scan naive/JINC, detector or automatic Beammap
producer regression, and non-detector Beammap calibration-lane regression,
all with exact source, binary, config, input, log, index, and retained-product
evidence.

## 2026-08-21 JINC Parallel Ownership Reconstruction

The independently accepted SCI-MAP-002 ownership contract has been
reconstructed on the current convergence line without replaying historical
`jinc_mm.h`. The implementation rejects map-index cardinality mismatches,
invalid or duplicate detector ownership, and incompatible signal, coefficient,
conditioning, coverage, kernel, or noise destinations before contribution
diagnostic allocation, destination mutation, output side effects, or the
parallel launch. The reconstruction adapts the old contract to today's JINC
state by including the absolute-denominator and contributor-count destinations
introduced after the original repair.

The established worker suffix beginning at `grppi::map` is byte-identical to
parent `c67f12120ca29f0c2d603fd551146635ef7b3782` (SHA-256
`5d3030566e7616139c73061f8f7556078a4e2e5b9be504577fc2fa6466309ccf`).
The focused six-case suite passes at 1, 2, 4, and 8 OpenMP threads; all 28
focused current-JINC tests pass; the CLI builds; all 732 runnable CTests pass
with the one established disabled exact-product-sequence test not run; all 203
baseline-tool tests pass; the 127-test required config preflight passes all
four mode kits, 8/8 compact-compatibility cases with zero skips, 100% compact
surface coverage, and every authority audit; and the 60-record validation and
three-change/five-integration-commit science-change ledgers validate.

No Unity rerun is required for this contract-only reconstruction. The accepted
`redu04` run used detector-grouped JINC mapmaking, the admitted domain is the
already governed unique per-detector ownership path, and neither worker
arithmetic nor valid-path output changed. The accepted targeted Unity result at
`e77460cff` therefore remains the science authority beneath this fail-closed
hardening. General JINC promotion remains bounded by the four deliberately
unavailable large FITS products. See the
[reconstruction record](../handoff/JINC_PARALLEL_OWNERSHIP_RECONSTRUCTION_2026-08-21.md).

## 2026-08-21 APT / ALIGN / JINC Convergence Audit

The exact Unity-tested JINC implementation
`e77460cffad49387795009539d6abc7e370e8b58` is now the application authority
for the targeted working-support incident repair. Local convergence branch
`codex/converge-apt-align-jinc` starts at that tested commit and carries the
documentation-only `redu04` validation record at parent
`91f42ccdc8ce9a4e6811f2f03857180d50d21345`. The tested tree is byte-identical
to local repair tree `59a142b334a4d7882f85f031ba090cdd74171839`;
the local cherry-pick history remains an equivalence backup and is not another
replay source.

The frozen [convergence audit](../handoff/APT_ALIGN_JINC_CONVERGENCE_AUDIT_2026-08-21.md)
accounts for all 76 commits on the historical SCI-ALIGN Lissajous line plus
the relevant side branches. None of the 76 commits is eligible for direct
application replay: ten touch application sources but remain an unaccepted
old-base repair, diagnostic-only mapmaking/configuration, or incomplete PTC
product repair; the other 66 are diagnostic tooling, generated evidence,
campaign/transport mechanics, or an unrelated handoff.

The audit also corrects the ancestry implied by merge subject `a71fce419`.
That merge contains compact APT v2 plus SCI-ALIGN native-cohort foundation
`c87d5693d`, but not consumer commits `fd3627fc7` and `9d9d55a54`. Those
patches are based on the superseded canonical APT v1 lineage and have no
independent exact-SHA acceptance record, so they must be reconstructed against
compact v2 rather than cherry-picked. Independently re-audited JINC ownership
repair `e6c8d1261` has now been cleanly reconstructed and locally validated on
the convergence line. PTC metadata repairs `7fc59344c` and `5c6309125` remain
a separate conditional lane, with the latter's fresh-root Unity replay still
pending.

The frozen audit itself changed no application code, production status, remote
ref, or general JINC/ALIGN/APT acceptance. Its first proposed reconstruction
has since passed the governed local gates above; broader application
integration remains blocked on the compact-v2 native-consumer reconstruction,
its independent review, and owner-controlled push.

## 2026-08-21 JINC Working-Support Incident Repair — Targeted Unity Validation Passed

The owner-provided Unity validation Beammap for observation 148670 completed
under JINC after the inactive-notch provenance fix but failed scientific
validation. The controlled comparison held the observation and merged
configuration fixed: current naive reduction `redu02` retained 4,829 good
detectors and had median map RMS `1.80e-7`, while current JINC reduction
`redu03` retained 462 good detectors, rejected 4,754 for signal-to-noise, and
had median map RMS `4.12e-6`. The byte-identical older JINC configuration at
`cfae989c` retained 4,973 good detectors. This exact application snapshot is
therefore rejected for JINC scientific use but remains diagnostic evidence.

The bounded incident repair is prepared on
`codex/repair-jinc-working-support` from exact failed application base
`a71fce4198769a88c6c0c85fc035ec3496ccbe03` plus the retained inactive-notch
hotfix `5ef2d011660d7d7d3e17e4a30874003f713746b5`. It leaves JINC accumulation,
signed `N/C`, formal `C^2/Q`, dimensionless cancellation conditioning, cache
geometry, phase sampling, and the ordinary mapmaking path unchanged. After
formal finalization it now applies the already governed `coverage_cut / 10`
normalization-support rule to the formal coefficient plane as a downgrade
only, atomically clearing rejected signal, coefficient, coverage, kernel,
noise, and persisted support pixels. A zero cut retains all formally supported
pixels. `MapBuffer::cov_cut` also receives a deterministic zero default for
directly constructed test and library buffers; configured reductions continue
to supply their requested value.

The focused production regression proves that a denominator which passes the
formal `2*gamma_n` resolution rule but would realize `1e10`-scale working-map
values is rejected by the empirical support floor, while seven ordinary pixels
remain exactly unchanged. All 22 focused JINC contract tests pass. The CLI and
all separate test targets build; 726/726 enabled CTests pass with the one
pre-existing disabled exact-product-sequence test not run. The complete
baseline-tool suite passes 203 tests plus 137 subtests. The required config
preflight passes 127 tests, all four mode kits, 8/8 compact-compatibility cases
with zero skips, 100% compact-surface coverage, and every authority audit. The
inactive-notch hotfix's new `run_tod_notch` shadow comparison is now explicitly
classified in the raw-execution census: 48 records, digest
`efd347b41857542b770de90c9c383a254fbb5a4890988f3b1da43f27de4bcf9f`, zero
review-required entries, and no drift.

The owner-run Unity rerun `redu04` passed that targeted incident gate with
Citlali `v4.0.0-3657-ge77460cf`, KIDs `04088da`, and Tula `f30f81d`. Its
requested and merged configuration hashes are byte-identical to failed JINC
run `redu03` (`d81ac8b1...` and `5035d010...`, respectively). The repaired run
retained 4,973 good detectors, rejected 181 for signal-to-noise, had median map
RMS `1.036e-7`, built all three empirical templates from 500 detectors each,
used template calibration for 5,133 of 5,234 detectors, found 494 reference
candidates, and bounded the final kernel peak at `7.798`. The failed run had
462 good detectors, 4,754 signal-to-noise rejections, median map RMS
`4.124e-6`, no empirical templates, 51 reference candidates, and final kernel
peak `1.403e4`. The healthy naive comparison retained 4,829 detectors with
median map RMS `1.800e-7`; the repaired detector yield also exactly restores
the older controlled JINC result's 4,973 detectors. The `redu04` log contains
zero error- or critical-level records and completes normally. Working-support
downgrades were active on 7,810,611, 9,901,143, and 9,859,003 pixels in the
three iterations, directly connecting the repaired behavior to the intended
support-floor mechanism.

This is accepted targeted Unity evidence that the JINC working-support defect
is repaired for observation 148670. It is not yet a complete validation-ledger
product snapshot: the local retrieval intentionally omits four indexed large
FITS products (all three good-detector cubes and the a1100 bad-detector cube),
so strict full-product comparison and general JINC promotion remain pending.
The retained small products, log, configuration snapshots, provenance, fit-QC
table, and compact APT v2 bundle are sufficient for the incident verdict and
are inventoried in
`handoff/JINC_WORKING_SUPPORT_UNITY_VALIDATION_2026-08-21.md`. No push,
production authorization, or unrelated JINC numerical broadening is performed
or implied.

## 2026-08-19 OWNER PRIORITY: APT-PROD-003 compact v2

APT-PROD-001/002 v1 histories remain immutable evidence, but new v1 issuance
and ordinary v1 admission are stopped. The valid 148669-to-148670 v1 product
is 249,525,124 bytes; 233,948,454 bytes are 261,700 repeated per-cell
transformation records for a 3,541,490-byte scientific table. The owner has
classified this as a contract-design defect.

The active APT-only repair is [compact v2](CANONICAL_APT_V2.md), a normalized
content-addressed ECSV bundle with one root manifest/receipt. All APT-dependent
baseline generation, pointing validation, consumer conformance, and scientific
acceptance remain suspended until the exact 148669/148670 equivalence, size,
determinism, relocation, tamper, guardian, and publication gates pass. No
ALIGN, JINC, map, RTC/PTC, CAL, TolTECA, Unity, or production work is activated
by this status.

This is the living roadmap and completion ledger for the Citlali refactor.
Update it when a phase gate, governing decision, or validated snapshot changes.

The current implementation checkpoint provides the compact v2
model, deterministic ECSV component codecs, content-addressed root bundle,
receipt-last no-replace publisher, fresh Beammap baseline adapter, matched-v2
consumer guardian, and read-only `validate-bundle-v2` /
`describe-baseline-v2` public protocol. New v1 issuance is mechanically
disabled. The TolProj-orchestrated `issue-observation-apt-v2` integration
candidate is active behind its closed request and verification boundary; the
standalone target canonicalizer and v1 migration command remain fail-closed.
This checkpoint authorizes only the controlled Stage 7 campaign documented in
`CANONICAL_APT_V2.md`, not general APT-library activation or scientific
acceptance.

## Governing Decision

On 2026-07-10 the project formally adopted the five-phase roadmap from the
[independent architecture review](../handoff/EXTERNAL_REFACTOR_ARCHITECTURE_REVIEW_2026-07-10.md).
The review verdict, **sound with material reservations**, is accepted.

The original [structural refactor plan](STRUCTURAL_REFACTOR_PLAN_2026-06-29.md)
remains the historical statement of intent. This document governs current
sequencing and exit criteria where the original plan differs.

The project will improve the existing tree incrementally. It will not restart
as a broad rewrite and will not rewrite the granular history of the validated
branch. The exact validated tree will remain available for forensic review.

## Current Integration Model

As of 2026-07-31, `codex/refactor-mainline` is the canonical application
integration branch. Continued scientific, diagnostic, configuration, and
operational development remains active there. The previous
`codex/structural-refactor` and `codex/fruit-loop-calibration-reference`
branches are retained as historical pointers rather than competing
application authorities.

The successor build proceeds independently on `codex/conan2-adaptation` in a
separate worktree created from the application mainline. It incorporates
mainline regularly and may return only after the bounded Adapt gates pass. It
does not wholesale-merge `citlali/v4.x_conan2`, replace the refactored
application, or mix numerical algorithm changes into build integration.

The live branch, upstream revision, gate, and import policy are recorded in
[`INTEGRATION_LEDGER.md`](INTEGRATION_LEDGER.md). The durable rationale is
[ADR 0008](adr/0008-application-mainline-and-build-adaptation-lanes.md).

## 2026-08-14 APT-PROD-001 Canonical Baseline APT v1 Candidate

The project owner accepted the frozen APT-E2E-001 audit at
`6cf83a21169516303db1fa30d26f4be32a813844` as architectural authority and
authorized the bounded Citlali-only producer package on
`codex/repair-apt-prod-001-canonical-baseline-v1`, created from exact
application base `46ad23888a40f5102cdfd50c06e49a549bdf8a20`.

The candidate defines the accepted `citlali-canonical-apt-v1` artifact model,
typed canonical ECSV codec, exact built-in field registry, order-independent
semantic SHA-256, occurrence-bound envelope SHA-256, separate exact-byte
transport SHA-256, raw manifest and complete `uid -> (network, channel)`
relation, executable artifact validator seam, and receipt-last no-replace
publication protocol. The Beammap adapter preserves the existing detector set,
row order, ToneFreq bits, exact integral values, and scientific table values;
any drift remains a stop condition rather than an allowed schema change.

The accepted identity decision is narrow: `uid` is a unique nonnegative exact
`int64` artifact-local row key in `0..2^53-1`, sparse permitted, and never a
persistent detector identity. Persistent measured-detector identity and tune
identity are omitted. Required nullable `fg`, `pg`, `ori`, and `loc` remain
nonidentity semantic content under explicit unresolved authority. Optional
`kids_flag` preserves the exact signed KIDs fit-report flag under declared
`kids:fit-report-v1` authority and remains distinct from `flag` and `flag2`;
simulation omits it when no fit report exists.

The candidate artifact contract in `validation/product_contracts.json` remains
explicitly `unactivated`. It does not amend an active validation profile,
activate downstream ingestion, migrate or repair historical APTs, establish
CAL physical-science closure, or change matcher, calibration, fit, map, RTC,
PTC, or detector-selection policy. Historical APTs remain historical/test-only
for this producer. TolTECA, TolProj, TolAPT, `toltec_beammap`, CAL, and ALIGN
remain outside this package.

The normative contract is
[`CANONICAL_APT_V1.md`](CANONICAL_APT_V1.md), with durable rationale in
[ADR 0010](adr/0010-canonical-baseline-apt-v1.md). The coherent producer
candidate was committed and pushed as
`d4a808c59f383a5f77059b83083af2a69802a12a`, with parent
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`, tree
`f77150abe863de73585d37a91485ea0e8a1951d0`, and full-index binary patch
SHA-256 `ab40312b4de4844e0fcc5d7bd646787a83e4e1b7dbbc00002911f7493d385ffd`.
That accepted producer artifact remains unactivated; push of the bounded
candidate did not authorize application-mainline integration or downstream
admission.

## 2026-08-14 APT-PROD-002 Observation-Specific APT v1 Candidate

APT-PROD-002 is a bounded Citlali-only successor to the pushed APT-PROD-001
producer. Its branch `codex/repair-apt-prod-002-observation-contract` starts at
exact baseline commit `d4a808c59f383a5f77059b83083af2a69802a12a`; the frozen
APT-E2E-001 audit `6cf83a21169516303db1fa30d26f4be32a813844` remains the
governing architecture authority. The independently accepted Phase-B
checkpoint changed exactly twelve authorized implementation, contract, and
test paths, with full-index binary patch SHA-256
`8f452e9775a5a74b688ef3766ec31ae327e80ba50359953eebb436a006114cb8`.

The candidate closes the row-position correspondence defect without inventing
a persistent detector namespace. A verified immutable Beammap baseline row is
referenced by its baseline occurrence and artifact-local `uid`; target,
relation, and output local keys are scoped to their own opaque occurrences.
Citlali reconstructs digest-bearing references from those scoped facts. Source,
application, seed-source, and output-presentation sequences remain explicit
complete permutations and are never identity.

The accepted persisted product is deliberately small: exactly one
observation-specific canonical APT-family ECSV and its adjacent
envelope-bound completion receipt. The complete target manifest and generalized
match relation remain typed logical records embedded in the final ECSV's
normative metadata and row evidence; they are not separately published files.
The relation retains every target matched/unmatched disposition, every baseline
seed matched/unused disposition, complete pair sets, network/matcher evidence,
and per-field transformation lineage. Unmatched targets have no fabricated
seed endpoint and use the contract's typed-missing transformation state.

JSON is used only by the explicit versioned machine request/response protocol.
It is not an APT scientific-data encoding. The protocol describes and verifies
the immutable baseline, accepts TolProj-issued observation and realized-match
facts as occurrence-scoped values, and lets Citlali own the fixed schemas,
canonicalization, identities, output-local UIDs, ECSV bytes, validation,
receipt, and no-replace publication. It does not implement or change matcher
policy.

The observation-specific KMP authority is closed to required `kids_fr`,
`kids_f_out`, and `kids_Qr`, plus artifact-optional `kids_flag`. Other
`kids_*` diagnostics remain transitively integrity-bound by their selected
source artifact SHA-256 and byte count but acquire no canonical field,
identity, matcher, transformation, output, unit, or authority meaning. Adding
one requires a separately reviewed field-specific successor authority.

The final publisher stages, rereads, recomputes, and validates the ECSV, refuses
replacement, publishes the artifact first, and makes the receipt visible last.
The accepted limitations remain explicit: the protocol is not fsync/crash-
durable before receipt; stdout failure after a successful receipt can yield a
false-negative acknowledgement recoverable through the validate operation;
and stdin has no owner-specified absolute size quota. These limitations do not
authorize policy accommodation in this package.

All three executable artifact-contract entries remain `unactivated`. This
candidate changes no validation profile, accepted run, ingestion path, CAL,
ALIGN, Beammap numerics, detector membership/order, TolProj repository, or
production state. The normative contract is
[`CANONICAL_APT_OBSERVATION_V1.md`](CANONICAL_APT_OBSERVATION_V1.md), with
durable rationale in
[ADR 0011](adr/0011-canonical-observation-apt-contract.md). The containing
coherent candidate commit is eligible for owner push only after the complete
Phase-C broad and retained gates pass; this record does not authorize that
push, downstream launch, or production use.

## 2026-08-05 SCI-MAP-001 Application Integration Candidate

The final independent re-audit at
`8fc716557ca78b0d220200a92be46fa3545797e9` and the final canonical
coordination candidate at
`c7bb0214edfd57fddf31165923f08784dfd1b8c9` accept the bounded
`SCI-MAP-001` scientific contract at exact application source
`af0c849ce59a5f80e5efc8db435bb6662863052f`. Within that scope the contract
is approved, the implementation is conformant, validation is complete within
the local plus owner-accepted bounded evidence scope, and the bounded verdict
is `accept`.

F001--F011 are closed. F012 is
`closed_bounded_owner_accepted` only for the exact-`ed28dafb` external
execution/completion, returned product/inventory, visible observation/coadd,
and SEQ/OMP claims. Its retained limitations are the absent independent raw
manifest and sample ledger, scan-farm pre-normalization planes and commit-order
trace, wrapper/Slurm/environment/collection/retrieval chain, and historical
same-case S-X observation-realization files. F013 remains `open_conditioned`
on `SCI-ALIGN-001`, `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and
`SCI-VAL-001`.

The coordinator-directed 2026-08-05 application-integration task separately
authorizes this MAP candidate for application integration. The dedicated
`codex/integrate-sci-map-001` branch was created from exact canonical
application base `9aae0e669384c5c0c0dda93debc194d6b8dac787` and advanced only
by fast-forward through `ed28dafb3`, `1b824f138`, `02b9eb303`, `f84b9fd7d`,
and `af0c849ce`. The excluded convolve/noise candidate
`02a198cbfb379eaf6ab279c5a3d44ee73ff90435` is not in that ancestry. Before
the integration records were edited, the branch tree was exactly the
`af0c849` application tree, `47aa745554e47514398e72d579625484abdcb79e`.
The branch-tip child of `af0c849` is a documentation-only integration commit
that changes this status, the integration ledger, and the dated
[application-integration handoff](../handoff/SCI-MAP-001_APPLICATION_INTEGRATION_DECISION_2026-08-05.md);
it is not a later application-source revision.

The owner subsequently fast-forwarded the verified MAP integration candidate
onto `codex/refactor-mainline`; its documentation-only integration tip is
`d5015fe716971bf8ea617e8a187311bf5af05185`, while exact accepted MAP
application source remains `af0c849ce59a5f80e5efc8db435bb6662863052f`.
Production remains `existing_use_only`, no upstream dependency is closed by
MAP acceptance, and neither production expansion nor Conan-lane import was
authorized by that integration.

## 2026-08-08 SCI-NOI-002 Application Integration

The project owner and coordinator authorize the bounded SCI-NOI-002
application candidate for integration. Current application mainline
`d5015fe716971bf8ea617e8a187311bf5af05185` is the direct merge-base and
ancestor of accepted candidate
`5b29e13548a6fec884c67b192dec20c92f0bbb62`; the candidate is exactly six
commits ahead with no mainline divergence. The dedicated
`codex/integrate-sci-noi-002` branch was therefore advanced by fast-forward,
without conflict resolution, patch reconstruction, or audit/coordination
history. Before the integration records were edited, its tree was the exact
audited candidate tree `641c724f40a9fa9f322f09c703705239439d2374`.

Independent Cycle 4 re-audit
`6de648f5ae2b37f5bc65162feae221f19bb84a5a` and canonical coordinator
closeout `d03ef80b31f704859ef836e368801dc17d92e76e` establish the bounded axes
`approved`, `conformant`, `complete`, and `existing_use_only`, with controlled
verdict `retain`. C4-R001--C4-R004 and Cycle 3 P1-001--P1-003 close. F001,
F002, F003, F004, F007, and F008 are closed within their exact contracts.
F005/RA-B004 remain `open_conditioned` under SCI-FLT-001, and F006 remains
`held_external` under SCI-FRUIT-001.

The exact candidate passed four of four required Release targets, 40/40
focused core tests, 2/2 focused and 32/32 full science-product tests, 127
applicable Python tests, 623/623 runnable CTests, and the complete 127-test
configuration preflight with no required skip or gap. The accepted paired
C++/Python matrix covers compact ECSV/NetCDF missingness, actual coadd
membership/count truth, preservation of configured standalone realization
files, and split Beammap zero/one/multiple logical-map packages. It found no
estimator, normalization, realization-generation/sign, count/default,
mapmaking/filter, output-selection/layout, physical-variance, or significance
scope expansion.

No astronomical reduction was required or claimed for this bounded closure.
The current intended-science-change ledger requires every entry to cite an
accepted reduction; it is deliberately unchanged rather than weakened or
bound to an unrelated historical run. The exact audited
`validation/product_contracts.json`, independent report/result/proposal, and
canonical closeout remain the authority for the externally visible product-
contract and schema corrections. Admitting deterministic audit evidence into
that separate ledger would require a future framework decision.

The documentation-only child of `5b29e135...` changes only this status, the
integration ledger, and the dated
[application-integration decision](../handoff/SCI-NOI-002_APPLICATION_INTEGRATION_DECISION_2026-08-08.md).
It does not modify the accepted application bytes. The owner fast-forwarded
this exact integration tip,
`4846fa4db39bd2f7d4ddc41f693836834cbc5ff4`, onto
`codex/refactor-mainline`; the exact accepted application source remains
`5b29e13548a6fec884c67b192dec20c92f0bbb62`. Production expansion, F005/F006
work, Unity execution, realization-count recommendations, calibrated
significance, and Conan-lane synchronization remain separate decisions.

## 2026-08-10 SCI-MAP-002 Application Integration

The project owner and coordinator authorize the owner-accepted SCI-MAP-002
third-successor application candidate for bounded local integration. The
dedicated `codex/integrate-sci-map-002` branch was created at exact canonical
application target `46ad23888a40f5102cdfd50c06e49a549bdf8a20` and merged exact
candidate `86f1582fad92bdd0453bca3264ce39478b00c227` with a no-fast-forward
merge. Local merge commit `214484e21a00e0c11d86c2b0460ec98b969469f2`
has those exact target and candidate commits as its first and second parents,
respectively, and its tree is the accepted candidate tree
`f655e96daa578bd77c9b16528c3aaadf882ee80d`. The complete four-commit MAP
repair lineage is preserved without conflict resolution, patch
reconstruction, squashing, rewriting, or application-byte alteration.

Independent re-audit
`a70424a69365d7ed20fb39c45bc6334cc9e7bafe`, report SHA-256
`fe07901504ad26916cab2c5589f452b1873c11ef215239653ed938f45eefd4d5`,
and canonical owner acceptance `3625bc1946910d8d6f13e82fa03f5815112d67a1`
close RA-001 as preserved, RA-002, RA-003 as preserved, RA-004 as preserved,
and RA-005. The accepted axes are `approved`, `conformant`, `complete`, and
`existing_use_only`, with verdict `accept`. The 43 imported application paths
are exactly the accepted candidate inventory. RTC-, PTC-, and Beammap-named
touches remain MAP processing-provenance and application wiring; they do not
authorize or introduce a separate CAL, RTC, PTC, Beammap, or unrelated
scientific change.

The exact integrated application bytes pass the six requested focused and
application build targets plus the complete default build; 75/75 focused JINC
and science-map FITS CTests; and all 666 enabled CTests among 667 discovered
tests, with the one unchanged disabled exact-product-sequence test not run.
The complete baseline-tool suite passes 173/173 tests, including 24/24 direct
product-contract tests. The required config preflight passes 127/127 tests,
all four mode kits, 8/8 compact-compatibility cases with zero skips, 100%
compact-surface coverage, and every typed authority audit. The raw-execution
census is 47 records with exact digest
`37bbb9c4a1a7ed78e3d79571a4cd6e0e745af2520eddb24f54ca191d52d4d1bf`,
zero review-required entries, and no drift. The 60-record validation ledger,
three-change/five-integration-commit science-change ledger, four-active/eight-
preparing validation-profile registry, product-contract JSON registry, and
session-exit audit with zero library exits, CLI exits, or growth all pass.
There are no required-data skips or unexpected error-level messages in the
passing gates.

This bounded local integration does not push or remotely merge the branch,
authorize Unity access or a reduction, expand production, launch BEAM or
other downstream work, start a re-audit, alter the accepted scientific
contract, or synchronize the Conan lane. Owner-controlled push and any later
application-mainline update remain separate actions.

## 2026-08-08 SCI-MAP-002 Local JINC Repair Candidate

The bounded SCI-MAP-002 application repair was prepared on
`codex/repair-sci-map-002` directly from exact canonical application mainline
`46ad23888a40f5102cdfd50c06e49a549bdf8a20`. That base contains accepted
SCI-MAP-001 application source and the integrated SCI-NOI-002 product,
identity, writer/finalizer, atomic-publication, and provenance seams. Frozen
coordination authority is Git object
`dd5894679bf12bf4a5fb551e871b3c6010ef9b9b` on
`codex/scientific-audit-framework`; its bounded handoff, coordinator review,
eight owner decisions, corrected scientific-contract audit, local evidence,
ledger proposal, and applicable handoffs govern this candidate.

The candidate preserves signed finite JINC lobes, phase-quantized point
sampling, the fully populated square cache, and the established accumulators
`N = sum(q_i c_i d_i)`, `C = sum(q_i c_i)`, and
`Q = sum(q_i c_i^2)`. It finalizes signal as `N/C` and the distinct formal
mapmaker weight as `C^2/Q`; replaces unit-bearing absolute conditioning gates
with finite-state, positive-`Q`, exact-cancellation, and documented
dimensionless realized-rho checks; and fails invalid selected-array identity or
JINC parameter domains before deposition or publication. Formal support is
authoritative, empirical policy may only downgrade it, coverage remains the
formal-support-only coefficient-squared integration-time sum in seconds, and
the kernel product is the realized processing-filtered source-template
response finalized as `K/C`.

One compact atomic mapmaking-provenance record per coherent observation or
declared processing segment carries requested/effective digests, resolved
array identities, realized numeric policy, and immutable product/HDU/digest
joins. No per-sample, per-detector, or per-pixel provenance is added. The
existing v3 provenance identity is retained because the new state is an
additive realization record rather than a format break.

The exact local tree passes all five required build targets and 634/634
enabled Citlali CTests (one pre-existing disabled test). The focused fixtures
cover signed direct equations; square edge, corner, and map-edge cropping with
no radial predicate; phase-bin boundaries and bounded point-phase refinement;
cancellation/rho, unit rescaling, finite-range and invalid-`Q` behavior;
selected-array and parameter-domain admission; formal/empirical support;
coverage seconds; kernel response; compact provenance joins; all-valid
no-broadening; and sequential/concurrent agreement under the declared numeric
policy. The complete baseline-tool suite passes 173/173 tests. The required
config preflight passes 127/127 tests, all four mode kits, 8/8 compact
compatibility cases with zero skips, 100% compact-surface coverage, and all
typed boundary audits. The raw-execution census remains the exact canonical
45-record digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`
with zero review-required entries.

This local candidate does not authorize Unity access, a reduction, evidence
collection, re-audit, production-status change, parameter campaign, or
algorithm broadening. Unity evidence and a fresh independent re-audit remain
external gates. The provisional processed-PTC diagnostic-writer issue is
outside this frozen repair, is unchanged, and is not used as positive JINC
validation evidence.

## 2026-08-08 SCI-MAP-002 Successor Repair Candidate

The owner-authorized successor repair is prepared on
`codex/repair-sci-map-002-successor` from exact rejected application candidate
`854a04b124e083e64706fd043e105182fee568af`. It retains that candidate's
approved JINC estimator, signed-lobe, square-support, conditioning, response,
coverage, and compact-provenance semantics while addressing only re-audit
RA-001--RA-005.

RA-001 resets `N`, `C`, `Q`, `sum(abs(q*c))`, contributor count, and formal
support as one iteration-owned state for the same Beammap active-map subset.
Per-map realization summaries persist across active-subset passes and are
reaggregated into coherent observation totals; joins are invalidated for each
new realization pass and regenerated by the writer. RA-002 replaces generic
provenance labels with compact exact digests of the realized upstream kernel
template and enabled processing operators, including loaded kernel images,
source-center state, FIR coefficients, configured notches, IIR settings,
typed raw/processed activation, PTC cleaner settings, and realized scan and
dynamic-notch counts. Coverage records the
finite-positive effective processed-timestream sample frequency used by
`sum(c^2/f_s)`. No per-sample, per-detector, or per-pixel provenance payload
is added. RA-004 applies strict-positive JINC parameter admission only when
JINC mapmaking is selected, preserving inactive-JINC behavior for naive
mapmaking.

The RA-003/RA-005 deterministic matrix now exercises both production JINC
population paths under sequential and OpenMP policies, active-subset atomic
reset and multi-pass summary coherence, below/equal/above-`r_max` response,
failure-before-cache-admission for a generated non-finite coefficient,
formal-support finalization, analytic-zero and coverage-rate behavior, compact
actual-identity serialization, immutable joins, and observation-state failure
suppression when joins are absent. Neighboring science-map and FITS writer
tests continue to cover required writer finalization and failure-before-write
boundaries.

The successor passes the `citlali_cli` build; 18/18 focused JINC tests; all
660 enabled CTests with the one pre-existing disabled exact-product-sequence
test not run; the complete 173/173 baseline-tool suite; and the complete
127/127 config preflight, four mode kits, 8/8 compact-compatibility cases,
100% compact-surface coverage, and all typed boundary audits. The raw
execution census remains the canonical 45-record digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`
with no review-required entries.

No Citlali reduction, Unity access, push, re-audit, BEAM/downstream launch,
production-status change, merge, or external contact is performed or implied.
A fresh independent re-audit remains a proposed, unexecuted next handoff.

## 2026-08-09 SCI-MAP-002 Second-Successor Repair Candidate

The owner-authorized second-successor repair is prepared on
`codex/repair-sci-map-002-successor-2` from exact application commit
`6c74d214a49af5520f02ca071b5d513b14b58b03`. The approved SCI-MAP-002 JINC
scientific contract is unchanged. Closed RA-001 active-subset iterative
reset/finalization behavior and closed RA-004 selected-JINC-only positivity
behavior are preserved; this candidate changes only RA-002, RA-003, and the
nonduplicative local RA-005 seams.

RA-002 now binds processing configuration only after observation setup has
constructed the actual kernel, FIR, configured-notch, IIR, PTC-cleaner, edge-
guard, sample-frequency, and population-topology state. Processing realization
binds only after successful raw execution. Compact per-scan traces are reduced
to ordered identities and counts for RTC/PTC flags, APT flags, map routing,
source and mean masks, processed signal/kernel state, PCA eigensystems and
applied cuts, configured/dynamic/per-detector notches, completed scans,
detector topology, outer policy, kernel template, and the exact FITS product
joins. Successful provenance rejects any required unavailable identity or
missing binding; no per-sample, per-detector, or per-pixel provenance payload
is serialized.

RA-003 and local RA-005 evidence now enter through the production Beammap
population dispatcher and production FITS writer. The deterministic matrix
covers two scans, two maps, two detectors, ordinary sequential/OpenMP outer
policies, detector grouping, a populated second active-subset pass, retained
inactive-map state, exact writer joins, post-raw binding order, and required-
output suppression before the first HDU when processing provenance is
incomplete. Existing signed-lobe, support, coverage, kernel, RA-001, and
RA-004 tests remain unchanged and pass.

The candidate builds `citlali_cli`; passes 19/19 focused JINC tests, 34/34
production science-map FITS tests, and 87/87 focused affected CTests; and
passes all 663 enabled CTests among 664 discovered tests, with the one
pre-existing disabled exact-product-sequence test not run. The complete
baseline-tool suite passes 173/173 tests. The full config preflight passes
127/127 tests, all four mode kits, 8/8 compact-compatibility cases, 100%
compact-surface coverage, and all typed boundary audits. The authorized raw-
execution census is 47 records with digest
`9c1633f362fa0534ea1b9f66cba6122fcec3b299aefe59504f19116de61900fb`
and zero review-required entries. The 60-record validation ledger and
three-change/five-integration-commit science-change ledger validate, and the
session-exit audit reports zero library/CLI exits and zero growth.

Phase 5 readiness remains `preparing` and not promotion-ready for its existing
external same-SHA, build-integration, and accepted-successor-baseline blockers.
No Citlali reduction, Unity access, push, re-audit, BEAM/downstream launch,
production authorization, merge, or external contact is performed or implied.

## 2026-08-09 SCI-MAP-002 Third-Successor Repair Candidate

The owner-authorized third-successor repair is prepared on
`codex/repair-sci-map-002-successor-3` from exact application commit
`02f443bfeb85f3b2e12a6eff60f3a77e77fe342c`. The accepted second-successor
re-audit record `550e677fbe3eb9777187acb53d816a55961a3511` is evidence only
and is not part of this application branch. This candidate is limited to the
two remaining RA-002 truthfulness defects and the nonduplicative RA-005
required-writer failure/serialization seam. Closed RA-001, RA-003, and RA-004
behavior and all approved scientific arithmetic, support, coverage, products,
and configuration policy are unchanged.

RA-002 now digests the exact realized source-protection mask and records every
successfully applied RTC notch operator in execution order with stage,
detector identity where applicable, center, width, and phase policy. The
compact observation identity includes the exact configured, fixed, shared-
dynamic, and per-detector operator sequence and truthful actual counts; a
successful realization rejects an unavailable required identity. Configured
provenance also identifies configured-notch widths and the fixed line-audit
center/width vectors. PTC PCA removal returns its final limit after forced-
index selection, standard-deviation selection, and clamping, and that returned
limit is published as `applied_cut` while configured/requested state remains
separate. No filter, mask, notch, PCA, or estimator arithmetic is changed.

The RA-005 seam uses the production required-output publication helper to
prove that a complete processing realization can be written, completed at the
observation boundary, and serialized with the exact joins generated by that
writer. A second complete-provenance attempt fails inside the required writer
after output work has begun and proves that the completion callback, JINC
observation state, product joins, and serialized successful publication remain
absent.

The candidate builds the focused JINC/science-map products and `citlali_cli`,
`citlali_test`, `citlali_science_map_truth_test`, and `citlali_safety_test`.
It passes 74/74 focused tests, 308/308 affected CTests, and all 666 enabled
CTests among 667 discovered tests; the one pre-existing disabled exact-product-
sequence test is not run. The baseline-tool suite passes 173/173 tests. The
full config preflight passes 127/127 tests, all four mode kits, 8/8 compact-
compatibility cases, 100% compact-surface coverage, and every typed boundary
audit. The raw-execution census is 47 records with digest
`37bbb9c4a1a7ed78e3d79571a4cd6e0e745af2520eddb24f54ca191d52d4d1bf`,
zero review-required entries, and no drift. The 60-record validation ledger,
three-change/five-integration-commit science-change ledger, four-active/eight-
preparing validation-profile registry, product-contract registry, session-exit
audit, and diff checks pass.

No Citlali reduction, Unity access, push, merge, re-audit, external contact,
production authorization, or BEAM/downstream launch is performed or implied.
Owner push, coordinator identity verification, and a separately authorized
fresh independent re-audit remain external follow-up dependencies.

## Historical 2026-07-31 SCI-MAP-001 Bounded Repair Lane

This section preserves the repair-lane chronology and its candidate-time gate
states. Statements below about a then-pending final re-audit are historical
evidence, not current instructions or current package status; the application-
integration disposition above supersedes them as live state.

The project owner approved a bounded repair of `SCI-MAP-001` findings
F001-F011 on `codex/repair-sci-map-001`, created directly from governing
application source `9aae0e669384c5c0c0dda93debc194d6b8dac787`. The audit and
coordination lines remain read-only authorities. The convolve/noise candidate
`02a198cbfb379eaf6ab279c5a3d44ee73ff90435` is deliberately excluded and does
not land first.

The repair scope is the accepted ordinary-naive, array-grouped Stokes-I
successor contract: contract-derived fixtures; typed exposure, count, support,
and validity state; atomic full-precision map-bundle admission followed by one
admission commit phase; centered integer common-grid embedding with `L = I`;
preservation of the existing `Q += u`, `N += u * signal`, and
`K += u * kernel` operation
order; nonprecision coefficient labeling; the eight distinct F010 products,
compatibility aliases, explicit absence rules, and lossless realized
provenance. [ADR 0009](adr/0009-science-map-bundle-admission-and-validity.md)
and [the scientific conventions](SCIENTIFIC_CONVENTIONS.md) record the durable
meaning.

The candidate records the closed pre/post observation/coadd coefficient
stages, freezes a validated raw F010 snapshot before filtering, and carries
that immutable input through filtered signal, coefficient, F010, and alias
HDUs with matching lossless `RAWPDGST` identity. Unsupported JINC,
detector-grouped, and other non-v1 profiles retain their established legacy
coadd arithmetic with explicit successor-product absence and no F009/F010
claim.

The ordinary primitive uses one detector/sample order for sequential and
requested-parallel calls. Concurrent scan commits are serialized and governed
by `within-scan-exact-scan-farm-2gamma-n-sumabs-v1`: binary64 planes are tested
against long-double per-scan sums at the pre-registered
`2 * gamma_n * sum(abs(scan_value))` bound, while integer fact planes are
exact. The raw-execution read census is advanced only for the new immutable
science identity's existing kernel and separate-polarimetry state reads; it
adds no raw configuration authority.

The local candidate snapshot passes the required implementation gates without
required-data skips or unexpected error-level records: all five requested
build targets complete; CTest executes 588/588 enabled tests successfully
(one pre-existing disabled test); the focused science-map executable passes
29/29 contract, provenance, and equation tests; its ThreadSanitizer build
passes 7/7 repaired-primitive tests; 147 baseline-tool tests pass; and the full
config preflight passes 127 unit tests, all four mode kits, all eight compact
compatibility cases, 100% compact-surface coverage, and every typed boundary
audit. The classified raw-execution census remains 45 records with zero
review-required entries at digest
`09572da976aec89d56506394420b478426a6efbd0942c864571a8f6f311da2f8`.
The successor validation epoch and product registry parse and list cleanly.
Those initial local repair-candidate results did not themselves satisfy F012
or the independent re-audit.

This lane does not authorize general reprojection, interpolation, GLS,
covariance regularization, new defaults, or changes to RTC, PTC, JINC,
noise-realization, convolve, Wiener, source-fitting, Pointing/OOF, Beammap, or
fruit-loop algorithms. Historical accepted products and profiles retain their
original versioned contracts. Contract status is `approved`; implementation
remains `nonconformant`, validation `in_progress`, production
`existing_use_only`, verdict `amend`, and re-audit `required`. F009 and F010
remain `addressed_pending_reaudit` until a fresh independent disposition.

The human evidence owner executed all seven exact-SHA
`SCI-MAP-001-UNITY-001` cases on 2026-08-03 and returned the corpus locally.
The 2026-08-05 read-only reconciliation binds every captured executable and
reduction index to candidate `ed28dafb37f9113c0d3c95297148157129a90886` and
records the exact product inventory, evidence limitations, missing `S-X-SEQ`
observation-level realization serialization, and typed-WCS/Stokes discrepancy
in the campaign closeout note. Do not repeat the campaign. The later owner
amendment accepts F012 only for bounded external product/execution/SEQ-OMP
claims and retains every missing lane as a limitation; this reconciliation
does not close findings. F013 continues to condition calibration/unit/response,
projection/WCS, coefficient/covariance, and upstream-eligibility conclusions
on `SCI-ALIGN-001`, `SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and
`SCI-VAL-001`. A fresh
`codex/reaudit-sci-map-001` worktree must assess the committed repair and the
returned external corpus before findings or production disposition can change.

The independent re-audit at
`851035e67f63bdb2bacc122b17566877a9e6db97` remains intact historical evidence.
The project-owner amendment at
`6409a36d324072c9b29145c620d01a0686275870`, reproduced byte-for-byte as
`handoff/SCI-MAP-001_OWNER_SCOPE_EVIDENCE_AMENDMENT_2026-08-05.md` with
SHA-256 `52be19700b73659ba1847012d4cb0766407399cda5899570acb79bf5b45221f3`,
authorizes a second bounded repair only for F005 aggregate/index fail-closed
safety and coadd-enabled observation-realization persistence. It also defines
the production WCS/card tests and accepts F012 only for the named external
product/execution/SEQ-OMP claims, with every missing lane retained as a
limitation and no Unity rerun required solely for those absences.

The second-cycle candidate rejects floating and signed-count aggregate
overflow and finite projected coordinates outside the representable index
domain before live bundle mutation. It persists required observation
realizations alongside coadd realizations, preserves observation/coadd
ownership and realized cardinality, and propagates missing required writer
slots before the first HDU. Production-path fixtures enforce typed/sidecar to
physical-FITS WCS separation `<= 0.1 arcsec`, exact orientation and centered
integer placement, finite/unit-bearing threshold-card identity and aliases,
sidecar agreement at `rtol=1e-12`, complete realization identity, and
unchanged-WCS atomicity. Normal finite-domain mapmaking, coadd, threshold, and
WCS policies are unchanged.

The complete local second-cycle gate set passes: `citlali_cli`, the monolithic
test executable, the safety executable, and the isolated production-FITS
executable build; 592/592 enabled CTests pass with the one pre-existing
disabled test unchanged; the focused contract/provenance/truth executable
passes 31/31; its ThreadSanitizer build passes 9/9 without a race report; the
production FITS suite passes 22/22; all 147 baseline-tool tests pass; and the
127-test config preflight passes all four mode kits, eight compact
compatibility cases, 100% compact-surface coverage, and every typed-boundary
audit. These are repair-candidate results for fresh independent review, not a
finding or conformance disposition.

The second-cycle independent re-audit at
`fc26e24e6543d1102f9fcc9bf4e849369b39dd04` proposed F005, F007, and F010
for closure, but found one remaining F004/F011 bookkeeping defect: completion
provenance multiplied both observation and coadd products by the global
filtered-stage count even though coadd-enabled filtering writes only the raw
observation stage and writes both raw and filtered coadd stages. Those proposed
closures are re-audit findings, not coordinator-integrated canonical closure.

The final bounded bookkeeping candidate now applies the already-established
observation and coadd output-stage counts separately. In the audited one
observation, one-coadd, three-map, two-realization filtered case it records the
exact 18 realization writes and 9 empirical product maps; the existing
non-coadd filtered and coadd unfiltered states remain unchanged. This alters
only realized provenance cardinality, outside the numerical and output-routing
paths.

The final-bookkeeping local gates pass: all six required build targets
(`citlali_cli`, the monolithic test executable, the safety executable, the
focused truth executable, its ThreadSanitizer build, and the isolated
production-FITS executable) complete; the exact three-state cardinality test
selection passes 3/3; 593/593 enabled CTests pass with the one pre-existing
disabled test unchanged; the focused truth, ThreadSanitizer, and production
FITS suites pass 31/31, 9/9 without a race report, and 22/22; all 147
baseline-tool tests pass; and the 127-test config preflight passes all four
mode kits, eight compact compatibility cases, 100% compact-surface coverage,
and every typed-boundary audit. These remain repair-candidate results for a
fresh independent exact-SHA re-audit, not a finding or conformance disposition.

F004 and F011 remain pending the final exact-repair-SHA re-audit. F005, F007,
and F010 retain only the second-cycle re-audit's proposed closure pending
canonical disposition. F009 and F010 remain `addressed_pending_reaudit`;
production remains `existing_use_only`. F012 is owner-accepted only in the
amendment's bounded terms. F013 remains conditioned on `SCI-ALIGN-001`,
`SCI-CAL-001`, `SCI-AST-001`, `SCI-PTC-001`, and `SCI-VAL-001`; this repair
closes none of them.

On 2026-08-01 the versioned human-run campaign package for exact candidate
`ed28dafb37f9113c0d3c95297148157129a90886` was prepared under
`validation/campaigns/SCI-MAP-001-UNITY-001/repair-ed28dafb/`. It pins all
seven repaired-success cases, successor product contracts, explicit owner
deployment values, native TolProj/TolTECA source ordering, independent F010
reconstruction inputs, collection manifests, and frozen analysis. Package
preparation did not access Unity. The owner subsequently executed a bounded
minimal transfer of the seven cases; the external products remain in the
owner-supplied local corpus rather than this repository. The durable
`SCI-MAP-001_EXISTING_CORPUS_CLOSEOUT_2026-08-05.md` records what is present,
what is unavailable, and the exact re-audit route. The package also records
ALIGN-OD1 through ALIGN-OD8 and
ALIGN-C001 as owner-approved at record commit
`4f905f4f353e91847a303f4f3959654f3f03c302`, with canonical identity correction
at `35cc8ce246e8e70c569e650be6c1eae2c91b80ef`, and the bounded repair/re-audit
handoff at coordination commit
`0309fd48a973a6e7e136224906ac49c02f0171be`, and clean coordination-ledger HEAD
`846128c8ee6dc27851bd6c71aeecbe4739e1d24a`. The dedicated ALIGN phase-0 repair
is active from base `9aae0e669384c5c0c0dda93debc194d6b8dac787`, but no ALIGN
application-repair commit or re-audit exists. ALIGN implementation therefore
remains nonconformant, validation is in progress, and production remains
`existing_use_only`. A MAP campaign result cannot close ALIGN, CAL, AST, PTC,
or VAL; F013 remains conditioned until the ALIGN repair, exact-repair-SHA
evidence, and fresh re-audit succeed.

## 2026-07-26 Conan 2 Build Review

The previously deferred TolTECA build implementation is now available and has
received an initial architecture review. Exact evidence, requirement
dispositions, compatibility gaps, and the bounded integration sequence are
recorded in
`doc/TOLTECA_BUILD_INTEGRATION_REVIEW_2026-07-26.md`.

The project selected the **Adapt** path. Tula CMake's typed Conan 2 feature
registry, generated-preset workflow, explicit first-party package graph, and
compiler matrix are accepted as the foundation for the successor build. The
reviewed `citlali/v4.x_conan2` target is not a drop-in application build: it
intentionally contains only a five-source static-library slice, 41 headers,
and the Gaussian-model test, with no production CLI or generated source
identity.

The full refactored application still requires 709 Citlali headers, eight
active compiled library sources, the CLI, more than 500 focused CTests,
embedded default configuration, and source/dependency provenance. Kidscpp v3
also omits the active TolTEC raw-data adapter and the presently constructed
but apparently unused sweep fitter. Direct HDF5 and Zlib ownership must be
made explicit.

Phase 5 build integration is therefore unblocked but not complete. The next
work is a bounded compatibility and target adaptation, followed by the full
local gate, a Unity point smoke run, and the frozen same-SHA four-mode matrix.
The existing build remains available until the new path proves all of those
gates. No numerical algorithm changes are part of this integration.

A 2026-07-31 isolated retest of the latest upstream revisions materially
improved this disposition: Tula, Kidscpp, and the Citlali CLI build under exact
Homebrew LLVM 20 and C++23, and their available in-tree tests pass. The
installed Citlali package-consumer test still fails because CPM-provided
NetCDF C++ headers and library metadata do not propagate through the exported
Tula package. The bundled macOS profile also resolves unversioned Homebrew
`llvm` rather than enforcing LLVM 20, one Tula CMake Python test assumes a
specific Conan launcher form, and the real TolTEC reader tests remain skipped
without fixtures. These are adaptation entry gates, not reasons to replace or
freeze the application mainline.

## Current Snapshot

- A 2026-07-30 coherent raw-I/Q event investigation has produced the first
  mode-aware observe-only production slice. The current RTC/PTC learning
  path records accepted intervals per detector UID (a run-scoped row key, not
  persistent identity) and compacts only within that UID, so a physical
  network event loses its tone-vector identity and
  fans out into many records. A versioned, fail-closed template schema,
  non-mutating classifier, alternating-half evaluation, typed configuration,
  strict template loader, all-network observation sidecar, and focused C++/
  Python tests are now in place. The extended evaluation scores all 11
  networks present in the corpus: 572
  event/network vectors and 1,210 quiet epochs. The same descriptive point
  selects 167/216 independently participating responses and 0/1,210 quiet
  epochs and surfaces 52/356 responses at shared epochs that did not
  independently trigger. Stable high-cosine but low-amplitude control modes
  show that cosine alone is not a pathology trigger. nw8 remains the positive
  benchmark and nw9 explicitly fails a single-mode stability gate, but
  neither result forms a runtime network allow-list. A catalog-time-blind
  three-state HMM trained on the first half of science observation 152431 now
  independently recovers 96.2% of catalog events in the held-out half and
  transfers with 78.3% and 81.1% recall to observations 152419 and 152433
  after unlabeled target-intrinsic shape normalization. All matched
  transitions have the expected direction and exceed 200 circular-shift null
  trials, while two quiet controls have zero catalog matches. Frozen-scale
  decoding exposes strong nonstationarity, including a 10.9-fold nw3 scale
  increase within 152431; shape evidence and absolute severity must therefore
  remain distinct. This is forensic validation, not an automatic flagger.
  See
  `handoff/SCIENCE_IQ_HELD_OUT_MODE_DETECTION_2026-07-30.md`. The opt-in
  production sidecar clusters RTC-seeded shared epochs and attempts a bounded
  raw-I/Q score for every raw network present, including networks that did not
  seed the event. It writes explicit template and compatibility status and
  changes no samples, flags, weights, learning state, or maps. The first
  bounded Unity
  smoke at `91f99bde` loaded all 11 templates and wrote a schema-valid
  sidecar, but exposed a lifecycle defect: standard RTC diagnostic output
  cleared detailed scan summaries before the observation-level sidecar read
  them, producing zero candidates. The corrected path now copies only
  threshold-passing seeds into a compact scan-keyed cache before detailed QA
  cleanup and clears that cache after sidecar publication. Its CLI build,
  14 focused tests, all 532 enabled CTests, and the full 123-test config
  preflight pass. A corrected observation-152433 Unity smoke, broader corpus
  validation, and same-input enabled/disabled output identity remain the next
  gates;
  coherent masking and subtraction remain disabled. See
  `handoff/COHERENT_RAW_IQ_MODE_OBSERVE_ONLY_ARCHITECTURE_2026-07-30.md`.
- A 2026-07-24 pointing fruit-loop investigation is active. Five controlled
  observations have exact no-feedback seeds but monotonically brighter and
  broader fitted sources through four feedback passes. Production
  subtract/add-back now has opt-in per-scan/array diagnostics; direct
  signal/kernel round-trip and controlled injected-Gaussian recurrence tests
  pass, rejecting a basic sign or unconditional double-add error. The frozen
  obsnum 133410 maps reveal that the low absolute flux cuts select a broad,
  one-sided positive model: 95--98% of active selected pixels and 57--79% of
  tapered positive model sum lie beyond 40 arcsec at the seed. Controlled
  learning, template-taper, and detector-weight Unity ablations are complete:
  learning and tapering are not material causes, and recomputing post-addback
  weights is image-array identical to the control. Fruit loops recover source
  width toward the propagated kernel width. The 13-variant follow-up matrix
  shows stable ten-iteration convergence and a strong PCA-depth response:
  cleaner strength, not broad model support or projection choice, controls the
  correction size. Cleaner-free real-source fits are farther from the
  matched-APT reference, so the reference mismatch does not establish a
  fruit-loop fault. A diagnostic-only, fail-closed full-PTC injected-source
  pair is now implemented locally: restart-matched control and injected
  branches differ only by adding a declared source through the pristine unit
  kernel before model subtraction, and the comparator fits their difference
  through every saved iteration. The first paired Unity run exposed an exact-
  restart defect: checkpoint v1 omitted retained PTC weight-validation state,
  so its control continuation diverged from the uninterrupted trajectory.
  Checkpoint v2 now stores and restores that state, rejects v1 checkpoints,
  and requires an exact uninterrupted-control gate before transfer metrics
  are interpreted. Its local CLI/test builds, all 514 enabled CTests,
  synthetic analysis-tool recovery, and complete config preflight pass. The
  corrected v2 Unity pair now passes exact continuation. The injected source
  recovers monotonically through iterations 9--13, its PSF converges to the
  realized kernel, centroids remain stable, and successive map changes shrink.
  The extended pair through iteration 18 again passes exact restart and
  converges to kernel-normalized recovery of 95.8%, 94.9%, and 98.3% for
  a1100, a1400, and a2000, with stable centroids, kernel-matched widths, and
  1.0--1.7% final map changes. The original monotonic growth is therefore
  resolved as stable recovery of cleaner-suppressed signal, not runaway
  feedback. The remaining 1.7--5.1% attenuation is a measured scientific
  limitation and not an automatic correctness fix. Production defaults
  remain unchanged. A 2026-07-26 calibration-reference assessment now
  separates the use cases: the existing products support qualified relative
  astrometry and effective processed-PSF use, do not support absolute
  photometric/transfer calibration, and cannot yet predict associated science
  response because no local pointing/science association or science-mode
  injection exists. The iteration-18 amplitude/shape plateau does not pass the
  all-array 1%, 2%, or 5% two-transition whole-map criterion; no production
  stopping policy was adopted. A minimum checkpoint-v2 Unity matrix and a
  bounded science-injection design await owner selection/approval; no new
  reductions have been requested. The follow-on 108-observation extension now
  has an independent, frozen quality baseline from all 324 RC1 array maps and
  processed kernels: 54 observations are labeled normal, 38 marginal, and 16
  stress for experiment design. The original five contain four normal, one
  marginal, and no stress observations. The 16 common-binary ten-iteration
  Stage A sentinels are now complete and downloaded: all 480 iteration metrics
  are finite, all 432 transitions are measurable or explicitly classified,
  and the predeclared Stage B gate passes with 8 normal, 5 marginal, and 2
  stress observations retaining all three source associations. At the strict
  combined endpoint gate, 7/48 array trajectories pass at 1%, 21/48 at 2%,
  36/48 at 5%, and 40/48 at 10%. One stress a2000 trajectory follows a
  cross-array-inconsistent source and eight trajectories have FWHM fits
  censored at the pointing fitter's upper bound; neither is counted as
  convergence. Astrometry is therefore qualified per source-associated stable
  trajectory, effective-PSF use is qualified only for uncensored fits,
  photometric calibration remains unsupported, and science response remains
  unmeasured. No stopping policy is adopted. The 92-observation Stage B array
  has now completed under the exact Stage A executable and unchanged policy.
  Its originally failed task 81 was rerun alone; all 16 Stage A and 92 Stage B
  jobs now pass product, log, config-checksum, and provenance audits. The
  complete analysis covers 108 observations, 324 array trajectories, and
  3,240 maps through iteration 9. It exposed a product-semantic defect in the
  historical pointing-table
  `sig2noise`: it is fitted amplitude divided by full-map RMS, so recovered
  source structure makes it a dynamic-range diagnostic rather than
  statistical significance. The population analyzer now excludes that legacy
  quantity from convergence, reports formal `amp / amp_err`, source-free
  background and roughness, and a versioned blank-sky empirical PSF S/N
  separately. A backward-compatible pointing-table v2 appends truthful
  `peak_over_full_map_rms` and `fit_sig2noise` columns while retaining the
  legacy column. The complete morphology-aware population supports a
  discussion candidate of 3% amplitude change, no evaluation before iteration
  6, and two consecutive all-array passes. Every one of 225 unresolved-source
  array trajectories resolves at 3%, with a 1.87% P90 and 3.57% maximum
  stopped-to-iteration-9 residual. Planetary disks use observation-epoch JPL
  Horizons diameters convolved with each realized kernel; only 77/99 planet
  trajectories resolve at 3%. The complete V0 multi-metric rule resolves
  57/108 observations. Of the 51 others, 23 are measurement-limited and 28
  retain measurable but unresolved trajectories appropriate for short
  checkpoint-v2 continuation. Formal and empirical point-source S/N rise
  while source-free background does not increase monotonically, confirming
  that the historical full-map dynamic-range decline is not scientific S/N
  loss. Separate PSF, centroid, map, support, learning, and noise criteria
  remain unapproved. See the
  [convergence-criteria discussion](FRUIT_LOOP_CONVERGENCE_CRITERIA_DISCUSSION_2026-07-27.md).
  The local implementation snapshot passes the `citlali_cli` build, all 517
  enabled CTests, all 135 baseline-tool tests, the complete 123-test config
  preflight and strict audits, and all 33 fruit-loop tool tests.
  Exact injected-source pairs remain reserved for one representative of each
  quality stratum. These are descriptive strata, not data rejection or
  production policy. See the
  [feedback investigation](FRUIT_LOOP_FEEDBACK_INVESTIGATION_2026-07-24.md)
  and
  [calibration-reference assessment](FRUIT_LOOP_CALIBRATION_REFERENCE_INVESTIGATION_2026-07-26.md),
  plus the
  [population extension plan](FRUIT_LOOP_POPULATION_EXTENSION_PLAN_2026-07-26.md).
- A 2026-07-24 reliability investigation is active for two long
  108-observation pointing jobs that received `SIGBUS` on the same a1400 Ceres
  solve after 45 completed observations. The fitter code is unchanged across
  the failed and current commits; 512 repeated synthetic fits and an exact
  138-fit replay from the downloaded scientific maps both pass, including
  ASan/UBSan instrumentation of the fitter translation unit. Corrected
  observation-boundary RSS is approximately 0.8--1.1 GiB rather than the
  step-wide 26/41 GiB Slurm peaks. Current work adds PID-level resource,
  executable-mapping, and robust signal diagnostics plus a low-level
  config-slicing/native-run harness. No algorithm or fit policy has changed,
  and a native Unity failure frame is still required before choosing a fix.
  TolPROJ now has a tested submission-time executable snapshot and
  checksum-verified node-local launcher on its development branch. This
  prevents future queued/running reductions from depending on a mutable
  `build/bin/citlali`, but is recorded as an operational safeguard rather than
  proof of the historical `SIGBUS` root cause.
  See the
  [investigation handoff](../handoff/CITLALI_MULTI_OBSERVATION_SIGBUS_INVESTIGATION_2026-07-24.md).
- Phase 5 validation-epoch preparation is complete as of 2026-07-24. The
  historical four-profile Phase 4 epoch remains active and immutable; a
  separate four-profile `phase5-v2.1-candidate-2026-07-24` epoch is registered
  as preparing. Its config comparison is exact except for the versioned
  `tolteca-native-project-bindings-v1` policy, which permits host/project path
  prefixes to move while preserving bound file and directory identities.
  Preparing profiles have no accepted baseline records, require an explicit
  comparator, and cannot report an accepted verdict. All four available V2.1
  suite fixtures pass config, product-contract, and product-comparison smoke
  gates; promotion remains blocked by runtime-provenance V1 in every fixture,
  missing science pointing provenance, the deferred build review, and the
  absence of a same-SHA four-mode candidate matrix. The one-command fixture
  verification matches all recorded outcomes. All 134 baseline-tool tests and
  the 123-test full config preflight pass. See the
  [successor-epoch preparation record](PHASE5_VALIDATION_EPOCH_PREPARATION_2026-07-24.md).
- Runtime resource debt D16 is closed as of 2026-07-24. TolPROJ keeps refactor
  runtime threads and generated Slurm CPUs coherent, rejects oversubscription
  before its recommended submission path, and preserves legacy defaults.
  Citlali uses an independent runtime safety net that resolves Slurm, affinity,
  and hardware
  availability, caps rather than aborts an allocated job, emits one warning,
  and writes `citlali-runtime-provenance-v2`. Local build, focused tests, full
  gates (500 CTests, 119 baseline tests, and 118 config tests), and the 147-test
  TolPROJ suite pass. The matching Unity case passed at `d339053cc` in
  `pointings_v22/redu00`: six requested threads matched six affinity-available
  and six effective OpenMP threads without adjustment; runtime provenance V2
  is valid, the run completed all 12 PTC chunks with no logged issues, and all
  non-profile scientific products are exact against `pointings_v21/redu00`.
  The intentionally mismatched direct-submission case then requested 12
  threads inside a six-CPU affinity allocation. It emitted exactly one
  resource-cap warning, continued with six effective and realized OpenMP
  threads, recorded the adjustment in valid V2 provenance, completed all 12
  PTC chunks, and again produced exact non-profile products. [The runtime
  resource contract](RUNTIME_RESOURCE_CONTRACT_2026-07-23.md) records the
  evidence.
- Refactor baseline: `376e0022`.
- Production code inspected by the external review: `84670829`.
- Latest accepted point reduction: Phase 3 exit checkpoint `redu66`, produced
  by `2a974e0dd`, is exact against full-Wiener checkpoint `redu65` across all 19
  non-profile scientific products, including complete RTC/PTC timestreams,
  with zero changed or skipped records. Their 490-leaf configs are exact. Both
  profiles contain the same multiset of 78 stage/context records; only elapsed
  values and concurrent completion order differ. The run has 12 complete PTC
  chunks, zero logged issues, all required provenance valid, and a successful
  VAST-backed exclusive output-root acquisition. `redu64` remains the accepted
  mature library-exit checkpoint and `redu63` the first compiled boundary.
  Observation-resolved astrometry
  provenance `redu61`, disabled polarimetry capability provenance
  `redu60`, external KIDs/config-source provenance
  `redu59`, post-processing authority cleanup `redu58`,
  realized provenance `redu57`, typed source-fitting `redu56`, source-finding `redu55`, map-filter `redu54`,
  enabled-filtering `redu53`,
  unfiltered `redu51`, and bounded full-noise-output `redu49` remain the
  immediate post-processing, pointing, and noise-products control fixtures.
- The Phase 4.1 self-contained point smoke `pointings_v21/redu00`, produced by
  `cfae989ce`, accepts the raw-only `phase4.1-v2.1` pointing policy. It
  completed all 12 PTC chunks in 135 log seconds with no error-level records,
  and every required provenance sidecar is valid. Requested and effective map
  filtering are disabled, realized filter-context and filtered-map counts are
  zero, and no filtered product directory exists. The raw pointing table
  contains three valid fits from three attempts, one for each array.
- Phase 3 full-Wiener point `redu65`, produced by `6dd0057f8`, is accepted
  against matched OG `redu10` at `ffc6b907`. Both use five noise realizations,
  a Gaussian template, and `lowpass_only: false`, and both execute all six
  expected Wiener core calls. The seven filtered products pass the strict
  scientific-tolerance gate with 148 compared records and no skips; the
  three-array pointing-fit table is exact. The refactor run has zero issues.
- Latest accepted OOF reduction: refactor `redu02` for observations
  152385-152387, produced by `9ea6d7f01`, is exact against accepted refactor
  `redu01`. The established OG `redu00` versus refactor relationship is
  unchanged. All 30 comparable products are
  present with no skipped records; pointing-table data and all per-observation
  ECSV/FITS dates are exact, and all scientific numeric differences pass the
  standard `2e-8 + 1e-10 * abs(reference)` tolerance. The only accepted
  differences are inactive RTC-despike config metadata recorded differently
  by the legacy and typed paths.
- Phase 4.1 OOF smoke `redu00` in the self-contained validation suite, produced
  by `e97de3fd`, intentionally enables diagnostic Gaussian fitting while
  retaining `psf_preserve` and `map_center`. All three observations completed
  in 59.4 seconds with zero logged issues, all required provenance valid, and
  nine valid array fits from nine attempts. Its matched APT differs materially
  from the older accepted OOF fixture, so this is mode-kit execution evidence,
  not a numerical replacement for accepted `redu02`.
- Latest accepted science reduction: clean single-job four-iteration sequence
  `redu28` through final `redu31`, produced by `a7a35a00`. Its 502-leaf config
  is exact against accepted `redu23`; all 12 FITS and 15 NetCDF product sets are
  complete. All 84 map layers pass the science-equivalence gate with maximum
  relative RMS `7.09e-14`, all integer diagnostics are exact, and all 1,394
  NetCDF variables pass. The final run has zero logged issues and every required
  provenance record is valid. This supersedes `redu23` as the science fixture.
- Latest accepted Beammap reduction: Phase 3 checkpoint `redu06`, produced by
  `6dd0057f8`, is exact against accepted `redu05` across all 12 comparable
  products and 16,453 comparison records, including complete detector TOD,
  diagnostic NetCDF, detector-fit tables, and six split-map FITS products. Its
  529-leaf config is exact, all required provenance is valid, and the log has
  zero issues. It accepts the fruit-loop input/feedback and mature Wiener
  failure-contract tranches for Beammap.
- The Phase 4.1 self-contained Beammap `redu00` produced by `cfae989ce`
  accepts the split-output correction. It completes 198 PTC chunks and three
  internal Beammap iterations in 3,779 log seconds with no error-level
  records, and all required provenance is valid. Its final APT marks 196, 27,
  and 38 bad detectors in a1100, a1400, and a2000; the split bad-detector FITS
  files now contain exactly those counts, while the good files contain 2,901,
  1,186, and 886 detectors. The three iteration fit counts, final
  network-position flags, and complete kernel-map diagnostic summary are exact
  against rejected predecessor `189bbf85d`, establishing that the fix changes
  output completion rather than Beammap science. Commit `e496dcb6e` catches
  the CCfits base exception at required map/header boundaries, and
  `a68bf1737` omits unavailable per-detector FITS keywords while retaining
  `NaN` in the authoritative APT table for compatibility with Unity's older
  CCfits. All 13 established Beammap product families pass. The immutable
  version-one contract continues to describe the historical 13-product
  snapshot. Successor contract `phase4.1-beammap-products-v2` classifies and
  schema-checks the required `citlali_restart_checkpoint.nc` and accepts all
  14/14 current products without changing the active V1 profile.
- `redu23` and `redu24` completed all 12 PTC chunks with zero error-level log
  records and complete TOD/diagnostic products. Their common numeric products,
  FITS maps, and pointing tables are exact; only profiling timing differs.
- `redu21` and `redu22` had exact common numeric products with complete TOD
  comparison, but both contained 12 logged NetCDF errors.
- The same YAML exposed two provenance defects in `redu22`: an effective IIR
  default appeared for a disabled filter and an extinction sentinel changed.
  `redu25` validates the intended disabled-state provenance correction with
  exact scientific products.
- Local `citlali_cli`/test builds and full config preflight pass.
- CTest discovers and passes all 490 tests. The Phase 4.1 config preflight now
  passes 117 focused tests; the checked leaf contract covers 578 leaves and the generated
  startup schema covers 728 normalized YAML nodes. The six new direct tests
  cover the production rejection of experimental maximum-likelihood
  mapmaking, analytic flux conversion, detector-specific calibration,
  detector pointing, and two source-finder safety boundaries. Eleven additional
  NGC4449 candidate tests cover cap-independent effective learning state,
  interval/penalty compaction, diagnostic decoupling, fruit-loop activation and
  realized feedback, Beammap policy isolation, and truthful standardized-map
  naming. Three additional tests cover the diagnostic-only learning/HK match
  contract and required sidecar output. Two additional Wiener tests cover
  fail-closed compensated-template normalization and unchanged
  well-conditioned convolution.

These facts are characterization evidence, not a production-equivalence claim.

### NGC4449 Full-Science Candidate Investigation (2026-07-21)

The first five-observation, ten-iteration NGC4449 run exposed four blocking
Citlali contracts: statically empty fruit-loop feedback, formal-weight
standardized maps mislabeled as S/N, a diagnostic learning cap that truncated
operational state in input order, and warning amplification that obscured QA.
The bounded investigation and candidate corrections are recorded in
`doc/NGC4449_CITLALI_INVESTIGATION_2026-07-21.md`.

Local builds, all 481 CTests, the 116-test config preflight, and 106 baseline-
tool tests pass. The changes are not an accepted scientific snapshot:
applying the full effective learned state intentionally changes the
previously cap-truncated flags and therefore requires a successor Unity science
profile and an intended-science-change ledger entry before acceptance. The
immutable phase-4 v1 product checks remain available for historical artifacts;
candidate v2 basic-map checks reserve S/N names for empirically calibrated
products. A new learning-housekeeping QA sidecar correlates deduplicated
busy-network pathologies with selected TolTEC thermometry and dilution-fridge
samples while remaining strictly outside the flagging and learning policy.

The first one-observation spatial-feedback control then exposed a separate
configuration-authority defect: science YAML requested
`pointing.source_strategy.fruitloops_center_mode: map_center`, but the science
execution path never loads the pointing plan and therefore continued with
automatic off-center peaks. A successor typed control,
`timestream.fruit_loops.source_center_mode`, is now owned by the processed-
timestream request, serialized in its snapshot and NetCDF config record, and
adapted directly to the fruit-loop processor. The additive `auto` default
preserves prior behavior. The NGC4449 successor requests `map_center`; a Unity
run must confirm the realized log before this becomes accepted science
evidence.

The project owner then approved state-complete cross-job fruit-loop
continuation so NGC4449 iterations can be extended without discarding learned
state. ADR 0006 defines the new required atomic
`citlali_restart_checkpoint.nc` artifact and explicit
`timestream.fruit_loops.restart_path`. The checkpoint stores compacted
operational masks and detector penalties, absolute iteration identity, ordered
observations, map type, creator version, and the complete learning-policy
snapshot; bounded diagnostic event history is intentionally excluded. Loading
is fail-closed for the stored compatibility contract, `path` and
`restart_path` are mutually exclusive, and `max_iters` is the absolute
exclusive stop. Local split-run tests show five completed synthetic learning
iterations plus a two-iteration restart exactly match seven uninterrupted
iterations. Local CLI/test builds, 488 CTests, the 116-test complete config
preflight, and all 108 baseline-tool tests (including 60 reduction-audit
tests) pass. A matched Unity split versus
uninterrupted science run is still required; the already-running older binary
cannot create this new checkpoint.

The first real exact-restart control, performed for the full-PTC
injected-source experiment on 2026-07-25, invalidated checkpoint schema v1.
Restarted absolute iteration 9 differed from uninterrupted iteration 9 by
4.6--26.6% relative RMS in signal maps and 8.3--49.7% in weight maps. The
missing state was the validated PTC weighting accumulator and finalized
detector-factor vectors retained in `PTCProc` across in-process iterations.
Schema v2 now stores those vectors and their phase, records a canonical
processed-timestream policy snapshot, and rejects v1 checkpoints. Focused
restart tests, the 500-test `citlali_test` binary, the local CLI build, and the
complete 123-test config preflight pass. A new uninterrupted v2 trajectory
plus exact restarted control is required before the injected-source transfer
experiment can be interpreted.

The completed older-binary continuation then exposed an invalid-success
boundary in lowpass-only kernel filtering. The latest radial `a1400` template
has `abs(sum)/sum(abs)` approximately `0.004`; unit-sum convolution therefore
amplified a compensated transfer kernel by a cancellation condition number of
about 251 and produced filtered signal/kernel/weight products with no valid
flux interpretation. A shared serial/OpenMP runtime contract now rejects a
non-finite, zero-L1, or cancellation-conditioned unit-sum template below a
`0.05` DC fraction, limiting normalization L1 gain to 20 and reporting all
conditioning values plus the full-Wiener corrective route. Well-conditioned
convolution behavior is unchanged. Local serial and OpenMP CLI/test builds,
all 490 CTests, the 117-test complete config preflight, and all 108 baseline-
tool tests pass. An NGC4449 full-Wiener Unity successor remains required
before acceptance; all filtered `a1400` products from the lowpass-only
NGC4449 series remain quarantined.

The follow-up raw-readout investigation on 2026-07-29 established the
producer-confirmed schema of `Header.Toltec.AdcSnapData`: shape `[2,4096]`,
with index `0` representing the beginning of the raw data file, index `1` the
end, and signed 12-bit ADC counts in `[-2048,2047]` stored in a NetCDF
`short`. The current input reader now names that boundary ordering and count
domain without changing numerical behavior. In the eight downloaded NGC4449
pointings, nw9 reaches both rails at both file boundaries in every
observation; nw3/nw4 have low headroom with sparse rail contact; and nw1,
nw2, and nw8 exhibit late map pathology without ADC saturation. ADC
utilization does not acquire the broader pathology's 152420 onset, so clipping
is a real nw9 validity problem but not its common cause. Retained debt D17 now
requires a cold-boundary saturation validator and explicit persisted
severity, while leaving warning, network-exclusion, and reduction-failure
thresholds unapproved pending representative validation.

## Active Phase

**Phase 4.1 - TolTECA operator config structure** is complete as of 2026-07-23,
and **Phase 4.2 - technique and performance review** is complete as of
2026-07-17. The project owner added both stages between the adopted Phase 4
evidence package and final Phase 5 integration. Phase 3 library/session work
is complete: local gates pass and Unity point `redu66` accepts the output-root
ownership repair and exact scientific behavior at the first compiled
boundary. Phase 2 config authority and provenance remains complete at Unity
point `redu62`. Formal Phase 5 integration remains blocked on the deferred
TolTECA build review; compilation-independent validation-contract and
integration-packet preparation may continue.

The TolTECA build owner is preparing a Citlali v4.x build approach intended to
apply here but has not yet provided an implementation for review. The
[`build integration requirements`](TOLTECA_BUILD_INTEGRATION_REQUIREMENTS_2026-07-23.md)
define the outcome and evidence expected from that work without prescribing
its tools or creating a competing build-system rewrite. This tree retains its
current working CMake path until the implementation can be evaluated and
adopted, boundedly adapted, or explicitly deferred.

Compilation-independent Phase 5 readiness is current as of 2026-07-24. The
existing local path builds `citlali_cli`; its previously established CTest
  gate remains valid, and the current 123-test config preflight and 134
baseline-tool tests pass. The 60-record validation ledger, three-entry
intended-science-change ledger, eight-profile/two-epoch registry, and
session-exit audit are valid, with zero supported library exits or growth. The
[`integration record template`](PHASE5_INTEGRATION_RECORD_TEMPLATE.md) now
captures the frozen-SHA build decision, local gates, four-mode Unity matrix,
science/capability disposition, and integration authorization without
pre-filling unavailable evidence. This readiness does not freeze a candidate
or close the deferred build criteria.

TolPROJ commit `e0754af` supersedes the custom suite installer and staging
layers introduced by commits `39f724d` and `8310c24`. The canonical
`suite.yaml` now contains only path-free observation selections and validation
metadata. `tolproj validation-suite init` queries TolPROJ's metadata database
and uses the existing native project builders to create ordinary `point`,
`oof`, `beammaps`, and `science` projects. The science project contains its own
complete pointing support, while the selected-observation Beammap builder
discovers source-matched pointing support from metadata. Raw discovery and
copying, tune reduction, cohort construction, APT seed selection and matching,
Beammap flux estimation, and `--refactor` reduction setup all remain owned by
the established TolPROJ commands. Native `project.yaml` is deliberately live
workflow state rather than a hashed immutable artifact. Verification guards
the portable selection and required native structure without objecting to
normal status, cohort, or APT updates. The workflow does not provision an OG
tree, submit jobs, or run Citlali. All 104 TolPROJ tests pass. Freshly created
Unity projects subsequently completed point, OOF, Beammap, and science smoke
reductions with their requested products and no unexpected errors, closing
the Phase 4.1 execution gate.

TolPROJ follow-up commit `d2c90f3` applies the first Unity setup corrections.
The native skeleton is mode-minimal: point and OOF omit an unused nested
`pointings/` directory; Beammap omits both unused `pointings/` and `apts/`;
science retains its pointing-support structure. All generated Citlali
`02_redu.sh` scripts now request the configured partition (`toltec-cpu` on
Unity). TolPROJ file logging is action-specific (`copyraw.log`,
`reducetunes.log`, `matchapts.log`, `pointings.log`, `science.log`, and
`flxscale.log`) instead of funneling independent steps into `tolproj.log`;
SLURM stdout remains separately named `<jobname>-%j.out`. All 105 TolPROJ tests,
full Ruff, and byte-compilation pass.

TolPROJ commit `9fb4c80` closes a science setup ordering hole found during the
first suite attempt. Science setup now verifies that every configured
`cal_objs` pointing-product directory exists before writing reduction configs,
and accepts `--pointing-reduction reduNN` so setup uses the same accepted run as
pointing flux calibration. This converts a late TolTECA `invalid calobj path`
failure into an actionable TolPROJ preflight error. All 106 TolPROJ tests pass.

The populated Unity tree then showed that existence alone is insufficient:
TolTECA recursively requires exactly one `ppt_*.ecsv` under each `cal_objs`
path, while an observation root can contain both raw and filtered tables.
TolPROJ commit `704b486` makes the established raw pointing product explicit,
validates exactly one table under `<obsnum>/raw`, and emits that directory for
science and Beammap pointing references. All 106 TolPROJ tests, full Ruff, and
byte-compilation pass.

Citlali commits `95b7b7f57` and `f59c663f8` establish config kit
`phase4.1-v2.1`. The self-contained OOF smoke showed that the prior default
produced vacuous all-zero pointing-fit tables, so OOF now enables diagnostic
Gaussian fitting while preserving PSF-preserving mapmaking and map-centered
fruit-loop support. Routine standalone, science-support, and Beammap-support
pointings now default to raw products; Wiener settings remain visible for an
explicit validation overlay but filtering is disabled. TolPROJ commit
`16ebe69` vendors the complete 29-file canonical kit byte-for-byte, retains
`phase4.1-v2` for historical reproducibility, and selects V2.1 only for fresh
`--refactor` setups. All 139 TolPROJ tests and Ruff pass. No Citlali
compilation is required for these numbered-YAML integration changes.

The four native-project smoke gates are complete. Point
`pointings_v21/redu00` verifies the V2.1 raw-only policy with three valid array
fits and no filtered outputs. OOF `redu00` verifies diagnostic Gaussian fits
under PSF-preserving mapmaking. Science `redu03` verifies the unchanged science
V2/V2.1 policy across two observations, four fruit-loop iterations, raw and
filtered coadds, noise products, and learning outputs. Beammap `redu00`
verifies all 198 PTC chunks, three internal iterations, APT good/bad
classification, and complete good/bad split FITS products after the required
split-output correction. All four runs completed without unexpected
error-level records. This closes Phase 4.1 and retained-debt item D09.

Compilation-side Phase 4 work is explicitly deferred as of 2026-07-16 pending
review of the TolTECA developer's revised C++ build and integration approach.
Do not change Citlali CMake structure, presets, dependency management, CI build
lanes, install/export rules, or cluster build helpers until that direction is
understood. This is a sequencing decision, not acceptance of the current build
as the final reproducible-build solution. Phase 4 continues meanwhile through
strict validation, current baseline/ledger work, controlled performance
evidence, and scientific-contract documentation that is independent of the
eventual compilation strategy.

The first compilation-independent Phase 4 tranche establishes a versioned
validation epoch. Four named profiles pin the current point, OOF, science, and
Beammap snapshots to their required provenance, exact low-level configuration,
and mode-appropriate product comparator. Point `redu66` is the zero-tolerance
structural-closeout snapshot; clean science `redu31` is the current
scientific-tolerance snapshot. One profile-driven command now performs the run
audit, config comparison, and product comparison without duplicating the
existing scientific comparator logic. Accepted snapshots are immutable.
Future intentional algorithm, default, schema, or product changes create a
successor epoch with a predecessor comparison and explicit scientific
rationale instead of silently replacing a baseline or loosening its policy.

The profile command now includes a fourth, versioned scientific-product
contract gate. `validation/product_contracts.json` classifies all accepted
FITS, NetCDF, ECSV, and CSV products: point 21/21, OOF 31/31, science 28/28,
and Beammap 13/13. Configuration-controlled families are evaluated against the
generated merged low-level YAML in both directions: requested output must be
present and disabled output must be absent. Products without an independent
switch remain required companions of their parent output; operational timing
and bounded learning records remain optional diagnostics. The contract records
scientific identity, coordinate frame, axes, units, indexing, missing-value,
and fatal required-write policies while naming existing metadata debt rather
than inventing semantics. See the
[scientific product contract](PHASE4_SCIENTIFIC_PRODUCT_CONTRACT_2026-07-16.md).

The intended post-baseline science-change census is now machine-readable in
`validation/intended_science_changes.json`. It identifies three accepted
imports from `gw_dev`: the RTC/PTC parallel determinism fix, the three-commit
Wiener optimization, and the active-detector PCA optimization. Full source and
integration commits are recorded. Four patch-equivalent cherry picks are
verified mechanically by stable Git patch identity; the manually transplanted
determinism fix is tied to its direct two-run OMP evidence. Every entry records
its expected behavior and numerical/schema effect, affected modes and product
families, accepted-run evidence, and limitations. Scientific behavior already
present at baseline `376e0022` is explicitly inherited rather than relabeled,
and later OG `ffc6b907` is recorded as a comparator rather than an imported
commit. Future non-structural changes require a ledger entry before acceptance.

The canonical human scientific contract is now
[`SCIENTIFIC_CONVENTIONS.md`](SCIENTIFIC_CONVENTIONS.md). It consolidates the
validated identity distinctions, sample/detector/map shapes, coordinate frames,
units, indexing, missing-data rules, requested/effective/observation/realized
semantics, output-failure policy, and active numerical gates. It explicitly
keeps enabled polarimetry and R-channel execution outside the validated
capability boundary and collects unresolved scientific-owner decisions without
inventing answers. During that census, the product registry's RTC/PTC
`output_scan_index` description was corrected to match the writer and NetCDF
metadata: it is the one-based original scan number, while dimension positions
remain zero-based. No product data or schema changed.

The canonical current software map is now
[`ARCHITECTURE.md`](ARCHITECTURE.md). It records the active target and CLI
entry, session/result boundary, runtime and scientific data flow, configuration
state transitions, lifecycle owners, failure and product contracts, and the
allowed direction for new dependencies. It explicitly distinguishes active,
transitional, unbuilt legacy/experimental, and deferred paths. In particular,
`Engine` remains an active but frozen compatibility aggregate, header-defined
mode and numerical code remains transitional, and the four unbuilt historical
main programs are not supported entry points. The document does not disguise
the still header-dominant physical build or authorize the deferred CMake and
dependency work.

The broader section-F.2 exit criteria are now mapped in the
[`Phase 4 closeout census`](PHASE4_CLOSEOUT_CENSUS_2026-07-16.md). Of the 15
criteria, ten are closed by implementation and evidence, two are closed by an
explicit owner scope decision or proportionality exception, and three are
compilation-dependent and remain deliberately deferred. The five focused ADRs
are indexed in `doc/adr/README.md`; root `CODEX.md` is now a concise canonical-
document redirect; and [`RETAINED_DEBT.md`](RETAINED_DEBT.md) records each
deliberate limitation with role owner, reopening trigger, and exit condition.
The adopted external-review Phase 4 compilation-independent criteria are
complete. The project owner subsequently added two explicit pre-integration
stages: the
[`four-mode TolTECA authoring structure`](PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md)
and the
[`whole-code technique/performance review`](PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md).
These additions do not change the census counts or waive the three deferred
build criteria. They do replace the previous instruction to remain idle until
the TolTECA build direction is available.

The TolTECA build owner is unavailable until the week following 2026-07-16.
The project owner authorized a bounded
[`Phase 5 preparation lane`](PHASE5_PREPARATION_AND_INTEGRATION_PLAN_2026-07-16.md)
so closeout planning does not sit idle. This is not formal Phase 5 integration
and does not waive criteria 6, 7, or 10. The source disposition, final
same-SHA validation matrix, and integration packet can be prepared now; CMake
changes, placeholder deletion, final candidate tagging, and integration remain
blocked on the build review. Phase 4.1 and Phase 4.2 are the only additional
tranches authorized by this scheduling workaround.

The controlled-performance evidence path is now specified without touching
deferred compilation infrastructure. A Unity-side wrapper records GNU Time
wall/RSS/I/O data together with Citlali log time, exact config leaves, bounded
input hashes, runtime policy, binary identity, serious log counts, and
profile-stage totals. An offline campaign analyzer requires same-node warmups,
at least three alternating measured Beammap pairs, matched config/input/runtime
policy, complete measurements, an explicit runtime budget, and required RSS
measurement; it reports paired ratios plus median and IQR. The checked-in
campaign is a diagnostic template rather than a mandatory Phase 4 run. If used,
the 5% wall-time ceiling applies and peak RSS remains required evidence with an
evidence-driven limit.

The GNU Time wrapper passed its first live Unity exercise in point `redu67` at
`7ca0be50c`. It captured matching retained and attached evidence, binary and
dependency identities, host/storage/runtime policy, 131.08 seconds external
wall time, 110.477 seconds Citlali time, and 908,316 KB peak RSS. The active
point profile accepted `redu67` against immutable baseline `redu66`: zero
logged issues, zero differences across 490 config leaves, and zero changes in
2,064 records from 19 products. This qualifies the wrapper integration but does
not constitute a Beammap performance conclusion.

The project owner accepted a proportionality exception to a dedicated Beammap
campaign on 2026-07-16. Twelve accepted refactor checkpoints range from
3,397.522 to 4,215.296 seconds with a median of 3,594.693 seconds, move in both
directions, and end with a 1.9% adjacent increase. A prior 13.0% total-time
increase coincided with 1.3% faster mapmaking and was concentrated in
VAST-sensitive PTC and diagnostics I/O. This history and repeated scientific
validation show no sustained regression signal. Serializing Citlali jobs would
not control unrelated VAST traffic, so eight dedicated hour-scale reductions
are not justified now.

Future naturally required Beammap validation should use the wrapper to collect
peak RSS and full provenance. A controlled campaign becomes mandatory only for
a sustained runtime regression, unexplained stage slowdown, memory failure,
peak RSS near node capacity, or a material hot-path change. Profiling overhead
is likewise investigated when a performance signal warrants adding an explicit
control. See the
[controlled performance protocol](PHASE4_PERFORMANCE_PROTOCOL_2026-07-16.md).

A planned post-refactor re-reduction of approximately 50 historical Beammap
observations will provide the broader operational performance census. The
[`corpus plan`](BEAMMAP_CORPUS_PERFORMANCE_CENSUS_PLAN_2026-07-23.md),
manifest template, and offline analyzer are collection-ready as of 2026-07-23.
They reuse the existing GNU Time evidence record, verify observation identity
against Beammap provenance, require one current record per expected
observation, extract workload and output-volume evidence, report population
distributions and workload relationships, and preserve only explicit
same-observation comparisons. Unlike observations are not treated as repeated
trials, and ranked observations are never silently excluded. Eight focused
tests cover completeness, identity, pairing, grouping, workload relationships,
conflicting overrides, and failure behavior. The census itself remains a
future release baseline, not a Phase 4 closeout prerequisite.

Retained-debt item D15 now has a bounded offline evidence path. The
[`fruit-loop convergence study`](FRUIT_LOOP_CONVERGENCE_STUDY_2026-07-23.md)
and manifest-driven analyzer compare consecutive raw coadds for every array,
check support, weights, aperture and peak behavior, and require stable effective
learning state before simulating an explicitly non-production stopping rule.
The five active NGC4449 spatial-feedback iterations pass the study protocol
but do not demonstrate an early stop: the sequence changes non-monotonically
when learned state takes effect, and only the final transition passes the
exploratory all-array rule. The older map-path continuation resets its learning
lifecycle and is not appended as false state-continuous evidence. Production
retains `max_iters`; representative checkpoint-complete sequences and
scientific-owner threshold approval are still required to close D15.

Phase 1 safety stabilization is complete for point, Beammap, science, and OOF.
OOF refactor `redu01` closes the multi-observation date-header gate and is the
accepted comparison against OG `redu00`. Do not reopen typed analysis-control
migration during Phase 4; validation and reproducibility are now the priority.

Operational config migration must proceed one authority domain at a time with
the one-way requested-to-effective-to-realized contract, focused tests, and the
existing mode gates. Compact-config production rollout and open-ended file
splitting remain out of scope.

The initial Phase 3 session boundary is implemented locally. A non-copyable
`citlali::session::ReductionSession` owns sequential run state and returns a
structured `ReductionResult` containing status, diagnostics, product roots,
and published provenance artifacts. Standard reduction loading and processor
selection now execute inside that session. The CLI remains the only layer that
prints result diagnostics and translates success to a process exit code.
Focused tests cover success, exception conversion, failure recovery, two
sequential runs, nested-run rejection, CLI policy separation, independent
header compilation, and multi-translation-unit linkage. Both local test
targets build, all 448 CTests pass, and full config preflight passes. This is
the facade checkpoint, not the Phase 3 exit gate: reachable library exits,
complete internal failure classification, remaining lifecycle ownership cuts,
and validation of the first `.cpp` boundary remain open. The
[bounded ownership plan](PHASE3_LIBRARY_SESSION_PLAN_2026-07-15.md) records the
sequence and stop rules.

The first failure-boundary and exit-census slice is also complete locally.
`ReductionSession` classifies canonical config, I/O, output, runtime, and
internal errors without terminating the process. Eight direct setup exits are
retired without touching numerical loops. An independent scan-context test
found and repaired a real include-order dependency on typed runtime policy.
The new conservative session audit follows 667 project-header dependencies
from the reusable entry and freezes a no-growth baseline of 94 direct library
exits across 22 files, with no CLI exits in the graph. The
[exit census](PHASE3_SESSION_EXIT_CENSUS_2026-07-15.md) defines the bounded
retirement order and separates low-risk setup/output work from mature
timestream and Wiener kernels. The first post-baseline cluster removes all six
TOD output-selection config exits. Invalid strings, empty or nonpositive chunk
lists, negative counts, and impossible selection modes now accumulate atomic,
path-aware config diagnostics. The adjacent effective row-selection boundary
now converts invalid effective modes, empty source-crossing selections, and
out-of-range chunks to canonical errors while preserving valid row assignment.
The current dependency-reachable count is 85. Its isolated test also
characterized the remaining ambient named-logger dependency in the legacy
`get_config_value` helper for later ownership work.

The first observation-input tranche centralizes the three duplicated KIDs
matrix validity checks used by direct, loaded, and gap-aligned RTC input.
Finite matrices retain the same path; NaN and infinite values now become
canonical I/O errors that a `ReductionSession` can report without terminating
the process. Three focused tests cover the contract, and the session audit is
down to 79 dependency-reachable library exits.

The observation/input setup census group is now complete. Detector-count and
cross-network sample-rate mismatches, invalid gap-alignment sample rates,
negative derived extinction, missing polarization calibration groups, invalid
IIR/Nyquist combinations, and Beammap fit-map shape mismatches all use explicit
canonical failure categories. Existing valid setup, metadata reads, and
numerical work are unchanged; the sample-rate path retains one metadata read
per network. Eight focused contract tests pass, and the session audit is down
to 71 dependency-reachable library exits.

Required FITS image and PHDU output-slot validation is now session-safe. Nine
map, Stokes, array, noise-map, and PHDU cardinality exits route through one
canonical required-output failure helper: the library logs the concrete slot
diagnostic and throws an output error, while only the CLI selects a process
exit code. Valid slot lookup and map writing are unchanged. Focused success
and failure tests pass, including every retired branch, and the session audit
is down to 62 dependency-reachable library exits.

The FITS/ECSV adapter tranche completes the output census group. CCfits'
nonstandard `FitsException` hierarchy is caught at operation boundaries and
classified as input I/O or required-output failure; ECSV input and atomic
publication use the same categories. Negative-path tests found and fixed the
distinction between `FitsError` and its sibling open/create exceptions. The
last apparent exit in this group was inside a fully commented, unused Gaussian
transfer-function prototype, which was removed as dead code. The audit reached
57 exits after this tranche.

The final three non-kernel mapmaking preconditions are also session-safe.
Unsupported polarization/grouping combinations, non-altaz Beammap requests,
and missing Wiener template FWHM values now throw canonical config errors;
successful policy and template setup are unchanged. The audit now reports 54
dependency-reachable library exits, all confined to mature RTC, PTC,
timestream, and Wiener implementations. Further retirement must proceed by
measured algorithm-boundary tranche with corresponding mode validation, not by
mechanical replacement.

Run-owned profiling migration is complete locally without changing production
timing records. `ReductionSession` owns and resets a non-copyable
`StageProfileCollector`; the explicit owner now crosses loading, processor
selection, reduction, iteration, observation, generic output, engine setup and
pipeline, Pointing ordered-output, and Beammap internal and specialized-output
boundaries. Output-directory configuration, every production timing scope,
and sidecar publication use that owner. The process-static collector and the
legacy implicit adapter are deleted, and the collector is not stored in
`Engine`.

Tests prove sequential-run reset behavior and verify representative reduction,
observation, and map-output records in the supplied collector. Both local build
targets pass, all 451 CTests pass, and full config preflight passes after the
atomic cutover. Unity point `redu63` confirms unchanged products and profile-
sidecar behavior. Its profile contains the same 76 stage/context records as
accepted `redu62`; only elapsed values and the natural completion order of
concurrent chunk writes differ.

The first concrete lifecycle cut after profiling removes a duplicate collector
reset from `run_reduction_pipeline`. Reset policy now belongs only to
`ReductionSession`, and a regression test proves that records created before
scientific-pipeline entry survive in the same run-owned collector. This is the
bounded stale-state repair required by Phase 3 step 4; no observation or scan
state was moved without a demonstrated hazard.

The first real compiled implementation boundary is accepted.
Timestream enum name tables and parse/format definitions now compile once in
`src/citlali/core/config/timestream_enums.cpp`; the public header retains enum
declarations and small predicates. The header shrank from 946 to 712 lines and
the new source is linked through `citlali`. One immediate before/after CLI
compile pair was 62.4 versus 63.7 seconds, so this slice demonstrates neither a
build-time win nor a material regression. All three local targets build, all
451 CTests pass, and full config preflight passes. Unity compile and point
`redu63` accept the boundary with zero product differences and no runtime
regression attributable to the extraction.

The first bounded mature-implementation exit tranche is accepted for its point
coverage.
Two PTC weighting exits now classify non-contiguous network grouping as input
I/O failure and impossible counters as an internal failure. RTC kernel setup
classifies mismatched kernel-image cardinality as invalid configuration. Valid
paths and numerical loops are unchanged; focused contracts cover each error
class. The dependency audit now reports 51 library exits and zero CLI exits.
Point `redu63` exercises the unchanged valid PTC weighting path exactly. The
next production tranche is fruit-loop map ingestion and requires matched
science and Beammap validation after its local checkpoint.

The fruit-loop map-ingestion tranche is locally complete. All 37 exits in
`TCProc::load_mb` now become canonical config or input-I/O failures at the
session boundary. Required file discovery, FITS header/schema, grouping and map
identity, WCS, and cardinality diagnostics retain their concrete context.
Optional `GROUPING` and `RADESYS` handling ignores only missing-key exceptions,
preventing real schema failures from being swallowed after the move to
exceptions. Valid loading and numerical processing are unchanged. All three
local targets build, all 453 CTests pass, full config preflight passes, and the
session audit is down to 14 library exits with zero CLI exits. Matched science
and Beammap fruit-loop validation is pending.

The three adjacent fruit-loop feedback exits are also retired locally behind a
header-isolated invariant boundary. Non-contiguous calibration grouping,
unknown detector-array identity, and out-of-range map indices now become
session-owned input-I/O failures before the affected map access. Interpolation
and map-to-TOD loops are unchanged. All three local targets build, all 454
CTests pass, and the session audit is down to 11 library exits, all in serial
or OpenMP Wiener filtering. This change shares the pending science and Beammap
fruit-loop acceptance runs.

The Wiener failure-boundary tranche is locally complete. Shared runtime
contracts cover serial and OpenMP template geometry, pixel spacing,
kernel/weight identity and shape, finite kernel peaks, and FFTW resource
creation. The OpenMP allocation path captures exceptions inside each worker,
synchronizes before worksharing, and rethrows only after leaving the parallel
region; partial FFTW resources are reset before failure. Valid filtering and
denominator arithmetic are unchanged. All three local targets build, all 455
CTests pass, full config preflight passes, and the conservative session audit
now reports zero dependency-reachable library or CLI exits. Standard point and
focused full-Wiener point are accepted. Fruit-loop science and fruit-loop
Beammap validation remain required before accepting these final mature
tranches.

The exit audit now also scans every implementation source under
`src/citlali/core`, closing a blind spot in the original header-reachability
census. The wider scan found and retired three invalid APT-table exits and one
invalid Lissajous chunk exit. Manual review confines the remaining textual
exits to successful CLI help/version handling and two legacy main programs that
CMake does not build. No supported non-CLI path retains explicit process
termination.

Unity point `redu64` accepts the standard point path at `6dd0057f8`. Its merged
configuration is byte-identical to `redu63`; the strict complete-product gate
opens every RTC/PTC array and reports 21 common products, 2,041 comparison
records, zero changed records, and zero skipped records. The audit reports 56
files, 22 stable comparable products, 12 PTC chunks, no logged issues, and all
required provenance valid. Total log time is 174.880 seconds versus 169.728
seconds and PTC chunk spacing differs by 0.5%, so no performance regression is
attributed. The queued science config enables fruit loops but retains
`wiener_filter.lowpass_only: true`, so it exercises convolution rather than
Wiener denominator construction. It remains the fruit-loop science gate. A
focused point run with noise maps enabled and `lowpass_only: false` supplies the
full-Wiener denominator gate. The fruit-loop science and Beammap runs remain the
mode-specific acceptance gates for map ingestion and feedback.

The full-Wiener gate is accepted on matched OG `redu10` and refactor `redu65`.
Their 490 low-level leaves differ only in the OG/refactor output directory and
the corresponding telescope-file path; the two telescope inputs have identical
SHA-256 hashes. Both runs execute six Wiener core calls with five noise maps and
`lowpass_only: false`. The strict filtered-product comparison reads seven
products and 148 records with zero changed or skipped records under the
established `2e-8 + 1e-10 * abs(reference)` profile. The pointing-fit table is
exact across all columns. Maximum signal and kernel absolute differences are
`6.34e-9` and `7.99e-10`. Refactor non-uniform denominator work totals 38.4
seconds versus 42.9 seconds for OG; the uncontrolled pair shows no performance
regression. The refactor run has no logged issues and valid required
provenance. OG's twelve `NetCDF: Not a valid ID` records are its known legacy
limitation and are not accepted as refactor behavior.

Beammap `redu06` accepts the remaining mature Phase 3 tranches for that mode.
Its low-level config is byte-identical to accepted `redu05`; both runs complete
198 PTC chunks and expose the same valid provenance and product inventory. The
strict zero-tolerance comparison reads every comparable FITS, NetCDF, and ECSV
product, including complete detector TOD, and reports 12 common products,
16,453 records, zero changes, and zero skips. Total log time is 4,215.296
seconds versus 4,136.440 seconds, a 1.9% uncontrolled difference with no
performance attribution.

The matched science attempt at `6dd0057f8` stopped during configuration before
creating a `reduNN` directory. TolTECA emitted the historical
`timestream.output.rtcdiag.enabled` leaf, which the new complete startup schema
did not recognize. Because that diagnostic prevented installation of the raw
execution adapter, the later kernel-template check reported the misleading
secondary error `wiener filter kernel template requires kernel`. Commit
`7ef43ef93` explicitly classifies the historical switch as an ignored
compatibility spelling: RTC diagnostics remain required and are always
written. The Wiener prerequisite now reads typed raw policy instead of mutable
`rtcproc` state, reducing the checked legacy access census from 44 to 43. The
exact failed science YAML now passes local configuration and reaches the raw
data boundary; all 456 CTests and full config preflight pass. A Unity science
rerun is required before closing the science fruit-loop gate.

The first repaired science rerun was invalidated by two Citlali jobs sharing
the same output root while `fruit_loops.save_all_iters=true`. One job advanced
to `redu26` and attempted to read `redu25/coadded/raw` while the other job was
still writing observation products into `redu25`; the resulting missing-map
diagnostic correctly exposed the incomplete input. This is an output-directory
ownership failure, not evidence of a numerical or fruit-loop ingestion change.
Production session execution now holds a nonblocking filesystem lease on the
configured output root from successful runtime setup through final provenance
publication. A competing Citlali process fails immediately with a required-
output diagnostic, while reductions using distinct output roots remain
independent. Focused tests cover contention, automatic release, independent
roots, and public-header linkage. The CLI build, all 460 CTests, and full config
preflight pass locally. Clean single-job science sequence `redu28` through
`redu31` then completed normally at pre-lease commit `a7a35a00`: every
iteration consumed its immediately preceding complete map directory, the final
run logged no issues, and exact-config scientific equivalence against accepted
`redu23` passed. This closes the fruit-loop map-input repair gate. The output-
root lease then passed its first Unity/VAST exercise in point `redu66`: the
parent log records successful exclusive acquisition, the run completed without
issues, and all non-timing products are exact against `redu65`. This closes the
Phase 3 output-ownership and compiled-boundary gates.

The runtime domain is the first operational Phase 2 migration. Requested,
effective, and realized runtime state are now separate in memory, and execution
consumes the effective thread and runtime policy. Remaining direct mutable
runtime reads are confined to config construction. The required, atomically
published `runtime_provenance.yaml` sidecar uses the stable
`citlali-runtime-provenance-v1` schema. Unity `redu27` validates the sidecar,
zero serious log issues, and exact pre-existing point products. The runtime
domain is complete; the next operational domain is timestream output selection
and chunking.

The timestream-output domain routes RTC/PTC output shape, outer-buffer
allocation, NetCDF serialization mode, metadata, selection, and scan-index
construction through typed configuration. The required, atomically published
per-observation `timestream_output_provenance.yaml` carries the versioned
requested/effective/realized output record. Unity `redu28` validates all 12
selected and realized RTC/PTC chunks, both registered TOD files, zero serious
log issues, and exact existing products. The former processor output-mode and
telescope chunking mirrors are removed; parser and writer boundaries receive
typed values explicitly. The local CLI/test build, all 229 tests, and full
config preflight pass. This domain is complete.

Work has started on the `raw-timestream` domain. Downsample enablement,
requested factor/frequency, anti-alias validation, and effective sample-rate
preflight now use typed raw-time-chunk configuration. Frequency-derived factors
are synchronized into the RTC downsampler only as an execution adapter. A
divergence test proves typed policy wins over stale processor mirrors. Typed
policy also controls FIR/notch/IIR setup, kernel-dependent allocation and
products, flux-unit selection, and extinction setup; processor objects retain
the corresponding numerical state. All 231 tests pass. Remaining RTC flagging,
source-protection, line-audit, and diagnostics boundaries are being migrated in
bounded clusters.

Raw source-protection activation now flows requested typed policy to realized
typed state and then to the RTC execution adapter. Learned-mask application,
FITS event-mask provenance, and RTC diagnostic impulsive-product shape consume
typed policy directly. The shared processed source-protection activation follows
the same direction. All 232 tests pass; line-audit and remaining diagnostic
configuration are the next raw-timestream clusters.

The line-audit cluster now uses typed policy for model-protected PTC audit
activation, model-subtraction requirements, notch-family selection, iteration
counts, frequency overrides, and dynamic edge-guard decisions. RTC diagnostic
sidecars, TOD headers, and chunk summaries serialize requested raw settings from
typed config. Existing RTC notch methods still consume the processor options
object as a numerical adapter, and realized edge-context/guard sample counts
remain processor state. The CLI build, all 232 tests, and full config preflight
pass.

The processed migration now has its first explicit one-way adapter:
`TimestreamFruitLoopsConfig` synchronizes the numerical `PTCProc` fields after
loading. A focused divergence test proves typed values overwrite adapter state.
This enables direct typed parsing to replace legacy parsing incrementally. All
236 tests and full preflight pass.

Direct typed parsing now owns the core fruit-loop lifecycle and model-selection
fields before the one-way processor adapter runs. This is a staged extraction:
expert fruit-loop numerical fields still arrive through the legacy parser and
typed mirror until their cohesive reader is moved. All eight real config
profiles, all 236 tests, and full preflight pass.

The direct fruit-loop reader now covers the complete typed fruit-loop surface,
including expert local-noise, adaptive-support, feedback, interpolation, and
post-addback controls. The legacy combined PTC parser remains temporarily for
other processed domains; fruit-loop execution state is overwritten only from
typed policy through the adapter. All 236 tests and full preflight pass.

Processed cleaning now has a complete one-way typed adapter covering all four
cleaner modes and correlation grouping. The local build caught and corrected
an `int` versus `Eigen::Index` boundary conversion before Unity. All 236 tests,
the CLI build, and full preflight pass.

The cleaning reader now directly owns core activation and mode-selection
policy before the one-way cleaner adapter runs. Expert mode parameters and
eigen-count padding remain in the compatibility parser for the next slices.
All 236 tests, all eight real config profiles, and full preflight pass.

Direct cleaning parsing now includes standard-PCA eigen-count normalization
and both current and legacy key aliases. Empty and short vectors receive the
same defaulting and padding behavior before the one-way adapter runs. All 236
tests, all eight real profiles, and full preflight pass.

Direct cleaning parsing now covers correlation grouping and null-model scalar
policy. Group-name canonicalization remains deliberately mirrored because it
still depends on cleaner-specific helpers. All 236 tests, all eight real config
profiles, and full preflight pass.

Direct cleaning parsing now covers Marchenko-Pastur and adaptive-selector
numerical policy, including adaptive frequency-band validation. The remaining
cleaning-parser dependency is cleaner-specific grouping-name canonicalization.
All 236 tests, all eight real profiles, and full preflight pass.

Raw input and metadata boundaries now use typed policy for duplicate-tone
frequency separation, RTC diagnostic FIR/source-bandwidth ratios, and whether
FITS/TOD tau metadata is calculated. The atmospheric calibration object remains
processor-owned numerical state. The CLI build, all 232 tests, and full config
preflight pass.

Learning collection and learned-mask/exclusion orchestration now reads typed
second-pass source-protection activation, radius, and score thresholds. This
removes another execution-facing dependency on `PTCProc` policy mirrors while
leaving its numerical implementation unchanged. All 235 tests and preflight
pass.

RTC diagnostic and RTC TOD diagnostic schema construction now receives typed
downsample and impulsive-capture policy explicitly. External raw product-shape
decisions no longer depend on `RTCProc` mirrors. Remaining raw-timestream work
is concentrated in numerical-method adapters internal to `RTCProc`; polarimetry
is tracked as a separate authority domain.

The first processed-timestream authority slice now routes fruit-loop
enablement, effective iteration count, retained-iteration output layout,
initial/previous model-map paths, and learning source-model availability
through typed fruit-loop config. Beammap and disabled-loop normalization is
recorded in typed effective state and copied into `PTCProc` only as an execution
adapter. All 232 tests and the full config preflight pass.

Processed-timestream orchestration now also uses typed fruit-loop policy for
model subtraction/add-back, source-subtracted weight retention, final noise-map
population, and beammap adaptive-gate setup. Processor state remains the home
of runtime model buffers and numerical kernels, but no longer decides whether
these operations are enabled. All 233 tests and the full config preflight pass.
Interpolation override selection and fruit-loop runtime-policy logging now use
the same typed authority; the processor retains only the realized interpolation
mode required by map-to-TOD execution. All 234 tests pass.

TOD, PTC-diagnostic, and FITS-map fruit-loop metadata now serializes typed
effective configuration, including array flux limits and pointing source-center
policy. Pointing warnings use the same authority. Runtime detector fit vectors
remain in `PTCProc`. The CLI build, all 235 tests, and full preflight pass.

Compact PTC-diagnostic `CONFIG.*` metadata now also reads typed cleaning,
weight-penalty, busy-row, and second-pass policy. This establishes a consistent
boundary: typed configuration is serialized as policy, while processor-owned
arrays remain realized diagnostics. All 235 tests and full preflight pass.

TOD NetCDF and map FITS cleaning metadata now uses typed processed-timestream
policy throughout. The only retained cleaner value at this output boundary is
the per-array removed-eigenmode count, which is a realized result rather than
configuration. The CLI build, all 235 tests, and full preflight pass.

Weighting metadata now uses typed raw and processed policy for scheme,
cutoffs, hybrid correction, and validation settings. The PTC diagnostic
sampling-window duration remains an explicit realized processor input pending
a typed representation. The CLI build, all 235 tests, and preflight pass.

Optional PTC TOD diagnostic block selection now reads typed processed policy.
Second-pass, correlation, busy-row, and adaptive-cleaner schema decisions no
longer depend on processor mirrors. The CLI build, all 235 tests, and full
preflight pass.

Processed effective-policy resolution is now being separated from YAML
parsing. Pure result types preserve requested values while recording cleaner
group canonicalization, weighting source-mask inheritance, validated-weighting
and busy-row dependency decisions, and disabled/beammap fruit-loop iteration
normalization. Cleaner-mode precedence and fruit-loop interpolation defaults,
overrides, and JINC fallback now use the same pattern; source-protection
activation has an explicit realized-state result. Existing mutating calls
remain thin compatibility adapters with unchanged warnings and processor
values. A non-wired `ProcessedTimestreamExecutionPlan` now provides separate
requested, effective, effective-resolution, and realized storage without
claiming complete output provenance. The CLI and test builds, all 243 tests,
all eight config profiles, and the frozen 171-path PTC boundary audit pass.
The boundary audit now also routes all 171 paths to their declared typed
reader, requires each leaf key in that source, and fails preflight on uncovered
paths or stale compatibility aliases. This mechanically satisfies the path-
coverage prerequisite for removing the legacy parser; the provenance and
cross-mode validation prerequisites remain open.
Focused adapter tests now assign and verify every field copied from typed
fruit-loop, cleaning, weighting, validation, correlation-penalty, busy-row,
and second-pass configuration into the processor compatibility targets. The
full C++ suite passes all 244 tests. The concrete `PTCProc` header remains a
contextual include rather than an isolated test dependency; that existing
header-boundary defect belongs to Phase 3 and was not expanded in this phase.
The non-wired execution plan now has an atomic repeated-run reset operation.
Disabled sections retain their requested parameter values while remaining
inactive, and reset clears all prior effective-resolution and realized state.
All 245 C++ tests pass. Current legacy reader objects are not reset piecemeal;
the contract became operational only with the complete Engine wiring described
below.
Pure YAML component serializers now cover the complete requested/effective
processed snapshot surface. The boundary audit enforces serialization of all
171 frozen legacy paths as well as typed-reader coverage. There is deliberately
no final provenance schema version, output filename, or writer yet; effective-
resolution and realized-state component serialization now also use explicit
availability records. Beammap `redu14` (`4b0126e7`) completed cleanly and
exactly reproduces accepted refactor `redu11` across all 5,234 detector maps.
It also passes the versioned OG scientific-equivalence profile with the exact
artifact-local detector-row/UID set and order (not cross-observation UID
persistence), flags, and product sets. The matched beammap gate is
therefore closed. `Engine` now owns and initializes the processed execution
plan, processed runtime accessors select its effective snapshot, and cleaner,
weighting, source-protection, interpolation, iteration-policy, and completed-
iteration decisions populate explicit resolution or realized records. The
legacy parser remains only as the compatibility seed and no provenance file is
published yet. Unity point `redu34` (`86c47fa7`) passes the strict complete-
product gate against accepted `redu33`: its 489-leaf config is exact, all 13
scientific product families are present, and every RTC/PTC timestream and map
record is exact with zero skipped records. The Engine authority change is
accepted; the versioned provenance root and atomic writer are next.
The v1 processed provenance sidecar is now implemented at the CLI success
boundary. It writes the authoritative plan only after completed iterations,
uses the shared atomic YAML writer, and fails the reduction on uninitialized
state or filesystem failure. Local CLI/test builds, all 252 tests, all eight
config profiles, and the 171/171 boundary audit pass. Unity output validation
of the new required sidecar passes at `81020d46` point `redu35`. The sidecar
contains all five effective-resolution and all three realized-state records;
its schema and hash are recorded in the validation ledger. Against accepted
`redu34`, the merged 489-leaf config and all 13 scientific product families,
including complete RTC/PTC timestreams, are exact with zero skipped records.
Point, beammap, and science processed provenance are accepted. The documented
compatibility-parser removal gate is closed.
Parser-removal preparation now includes a complete default-snapshot parity
test using a real value-initialized `PTCProc`. Typed defaults and the legacy
compatibility snapshot are identical across every serialized processed field.
Together with 171/171 reader coverage and exhaustive one-way adapter tests,
this closes the deterministic omitted-default prerequisite without changing
production parsing. All 253 C++ tests pass.
The six PTC-to-typed mirror calls are now consolidated behind
`seed_processed_timestream_config_from_legacy(...)`. Production still performs
the same compatibility seeding, typed reads, resolution, and one-way adapter
steps, but the legacy parser exit is now one named boundary. Local CLI/test
builds, all 253 C++ tests, config preflight, and provenance-audit tests pass.
Unity beammap `redu15` at `50235fd6` closes the beammap processed-provenance
gate. Its 529-leaf config is exact against accepted `redu14`; all 12 comparable
FITS, NetCDF, and ECSV products are exact with no skipped records; the required
sidecar passes semantic audit; and wall time improved from 3576.607 to 3458.917
seconds. The final matched science pair is OG `ffc6b907` `redu27` and refactor
`50235fd6` `redu24`; the intermediate `reduNN` directories are retained
fruit-loop iterations, not independent runs. Their 502-leaf configs differ only
in input/output path strings. Science-equivalence profile v2 preserves the
`1e-8` raw-map bound and separately enforces the owner-approved 1.5% filtered-
map bound. All 63 raw layers remain within `2.33e-11`; the 21 Wiener-filtered
layers peak at 0.986%; product sets and integer diagnostics are exact; all
other numerical bounds pass. Refactor wall time is 2686.252 seconds versus
2754.146 seconds for OG. The science processed-provenance gate is accepted and
recorded in the validation ledger.
The processed authority migration is now operationally complete in production:
`Engine::get_ptc_config` starts from typed defaults, reads all 171 paths through
typed readers, resolves the effective plan, and populates `PTCProc` only through
one-way execution adapters. The legacy parser call, compatibility seed, and all
processed PTC-to-typed mirrors are removed. The retired-boundary audit rejects
their reintroduction while preserving 171/171 reader and serializer coverage.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and 13
focused Python tests pass.
Unity point `redu36` at `c22bc127` closes the production parser-retirement
gate. Its merged config is an exact 489-leaf match to accepted `redu35`; all 13
scientific product families, including every RTC/PTC array, are exact with zero
missing, extra, changed, or skipped records. Processed and runtime provenance
are byte-identical; timestream-output provenance differs only in the expected
`redu35`/`redu36` paths. The run completed without serious log issues in 53.277
seconds versus 60.159 seconds for the baseline. This acceptance is recorded in
the validation ledger. The frozen 171-path inventory now lives in the versioned
`processed_timestream_legacy_paths.json` manifest. Boundary-audit schema v5
validates its canonical ordering, declared count, and digest before checking
171/171 typed-reader and serializer coverage. The unreachable
`PTCProc::get_config` declaration and roughly 1,190-line body are deleted.
Local CLI/test builds, all 252 C++ tests, all eight config profiles, and eight
focused boundary-audit tests pass after deletion. The processed-timestream
authority migration and its legacy-parser cleanup are complete.
Raw-timestream characterization is the next bounded Phase 2 domain. The frozen
RTC boundary contains 169 raw paths plus two adjacent polarimetry paths,
originally 14 direct parser exits, one production parser call, and ten
legacy-to-typed mirror helpers. The authority inventory now labels raw execution as legacy-authoritative
instead of incorrectly claiming a typed-to-legacy adapter. The finite transition
contract is `doc/raw_timestream_config_transition.md`. No RTC execution behavior
has changed. The non-wired preparation checkpoint now has 169/169 direct typed-
reader and request-serializer coverage. A 40-record external RTC access census
classifies 22 executor operations, six observation-state accesses, seven
output/realized-state accesses, one raw policy read, and four separate-domain
polarimetry accesses, with zero unreviewed records. An unwired execution plan
separates requested, context-free effective, per-observation, and realized
state and resets observation state between runs. All 260 C++ tests, 21 focused
config-tool tests, and all eight config profiles pass. Production remains
legacy-authoritative. The complete unwired typed-to-RTC adapter now covers all
169 raw paths, with a real-`RTCProc` request round trip, disabled-sentinel
checks, and a separate observation-state overlay for sample rate, downsampling,
edge context, source protection, and extinction. The frozen audit enforces
169/169 adapter coverage. All 264 C++ tests, 22 focused config-tool tests, and
all eight config profiles pass. Pure observation resolution now covers native
and effective sample rate, derived downsample factor and anti-alias checks,
filter edge guard/context contributions, source-protection activation, and
extinction-model selection. Filter transient estimates and extinction-model
selection are shared by the typed resolver and legacy processors rather than
duplicated. Focused tests prove edge-guard parity for sum/max policies and
extinction parity across representative tau values. All 271 C++ tests and full
preflight pass. Constructing the typed plan as a non-authoritative production
shadow is the next gate before the authority flip. That context-free shadow is
now active: the Engine directly reads an isolated typed request, constructs the
raw execution plan, adapts into a temporary RTC policy object, and requires its
deterministic 169-path snapshot to equal the legacy parser/mirror snapshot.
Legacy `rtcproc` still drives execution. The frozen audit requires one typed
read before the parser and one comparison after all ten mirrors. The generated
default config, disabled expert semantics, and injected divergence behavior are
covered by focused tests. The per-observation shadow is now active at the
existing lifecycle boundaries: input preparation records and compares native
and effective sample rate, downsample factor, edge guard/context, and raw source
protection; observation setup records and compares extinction activation and
model. Legacy `rtcproc` remains the execution authority. A second observation
resets the first observation's state and realized counters. Frequency-derived
downsampling exposes a pre-existing ordering gap because legacy configures its
edge guard before deriving the factor; that single comparison is explicitly
marked deferred rather than changing numerical behavior. All other divergence
fails with field-level diagnostics. The external RTC census is frozen at 44
classified records with zero review-required entries. Local CLI/test builds,
all 277 C++ tests, 23 focused config-tool tests, all eight profiles, and full
preflight pass. Unity validation of this shadow checkpoint is pending; no raw
authority flip or parser/mirror retirement is permitted before that gate.
The versioned `citlali-raw-timestream-provenance-v1` schema was prepared but not
yet wired at this checkpoint. It serializes the complete requested/effective
config, context-free resolutions, explicit observation-field availability and
edge-guard deferral, an execution-completed marker, and realized counters. Its
atomic writer rejects uninitialized plans and propagates publication failures.
All 281 C++ tests and full preflight pass. Production publication remained
deferred so required-output placement and lifecycle completion could be reviewed
with the Unity shadow checkpoint rather than introduced without mode evidence.
The remaining 14 direct exits in `RTCProc::get_config` are removed. Legacy
cross-field checks now append exact invalid-key paths to the existing config
diagnostics and continue safely through malformed notch vector shapes; the CLI
validation boundary remains responsible for rejecting the reduction. Valid
configuration behavior is unchanged. The frozen raw boundary now requires zero
direct parser exits. Local builds, all 282 C++ tests, and full preflight pass.
Unity point `redu37` accepts the complete raw-shadow checkpoint at `cd8da24f`.
The run used the same 489-leaf merged config hash as accepted `redu36`, completed
all 12 PTC chunks with zero logged issues, and retained the exact 36-file/14
stable-product inventory. Strict comparison including complete RTC/PTC
timestreams found 13 common product families, zero missing or extra products,
zero changed records, and zero skipped records. Runtime and processed
provenance are byte-identical; output provenance differs only in expected
`redu36`/`redu37` paths. Logged runtime was 51.723 seconds versus 53.277 seconds
for `redu36`. This closes the Unity point gate for observation shadowing,
prepared raw provenance, propagated parser diagnostics, and yaml-cpp 0.7
compatibility. Beammap/science evidence remains required before raw authority
flip and parser/mirror retirement.
The accepted point shadow gate now permits required production raw provenance.
Each successfully completed observation atomically publishes
`raw_timestream_provenance.yaml` in its observation directory after required TOD
writers and observation products have completed. The observation lifecycle owns
the completed-scan count and expected required TOD-write count; flagged-sample
and dynamic-notch counts remain explicitly unavailable rather than being
guessed from mutable RTC state. Publication failure propagates and fails the
reduction, and the writer rejects observation, completion, or realized-count
state that is incomplete. Repeated-observation tests prove state reset and
independent sidecars, while a filesystem-failure test proves required-output
propagation. The run-audit tooling can require and semantically validate every
observation's sidecar, including science reductions. It pairs setup-time output
provenance with completion-time raw provenance, rejects missing observation
sidecars, cross-checks scan counts, and validates resolved sample-rate state.
Local CLI/test builds, all 287 C++ tests, 11 provenance-audit tests, and full
config preflight pass. Unity point `redu38` accepts the required raw provenance
at `6bbc12ce`. It has the identical merged config and stable 14-product inventory
as accepted `redu37`, zero serious log issues, and a valid observation sidecar
recording 12 completed scans and 48 required writes. Strict comparison opened
all RTC/PTC arrays across 13 common product families and found zero missing,
extra, changed, or skipped records. Logged runtime was 51.459 seconds versus
51.723 seconds for `redu37`. The point publication gate is closed and recorded
in the validation ledger. Beammap and science acceptance remain pending; raw
execution therefore remains legacy-authoritative.
A science cross-mode attempt at `5d403887` stopped before observation numerical
processing because the shadow compared typed physical downsample factor 1 with
legacy RTC's disabled value 0. Legacy initializes and reads that factor only
when downsampling is enabled, so inspecting it while disabled is outside the
legacy contract and can read inactive state. Observation parity now always
compares enablement and compares factor only when enabled; typed observation
state still records the physical identity factor 1 and unchanged sample rate.
A focused science-style test preserves enabled-factor divergence detection and
accepts the disabled legacy sentinel. Local CLI/test builds, all 288 C++ tests,
and full config preflight pass. The science and Beammap gates must be rerun.
The repaired `2d6f80a3` candidate closes both cross-mode publication gates.
Beammap `redu17` has one complete raw sidecar with 198 scans and 198 required
writes; all 12 complete product families are exact against accepted `redu15`,
with zero skipped records and runtime 3397.522 versus 3458.917 seconds. Science
final iteration `redu29` has two complete raw sidecars, each with 124 scans and
248 required writes; all 27 complete product families pass the strict gate
against accepted `redu24` with zero changed or skipped records. Its largest
absolute difference is `4.452e-10`, within established tolerance, and runtime
is 697.572 versus 705.784 seconds. Both runs have zero serious log issues and
log each published sidecar path. The validation ledger records both accepted
checkpoints. Point, Beammap, and science prerequisites are now satisfied for
the bounded raw execution-authority cutover; OOF reuses the accepted pointing
execution gate, and polarimetry remains outside this authority claim.
The bounded raw execution-authority cutover is now implemented locally. Direct
typed parsing initializes requested/effective plan state and the one-way
production `RTCProc` adapter. The legacy parser and ten mirrors remain only as
a temporary read-only oracle whose deterministic snapshot must match the
production RTC before execution. Focused tests prove stale processor state is
overwritten, disabled requested values remain intact, and divergence fails.
The CLI build, all 291 C++ tests, all eight real config profiles, the complete
169-path boundary audit, and the frozen 44-record execution-read census pass.
Unity point, Beammap, and science cutover validation is the next gate; parser
and mirror retirement is prohibited until it passes.
The first Unity point cutover attempt at `475bf8e22` reached map output but
failed because the production `RTCProc` no longer received the adjacent legacy
polarimetry initialization. For an unpolarized run, that parser side effect
creates the mandatory Stokes-I entry; without it, `stokes_params` was empty and
map indexing read invalid state. A narrow legacy-polarimetry runtime adapter now
copies only enablement, grouping, and Stokes labels from the temporary parser
object. Polarimetry remains outside the raw authority claim. A focused
regression test and the boundary audit require this transfer. The repaired
candidate builds locally, all 292 C++ tests and all eight profiles pass, and
full preflight has zero drift. Unity point cutover validation must be rerun.
The repaired point run `redu40` completes with zero serious issues, all required
provenance valid, and exact scientific products and complete timestream arrays
against accepted `redu38`. The strict gate nevertheless rejects two metadata
records: disabled `CONFIG.TODIIRHP.FREQ_HZ` changed from the established
processor-effective sentinel `0.0` to the preserved inactive request `0.1` in
the RTC and PTC NetCDF products. Raw provenance correctly retains the request
and explicit disabled resolution, so the fix is a pure FITS/NetCDF metadata
projection rather than a plan mutation or processor readback. Disabled IIR
metadata now resolves to frequency `0.0`, order `1`, and zero-phase `false`;
enabled values pass through. All 293 C++ tests and full preflight pass locally.
One final point rerun is required before starting the expensive Beammap and
science cutover gates.
The raw execution-authority cutover validation gate is closed. Point `redu42`
at `880869b3` passes the complete strict comparison against accepted `redu38`:
13 common product families, zero changed or skipped records, valid byte-stable
raw/processed/runtime provenance, zero serious issues, and runtime 54.412 versus
51.459 seconds. Beammap `redu18` at `398d5127` has exact numerical products and
all 5,234 detector results against `redu17`, zero skipped records, valid
byte-stable provenance, and zero serious issues. Its six accepted rtcdiag
metadata changes expose configured values beneath a disabled local-residual
section instead of legacy processor defaults. Science final iteration `redu33`
at `398d5127` passes against `redu29` with 27 common products, zero changed or
skipped records, maximum absolute difference `3.746e-10`, byte-stable
provenance, zero serious issues, and runtime 704.234 versus 697.572 seconds.
The validation ledger records all three accepted gates. OOF reuses the point
execution gate; polarimetry remains separate. The temporary 169-path raw parser
and ten oracle mirrors may now be retired as the next bounded change while
retaining the narrow adjacent polarimetry compatibility boundary.

The authorized raw-parser retirement is complete locally. The declaration and
roughly 1,080-line `RTCProc::get_config` implementation, all ten raw reverse
mirrors, and the context-free parity oracle are removed. The versioned
`raw_timestream_legacy_paths.json` manifest preserves the canonical 171-path
historical surface and digest. The boundary audit now rejects reintroduction of
the parser, a raw mirror, or the parity comparison while continuing to enforce
169/169 direct-reader, serializer, and typed-to-RTC adapter coverage. The two
adjacent polarimetry keys use a dedicated compatibility reader and one-way
runtime adapter; they do not repopulate raw typed state. A forward TOD output-
context helper formerly hidden in the mirror umbrella now has its own named
header. Fresh local CLI, primary-test, and safety-test builds pass all 285 C++
tests; 12 focused raw-boundary audit tests, the unchanged 44-record execution
census, all config profiles, full preflight, and the validation ledger pass.
Unity point `redu43` at `11afd6f6` closes the retirement gate against accepted
`redu42`. The merged 489-leaf config is exact, all 13 product families and
complete RTC/PTC arrays are exact with zero changed or skipped records, all
required provenance is valid, and raw, processed, and runtime sidecars are
byte-identical. Output provenance differs only in the expected reduction-number
file paths. The run has zero serious issues and completed in 54.182 seconds
versus 54.412 seconds. The validation ledger records the acceptance. The raw-
timestream authority migration, including legacy parser/oracle cleanup, is now
complete; polarimetry remains a separate compatibility domain.

The mapmaking authority migration has passed its first Unity mode gates. All
22 frozen `mapmaking.*` leaves now enter typed request state through
one boundary. `MapBuffer`, JINC, maximum-likelihood, observation-map, and
coadd-map configuration no longer parse YAML. One-way adapters construct the
legacy numerical mapmakers and WCS buffers from typed state. The immutable
execution plan preserves the requested grouping while exposing the resolved
effective grouping to downstream accessors; the transitional root request is
no longer mutated by map-count setup. Successful reductions must atomically
publish versioned `mapmaking_provenance.yaml`, and write failures propagate.
The effective plan also records the uncalibrated TOD-type unit substitution
without changing the requested `cunit`. Version-2 provenance now records one
identified observation per input in the final fruit-loop iteration, each
observation's map count, effective pixel size, required logical map-product
count, optional coadd cardinality, and completion state. Lifecycle counters
reset between fruit-loop iterations and advance only after required output
stages return successfully; CLI completion rejects incomplete or inconsistent
counts. The audit accepts historical version-1 sidecars but applies strict
cardinality semantics to version 2. The boundary preflight freezes the
22-path digest, enforces 22/22 reader
coverage, rejects retired parser symbols, and checks the production authority
sequence and provenance writer. Local CLI/test/safety builds, all 305 C++
tests, all eight config profiles, and the full preflight pass. A strict point
run is required first to validate the lifecycle wiring and new sidecar;
Beammap and science runs then validate their mode-specific output cardinality.
This Unity validation is the last mapmaking provenance sub-gate. Unity point
`redu44`, final science
iteration `redu03`, and Beammap `redu00` all embed `5c8f5eb4`; their merged
configs are exact against accepted `redu43`, `redu33`, and `redu18`
respectively. All three runs have zero serious log issues and valid mapmaking,
raw, processed, output, and runtime provenance. Point has 13 exact complete
product families including RTC/PTC TOD. Science has all 27 products with zero
skips and passes the scientific-equivalence profile; its largest map
RMS-relative difference is `5.87e-14`. Beammap has exact non-map products,
exact identity and flags for all 5,234 detectors, and zero RMS difference in
every accepted good/bad signal, weight, and kernel map. Point, science, and
Beammap runtimes are 55.341, 699.904, and 3483.362 seconds, respectively,
versus 54.182, 704.234, and 3580.078 seconds for their baselines.

Version-2 cardinality validation is accepted at `e8e42945`. Point
`redu45` is exact against `redu44`: its 489-leaf merged config is unchanged,
all 13 product families including complete RTC/PTC arrays compare exactly, the
strict audit reports zero issues, and runtime is 56.176 seconds versus 55.341
seconds. Final science iteration `redu07` is accepted against `redu03`: its
502-leaf merged config is unchanged, all 27 products are present with no
skips, the dedicated science-equivalence profile reports a maximum map RMS-
relative difference of `6.23e-14`, and runtime is 709.597 seconds versus
699.904 seconds. Both version-2 sidecars report complete, internally
consistent observation/coadd cardinality. Beammap `redu01` is exact against
`redu00`: its 529-leaf merged config is unchanged, all non-map ECSV/NetCDF
products compare exactly, the artifact-local detector-row/UID set, order, and
all 5,234 flags are exact (without claiming cross-observation UID persistence),
and every accepted good/bad signal, weight, and kernel map has zero RMS
difference. Its strict audit reports zero issues, 198 completed PTC chunks,
and one completed 5,234-map observation with no coadd; runtime is 3449.262
seconds versus 3483.362 seconds. The validation ledger records all three
accepted runs. The mapmaking authority and provenance domain is complete.

The bounded coadd authority domain is implemented locally without changing
coaddition numerics. Its frozen one-path reader owns `coadd.enabled` and
preserves the requested value. `CoaddExecutionPlan` resolves effective
activation from the mapmaking plan without mutating that request. Successful
CLI reductions require atomic `coadd_provenance.yaml` using schema
`citlali-coadd-provenance-v1`; its realized map and required-write cardinality
is a one-way snapshot of the already validated mapmaking coadd lifecycle, and
the reduction audit rejects disagreement between the two sidecars. The legacy
coadd reader and reverse mutation helper are removed. Local CLI/test builds,
all 314 C++ tests, all 38 focused config tests, 24 reduction-audit tests, all
eight config profiles and full preflight pass. Unity point `redu46` at
`c2e053b3` closes the disabled-coadd gate against accepted `redu45`: all 489
config leaves and all 13 complete scientific product families, including RTC
and PTC timestream arrays, are exact with zero skipped records or serious log
issues. The new coadd sidecar records requested/effective disabled activation,
no execution or cardinality, and agrees with the unchanged mapmaking sidecar.
All prior provenance is byte-identical except the expected reduction-number
TOD paths. Runtime is 53.804 seconds versus 56.176 seconds. Final science
iteration `redu11` at `c2e053b3` closes the enabled-coadd gate against accepted
`redu07`: all 502 config leaves match, all 27 products are present with zero
skipped records or serious log issues, and the science-equivalence profile
accepts a maximum map RMS-relative difference of `7.65e-14`. Coadd provenance
records requested/effective enabled, successful execution, three maps, six
required logical writes, and completed outputs; every value agrees with
mapmaking provenance. Runtime is 719.154 seconds versus 709.597 seconds. The
33-record validation ledger passes. The coadd authority and provenance domain
is complete.

The bounded `noise-products` implementation checkpoint is complete.
The six frozen `noise_maps.*` inputs now have one direct typed reader, a
requested/effective/realized `NoiseExecutionPlan`, and a one-way adapter into
the mature observation/coadd map buffers. The existing deterministic Boost
MT19937 identity is now explicit and versioned as fixed internal seed `5489`;
no user-facing seed knob was added. Required atomic
`noise_products_provenance.yaml` records activation/count resolution, final-
iteration observation/coadd realization cardinality, empirical-product count,
realization-image count, and completion. The reduction auditor validates those
semantics and cross-checks scientific-map cardinality against mapmaking v2
provenance. The legacy noise readers and reverse request mutations are retired.
The CLI/test build, all 328 CTest cases, all eight config profiles, the frozen
six-path audit, 48 config-boundary tests, and full preflight pass. No noise-
generation or product algorithm changed.

Unity point `redu47` at `1faec7cc` closes the disabled-noise path against
accepted `redu46`: all 489 config leaves and all 13 complete product families
are exact, with no skipped records or serious log issues. Point `redu49`
closes the bounded full-output fixture with ten realizations per scientific
map, three empirical-product maps, and 30 realization-image writes. Its
realization, empirical-variance, and empirical-weight outputs agree with the
matching OG fixture at maximum RMS-relative differences of `7.65e-14`,
`8.84e-14`, and `6.42e-14`, respectively. The final science iteration
`redu15` closes the generation-only coadd path: six observation maps produce
60 realizations and three coadd maps produce 30, for exactly 90 total with no
optional empirical products or realization files. Its 502-leaf config is
exact against accepted `redu11`; all 27 scientific products are present with
no skips, and the science-equivalence profile accepts a maximum map RMS-
relative difference of `6.93e-14`. Against the matching OG science run, the
profile accepts the previously approved filtered-map differences with maximum
map RMS-relative difference `0.00986`. All three candidate runs have valid
version-1 noise provenance and zero serious log issues. The noise-products
authority and provenance domain is complete.

The bounded pointing implementation is locally complete. Its frozen five-key
surface now has a direct typed request reader, a separate effective execution
plan, and a one-way adapter for the three mature PTC source-center fields.
Effective fit activation preserves the request and depends only on availability
of normalized observation maps from mapmaking. Optional filtering and coaddition
occur downstream and do not disable raw pointing fits. Required atomic
`pointing_provenance.yaml` records the request, resolution decisions,
per-observation map/fit cardinality, and realized completion. The reduction
auditor validates those semantics and cross-checks observation identity and
map counts against mapmaking v2 provenance. The CLI/test builds, all 336 CTest
cases, the frozen boundary audit, all eight compact profiles, and full config
preflight pass. Gaussian fitting, Ceres use, source finding, and map numerics
are unchanged. Unity point validation remains the sole exit gate before this
domain is complete.

The first Unity candidate, point `redu50` at `98d2a5d2`, correctly exposed an
effective-policy error. Its 489-leaf config, maps, timestreams, diagnostics, and
all non-fit products are exact against disabled-noise `redu47`, and it has zero
serious log issues. However, the new plan incorrectly treated disabled map
filtering as making pointing fits unavailable. The resulting three-row pointing
table zeroed all 11 fitted columns instead of preserving the accepted fits. The
gate is failed. Pointing fit availability now follows mapmaking alone; the
semantic auditor rejects the invalid `redu50` sidecar, and focused tests cover
both filter-independent fitting and mapmaking-disabled fitting. Local builds,
all 336 CTests, 43 baseline-tool tests, 54 config tests, all eight profiles, and
full preflight pass. A corrected Unity point run remains required.

Corrected Unity point `redu51` at `a9d17fa1` closes the pointing gate. Its
489-leaf merged config is exact against accepted disabled-noise `redu47`; all
13 scientific product families, including every RTC/PTC timestream record and
all pointing-fit columns, are exact with zero changed or skipped records. The
candidate has zero serious log issues and valid pointing provenance recording
one observation, three scientific maps, three fit attempts, and three valid
fits. Runtime is 59.971 seconds versus 58.627 seconds. The validation ledger
records the accepted checkpoint. The pointing authority and provenance domain
is complete; post-processing is now the active bounded domain.

Post-processing characterization freezes 35 supported leaves: 24 under
`post_processing.*` and 11 under the historical top-level `wiener_filter.*`
prefix. The latter controls filter template construction and convergence and
therefore belongs to the same authority domain. The starting boundary is
intentionally mixed: the legacy Wiener parser still reads 21 leaves and
reverse-mirrors most of them into typed state, while direct typed readers cover
13 other leaves. The initial typed-request gaps,
`post_processing.source_fitting.model` and
`wiener_filter.kernel_template_tail_mode`, now have closed-enum representation
in a complete 35-leaf direct request reader. That reader now runs during
`Engine` config loading as a fail-fast, read-only shadow. Activation and
histogram always compare; detail fields compare only when the legacy path
loads them, so disabled requested values are preserved without false mismatch
reports. The legacy parser and reverse mirrors still drive execution. Focused
shadow tests cover inactive science policy, pointing fit values, active filter
values, and mismatch diagnostics. The CLI/test builds, all 342 CTests, 60
config tests, all eight compatibility profiles, and full preflight pass. See
`doc/POST_PROCESSING_CONFIG_AUTHORITY.md`.

Unity point `redu52` at `d9db1183`, the first enabled-filtering overlay,
reached both the raw and filtered
pointing-fit stages, then failed during lifecycle recording with `pointing fit
results already recorded`. This exposed a provenance-model defect rather than
a fitting or mapmaking failure: version 1 represented only one fit event per
observation even though filtered pointing output deliberately fits the maps a
second time. The execution plan now names the raw and filtered fit stages,
enforces exactly one result per expected stage, and records their cardinalities
separately in `citlali-pointing-provenance-v2`. The reduction auditor accepts
both historical v1 and current v2 sidecars and validates stage expectations
against filtering/coadd policy. Numerical fitting and product-writing order are
unchanged. Local `citlali_cli`/test builds, all 344 CTests, 45 provenance-tool
tests, 60 config tests, all eight compact profiles, and full preflight pass. The
same enabled-filtering point overlay must pass on Unity before post-processing
authority migration proceeds.

Unity point `redu53` at `c75f079b` closes that repair and enabled-filtering
gate. It completes in 59.772 seconds with zero serious log issues. Its v2
pointing sidecar records one observation, three raw and three filtered fit
attempts, all valid, and completed output. All 13 products shared with accepted
unfiltered refactor `redu51` are exact, proving the overlay and lifecycle repair
did not alter the raw path. Against matching OG point `redu09`, all eight
filtered products are present with no skipped or changed records under the
standard numerical gate; the three-row pointing-fit table and 195-row source
table are exact. Maximum filtered signal absolute difference is `2.97e-11`.
The 490-leaf merged configs differ only in their two expected output paths.
The validation ledger records the accepted checkpoint. Post-processing may now
advance from request shadowing to a separate effective execution plan.

The first post-processing authority checkpoint is complete locally. A
`PostProcessingExecutionPlan` now owns the immutable 35-leaf request, a
separate effective snapshot, explicit resolution reasons, and reset realized
state. Effective map filtering and source finding are suppressed only when
mapmaking is unavailable; pointing and Beammap source fitting remains required
whenever mapmaking is available, independent of optional filtering. The plan
is constructed once during config loading and the legacy state is still
compared against its request. Production filtering, finding, fitting, and
output consumers have not been switched yet, so this checkpoint changes no
numerical or output behavior. Focused plan and frozen-boundary tests, all 347
CTest cases, all eight compatibility profiles, and full config preflight pass.
The next bounded cutover at that checkpoint was the one-way typed map-filter
adapter, followed by source finding and source fitting; accepted `redu53` is
the validation baseline after a consumer cutover, not for plan construction
alone.

The map-filter consumer cutover is complete and accepted. The duplicate serial and
OpenMP Wiener YAML parsers and the reverse Wiener-to-typed mirror are removed.
A single one-way adapter copies the effective typed filter snapshot into the
mature numerical target while preserving conditional Gaussian/Airy FWHM
loading and arcsecond-to-radian conversion. Filter activation, runtime noise/
kernel dependency checks, required filtered-output policy, and map-diagnostic
edge-guard metadata now consume effective typed policy. The Wiener algorithms,
map arrays, and output ordering are unchanged. The frozen audit rejects parser,
reverse-mirror, output-policy, or adapter drift. Local CLI/test builds, all 347
CTest cases, 60 config tests, all eight compatibility profiles, and full
preflight pass. Unity point `redu54` at `a89e0ee5` reruns the unchanged
enabled-filtering overlay with zero serious log issues, all required provenance
valid, and the same 21-product inventory as `redu53`. Its 490-leaf merged
low-level config is byte-identical to `redu53`; all 2,041 compared records pass
the established tolerance with no skips, and all 639 non-PTC records compared
against matching OG `redu09` pass as well. The 16 non-bitwise records are
confined to three filtered a1400 products, have no finite-mask mismatch, and
have maximum absolute difference `8.73e-11`.

The source-finding consumer cutover is complete locally. Its duplicate YAML
parser and observation-to-coadd reverse mirror are removed. One adapter writes
`source_sigma`, the arcsecond-to-radian source window, and finder mode directly
from the effective typed plan to the observation map buffer and, when enabled,
the coadd map buffer. Source-finding execution and output activation now use
the same effective authority. The legacy shadow retains activation parity but
no longer compares details that legacy state does not own. Detection, fitting,
map arrays, source tables, and output order are unchanged. Both local targets
build, all 349 CTests and 61 config-boundary tests pass, all eight compatibility
profiles pass, and full preflight is clean. Unity point `redu55` at `aa593a2b`
closes this gate with zero serious log issues, all required provenance valid,
and bit-for-bit identity across all 2,041 records in the 21 common products
against `redu54`, including full RTC/PTC timestreams, 195 source rows, and both
pointing tables. The 490-leaf config is byte-identical to `redu54`; all 639
non-PTC records also pass against matching OG `redu09`. Source fitting is now
the active bounded consumer cutover.

The source-fitting consumer cutover is complete and accepted. The mixed
YAML-to-`mapFitter` parser is removed. A standalone
one-way adapter now projects the effective typed fitting request into the
mature fitter target, preserving arcsecond-to-pixel conversion, fit-angle
policy, two-element amplitude/FWHM vectors, and the historical rule that a
nonpositive limit factor retains the fitter's established default. The
Gaussian fitting implementation and its numerical inputs are otherwise
unchanged. Source-fitting details are no longer copied into or compared
against legacy config state; the temporary legacy shadow now covers only the
remaining activation and histogram values it actually owns. Both local
targets build, all 350 CTests and 62 config-boundary tests pass, all eight
compatibility profiles pass, and full preflight is clean. Unity point `redu56`
at `9f8ad50e` closes the gate with zero serious log issues, all required
provenance valid, the same 50-file inventory, byte-identical 490-leaf merged
config, and bit-for-bit identity across all 2,041 records in the 21 common
products against `redu55`, including the 195-row source table and complete
RTC/PTC timestreams. Realized post-processing state and required provenance are
now the active bounded work.

The realized post-processing implementation is complete and accepted for the
point workflow. Per-iteration state records observation and coadd filter
contexts and map counts; source-finding contexts, detected candidates, catalog
fit attempts/valid fits, and successfully written source-table rows; raw and
filtered pointing fit contexts; and Beammap fit contexts. These fitter families
remain separate by project-owner decision rather than being collapsed into one
ambiguous total. Completion rejects missing or inconsistent cardinality and is
cross-checked against completed mapmaking. Source finding without map filtering
is now a fail-fast configuration error because the supported execution path
operates only on filtered maps.

The CLI publishes required atomic `post_processing_provenance.yaml` using
`citlali-post-processing-provenance-v1` only after successful pipeline output
and realized-state completion; write or lifecycle failures fail the reduction.
The reduction auditor validates internal cardinality and activation semantics,
cross-checks filter map counts with mapmaking v2, and cross-checks raw/filtered
pointing fit totals with pointing v2. The frozen source-boundary audit requires
the lifecycle hooks, schema, atomic writer, and single CLI completion/write
calls. Local `citlali_cli` and `citlali_test` builds pass, all 357 CTests pass,
43 reduction-auditor tests pass, all eight compact profiles pass, and full
config preflight is clean. No filter, source-detection, Gaussian-fit, or map
numerical algorithm was changed.

Unity point `redu57` at `f8a4a596` closes the point gate. It has zero serious
log issues, a valid required sidecar with one observation filter/source/table
context, 195 source rows, and separate three-map raw/filtered pointing fit
contexts. Its 490-leaf merged config is byte-identical to `redu56`, and all
2,041 records in the 21 common products, including full RTC/PTC timestreams,
are exact. Science must still exercise coadd-only filtering/source routing and
Beammap must exercise iterative detector-fit cardinality. Those expensive mode
gates are intentionally batched until after the remaining activation-only
legacy shadow is retired locally; the domain is not complete until both pass.

The activation-only compatibility shadow is now retired locally. The complete
typed post-processing request is loaded once before mapmaking setup, owns the
histogram setting that map buffers consume through the existing one-way
adapter, and initializes the effective execution plan without a second YAML
activation pass. Disabling mapmaking no longer mutates requested filtering,
finding, or fitting policy; effective suppression remains the execution plan's
responsibility. The established no-map Beammap single-iteration optimization
is preserved separately. Both local targets build, all 355 CTests and 63
config-boundary tests pass, all eight compact profiles pass, and full preflight
is clean. This cleanup still requires a point run after Unity compilation; it
is not covered by the preceding `redu57` acceptance. Unity point `redu58` now
closes that gate with the same config and post-processing provenance hashes,
zero serious log issues, and exact identity across all 2,041 records in the 21
common products, including full RTC/PTC timestreams.

The next Beammap authority domain is characterized without changing runtime
behavior. A versioned manifest freezes all 74 `beammap.*` leaves; there are no
known typed-model gaps, and config literals remain confined to the declared
loading and validation boundary. One typed-to-legacy adapter copies only the
fit support radius into the mature `map_fitter`. Dedicated requested/effective/
realized Beammap provenance is explicitly missing. The six-test static audit
is part of the full preflight and will reject surface, reader-boundary,
authority, or adapter drift.

The final post-processing mode gates are accepted. Science final iteration
`redu19` at `342a021c` has zero serious log records and valid required
provenance. Its realized record contains no observation filter contexts and
exactly one coadd filter context with three filtered maps. Against accepted
science `redu15`, the low-level config is byte-identical and the strict full-
depth comparison finds 27 common products, no missing or extra products, no
skipped records, and no changed records outside the standard tolerance.

Beammap `redu02` at the same commit has zero serious log records and valid
required provenance. Its realized record contains exactly three detector-fit
contexts with 15,407 attempts and 15,407 valid fits. Against accepted Beammap
`redu01`, the low-level config is byte-identical and the strict full-depth
comparison, including complete detector TOD and split FITS maps, finds 12
common products with no missing, extra, skipped, or changed records. The
profiling sidecar differs only in elapsed timing and is excluded from the
scientific gate. Post-processing authority and provenance are complete.

Project-owner decision (2026-07-10): every output explicitly enabled in the
configuration is required. RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` write
failures must fail the reduction. There are no best-effort enabled products.

Immediate work order:

1. Begin the bounded Beammap effective-plan and provenance migration using the
   accepted sequence in the Beammap authority review.
2. Ask only the owner questions needed by the first Beammap implementation
   cut; do not silently change phase, prior, split, reference, or source-flux
   behavior.
3. Preserve Gaussian fitting, prior matching, detector flagging, RTC/PTC,
   mapmaking, and all other mature numerical algorithms.
4. Keep compact-config rollout, polarimetry expansion, and Phase 3 compiled-
   boundary work paused.

### Parallel Review Synthesis - 2026-07-14

Three read-only reviews were completed and adopted as advisory detail under
this living roadmap:

- [Phase 2 completion census](../handoff/PHASE2_COMPLETION_CENSUS_2026-07-14.md)
- [Beammap authority design review](../handoff/BEAMMAP_AUTHORITY_DESIGN_REVIEW_2026-07-14.md)
- [compact configuration and TolTECA usability review](../handoff/CONFIG_USABILITY_TOLTECA_REVIEW_2026-07-14.md)

They agree with the active sequence and expose no reason to reopen the nine
completed authority domains. Phase 2 remains incomplete: Beammap and the
minimal KIDs external boundary are implementation-ready; polarimetry and atomic
astrometry/photometry still require scientific-policy decisions. Domain-level
completion must not be mistaken for the global Phase 2 exit gate.

After the post-processing gates close, the adopted shortest sequence is:

1. Complete the bounded Beammap effective-plan and provenance migration,
   preserving all mature numerical algorithms.
2. Complete atomic Beammap photometry observation configuration, including
   replacement rather than merging of per-observation calibrator flux. Keep
   source identity in telescope data and leave flux estimation to TolProj.
3. Record the minimal external KIDs schema/config identity and the durable
   ordered configuration-source manifest.
4. Mechanically disposition polarimetry as either supported and validated or
   rejected as an unavailable capability.
5. Run current matched point, OOF, Beammap, and science snapshots on the final
   Phase 2 candidate before beginning Phase 3.

The frozen 74-leaf `beammap.*` manifest is the correct Beammap policy boundary,
not a claim to contain every scientific input used by a Beammap reduction.
`beammap_source.fluxes` remains an adjacent photometry input. The
review identified a concrete stale-state risk there: a later observation can
inherit a per-array source flux omitted from its own input. The Beammap work
must therefore reference an atomically constructed observation photometry value;
it must not absorb that adjacent domain or preserve merge semantics.

For Phase 2, "reviewed overlay fixtures" means retained matched low-level mode
overlays plus durable ordered-source evidence. Compact-config production
deployment and its full hermetic TolTECA numbered-overlay acceptance suite are
explicitly deferred rollout blockers, not Phase 2 exit requirements. Current
`*_standard` compact profiles remain translation prototypes and must not be
presented as approved operational defaults. Normal compact controls must also
be audited in both directions: user-facing low-level paths must be reachable,
and ordinary compact fields must not write expert-only policy.

Open scientific and operational choices listed in the reviews will be asked
only when the next implementation depends on them. They must not be inferred
silently. In particular, Beammap source-flux failure behavior, phase/prior/
split/reference fallbacks, HWPR and polarimetry capability, astrometry frame
and time rules, supported KIDs types, and ownership of ordered TolTECA source
provenance remain owner decisions.

### Phase 1 Progress

- The 12 `NetCDF: Not a valid ID` errors in `redu21`/`redu22` were traced to
  the PTC TOD stream, one error per requested output scan. The schema omitted
  four second-pass rejection/source-protection variables that the append path
  wrote unconditionally. Signal, flags, weights, and earlier diagnostics had
  already been written before each exception, which explains why pairwise
  numeric comparison passed despite incomplete diagnostics.
- The PTC TOD schema now includes all four fields. A focused NetCDF schema test
  creates the file layout and checks their presence. Local `citlali_cli` build
  and `citlali::safety::ptc_tod_schema.includes_all_second_pass_summary_fields`
  pass. Unity reduction validation is pending.
- CTest is now enabled at the project boundary and the focused safety target is
  discoverable from the normal top-level build directory.
- Parsed enum failures now enter the authoritative invalid-key diagnostics
  instead of silently retaining their typed default. Legacy authoritative
  range parsing and typed validation reject NaN and infinity for ordinary
  numeric fields. The four documented line-frequency inheritance fields retain
  their explicit NaN sentinel but reject either infinity. Focused parser and
  finite-value tests pass locally.
- Required RTC TOD, PTC TOD, `rtcdiag`, and `ptcdiag` NetCDF failures now retain
  the failing path in an error diagnostic and propagate out of the reduction.
  Ordered writers cancel as one output domain, so a failure wakes workers
  waiting on the same or another product stream instead of deadlocking. Focused
  serialization, cancellation, and cross-stream cancellation tests pass.
- A real fixed-size NetCDF failure test now writes the first row, injects an
  out-of-range second write, verifies that a waiting third writer is cancelled
  and the partial product is explicit, confirms a nonzero CLI result, then
  recreates and completes the product with a fresh writer domain in the same
  process.
- The owner-thread failure state now lets Pointing, Lali, and Beammap rethrow
  required output failures after GrPPI worker drainage, so the normal CLI error
  boundary can report them without an exception escaping a worker thread.
- Disabled IIR and extinction mirrors now preserve legacy effective provenance:
  IIR uses frequency `0`/order `1`/zero-phase `false`, and extinction uses
  `N/A`. Enabled values are unchanged. Four focused mirror tests pass.
- Reduction audit comparison now treats any error-level log record as blocking;
  `redu22` correctly fails the audit with 12 errors while the clean `redu23` to
  `redu24` comparison passes.
- Reduction product comparison now has an explicit strict mode. It fails on
  product-set differences, skipped items, or changed records. A complete TOD
  comparison of `redu23` and `redu24` passes with zero changes/skips when the
  volatile profile sidecar is explicitly excluded; retaining that sidecar or
  the default large-array cap correctly fails the gate.
- The pre-existing `citlali_test` target was found to have substantial test
  infrastructure and source decay. It has now been decoupled from the obsolete
  Google Benchmark runner, modernized for typed config and explicit alignment
  and output-path ownership, and reactivated with all 201 declared legacy tests
  passing. The seven utility tests that had remained inside a block comment now
  exercise the current Tula APIs with assertions. Together with the 18 focused
  safety tests, CTest discovers and passes 219 tests with none skipped or
  disabled. The local CLI build and complete config preflight continue to pass.
- Enabled timestream products now carry mode- and config-derived expected write
  counts. Pointing, Lali, and Beammap verify RTC TOD, PTC TOD, `rtcdiag`, and
  `ptcdiag` cardinality after worker drainage and before map finalization, so a
  silently omitted required chunk fails even when no individual write throws.
- Main timestream scan generators now own their cursors per pipeline invocation
  instead of sharing function-local static counters. Focused tests prove exact
  enumeration and a clean scan-zero start after an earlier cursor is abandoned.
- `redu25` (`c2ec8ae5`) finished with zero serious log issues and the same
  complete 33-file/14 stable-product inventory as `redu24`. Scientific arrays,
  maps, and tables are exact. The only strict-comparison differences are the
  intended disabled-IIR effective-provenance changes in RTC/PTC metadata.
- Beammap detector-specific TOD now obeys the required-output policy. Config
  preflight rejects enabled output with no slots or non-detector map grouping;
  unavailable scans, PTC samples, or pointing fail at runtime instead of
  silently skipping the declared product.
- Enabled learning diagnostics now fail on open, write, flush, or close errors.
  Required Beammap PTC TOD metadata updates likewise fail when the file or
  `FRUITLOOPS_ITER` variable is unavailable.
- ECSV table output is now published atomically through a temporary file.
  Failure removes the temporary product and propagates instead of silently
  substituting a differently named ASCII table.
- `validation/accepted_runs.json` is the checked-in machine-readable validation
  ledger. Its first record captures the accepted `redu25` point checkpoint,
  including explicit unavailable provenance and the two intended metadata
  differences. A standard-library validator enforces its core consistency
  rules.
- `redu26` validates the full current Phase 1 checkpoint at `9ef7da8a`. It has
  zero serious log records, the same complete 33-file product inventory and
  merged-config hash as `redu25`, and zero changed or skipped records in the
  strict comparison including every TOD array. Total logged runtime was 59.25
  seconds versus 61.51 seconds for `redu25`; this is recorded as run variation,
  not a performance conclusion.
- Phase 2 preparation now has a checked authority inventory covering 13 config
  domains. It enforces the one-way requested-YAML to typed-config to legacy
  adapter contract and records a concrete exit gate for each domain. Seven
  domains remain materially mixed, four are typed-authoritative without an
  adapter, Beammap is typed-authoritative with one fitting adapter, and KIDs is
  an explicit external boundary. This checkpoint changes no runtime behavior;
  operational authority migration remains gated on the remaining Phase 1
  validation decisions.
- Phase 1 science validation at refactor `redu12` (`59c35e60`) completed both
  observations with 248 PTC chunks, zero logged issues, and the expected 25
  stable products. Against same-config refactor `redu10` (`9ef7da8a`), all 24
  compared FITS/NetCDF products have zero changed or skipped records. Against
  deterministic OG science `redu15`, all nine FITS products remain within the
  current tolerance, while 30 RTC/PTC diagnostic records differ under the
  generic pointwise comparator. Scientific-owner review accepted those
  differences on 2026-07-11: all integer diagnostics are exact, map RMS drift
  is at most `2.31e-11`, PTC weight RMS drift is `2.14e-12`, and the largest
  near-zero detector-median difference is `2.85e-5` absolute and `2.42e-4`
  fractional. The versioned `science-scientific-equivalence-v1` gate enforces
  the accepted bounds and the validation ledger records the checkpoint.
- The intervening science `redu11` failed after observation 0 when observation
  1 metadata loading raised an unqualified NetCDF `No such file or directory`.
  Its merged config was identical to the successful runs. Metadata-load
  failures now report observation index, name, and telescope filepath; all 220
  local tests pass. The successful `redu12` shows this was not a persistent
  numerical or lifecycle failure.
- Beammap refactor `redu10` (`f278bd32`) and `redu11` (`9ef7da8a`) use identical
  merged configs and are numerically repeatable: all six large split FITS
  products, both APT tables, RTC/PTC diagnostics, and the complete detector-TOD
  `signal`/`flags` arrays have zero changed records. The matched OG Beammap pair
  is also deterministic. Scientific-owner review accepted the bounded OG to
  refactor differences on 2026-07-11: the artifact-local detector-row/UID set,
  order, and flags are exact (without claiming cross-observation UID
  persistence); the worst good-detector signal and weight RMS-relative
  differences are
  0.625% and 0.308%; sensitivity differs by at most 0.255%; and positional and
  FWHM differences are sub-microarcsecond. The versioned
  `beammap-scientific-equivalence-v1` gate now enforces these limits and the
  validation ledger records the accepted checkpoint. Any future threshold
  breach is numerical creep and requires investigation rather than automatic
  tolerance relaxation.

The first Beammap authority preparation checkpoint is complete locally without
changing production execution. Mechanical boundary checks expand 59 typed
reader roots and 59 serializer roots to exact 74/74 frozen-path coverage. A
pure, production-unwired `BeammapExecutionPlan` preserves requested values and
separately characterizes current phase correction, prior inheritance and
missing-path disablement, split-flag normalization, convergence availability,
and mapmaking-disabled iteration policy. Cold-boundary validation now rejects
non-finite Beammap vector and scalar values and enforces reader-established
vector cardinality. The existing typed request and one-way fitting adapter
remain production authority, and dedicated Beammap provenance remains missing;
this is preparation for a later bounded consumer cutover, not a completed
migration claim.
The local CLI and test targets build, all 363 CTests pass, and full config
preflight passes 74 boundary tests, all eight compatibility profiles, and the
complete authority audit suite. Because the plan and serializer are explicitly
unwired, this checkpoint does not require a Unity reduction.

## Beammap Effective-Plan Boundary Activated

The next bounded checkpoint constructs `BeammapExecutionPlan` in production
from one raw 74-leaf request plus explicit-key presence. Policy correction no
longer mutates values inside the family YAML readers. The immutable request is
preserved while a separate effective snapshot records phase correction, prior
inheritance and missing-path disablement, split-flag normalization, convergence
availability, and mapmaking-disabled iteration behavior.

Existing mature Beammap algorithms temporarily consume a one-way copy of the
effective snapshot through `ReductionConfig::beammap`, preserving their current
inputs without creating reverse synchronization. The existing map-fitter
radius adapter is the first bounded consumer to read effective plan policy
directly. The boundary audit enforces the ordered read/resolve/install/adapt
sequence and rejects reintroduction of the retired reader-side mutation
helpers. Dedicated Beammap realized lifecycle and provenance remain missing,
and the component serializer remains unpublished.

Local verification is clean: `citlali_cli` and `citlali_test` build, all 364
CTest cases pass, and full config preflight passes 74 tests, all eight compact
compatibility profiles, 100% compact-surface coverage, and every authority
audit. This changes production configuration construction, so the eventual
Beammap provenance checkpoint requires a Unity compile and matched Beammap
reduction before the domain can be accepted. The next local work is realized
iteration/output state and required atomic provenance; do not spend a Beammap
run on this intermediate commit alone.

## Beammap Realized Lifecycle And Provenance Prepared

The next local checkpoint adds an explicit Beammap observation and internal-
iteration lifecycle around the established execution without changing its
numerical control flow. Each enabled-mapmaking observation records identity,
detector/map/scan counts, contiguous iteration indices and phases, active map
counts, one or two completed mapmaking passes, the source-aware RTC decision,
fit completion, newly/total converged maps, and maximum-iteration or all-maps-
converged termination. Disabled mapmaking records a successful zero-product
execution instead of manufacturing observations or fit contexts.

Completion requires every internal stage and observation output to finish. It
then cross-checks Beammap observation identity/map counts against the completed
mapmaking plan and requires the post-processing Beammap fit-context count to
equal the exact number of completed internal iterations. Map write counts and
fit attempt/valid aggregates remain owned by their existing plans rather than
being copied into Beammap state.

Successful Beammap reductions now require atomically published
`beammap_provenance.yaml` with schema `citlali-beammap-provenance-v1`. The file
contains the complete requested and effective 74-leaf snapshots, effective-
resolution reasons, observation/iteration lifecycle, and terminal realized
state. Incomplete lifecycle and publication failures propagate to the CLI.
The strengthened boundary audit requires all lifecycle hooks, exact 74/74
reader and config-serializer coverage, and one ordered CLI completion/write
path.

Local verification is clean: both build targets pass, all 372 CTests pass,
and full preflight passes 75 Python tests, all eight compatibility profiles,
100% compact-surface coverage, and every authority audit. The authority
inventory deliberately remains `partial` until a matched Unity Beammap run
accepts this sidecar and scientific products. Observation-resolved prior and
reference decisions, adjacent atomic `beammap_source.fluxes` state, and any
additional Beammap-specific optional-product cardinalities required by the
design review remain bounded follow-up work; this checkpoint does not claim
the Beammap domain complete.

Enabled detector-specific Beammap PTC TOD is now an explicit required
observation product in the realized plan. The record is updated only after the
existing atomic NetCDF writer returns and captures the output iteration plus
detector, slot, and maximum-sample dimensions. Observation completion requires
exactly one such write when `beammap.detector_tod_output.enabled=true`, rejects
duplicates, and requires zero writes when disabled. This implements the
project-wide enabled-output decision without selecting new scan slots or
changing the detector-TOD numerical content.

The CLI and test targets build, all 373 CTests pass, and full preflight remains
clean with 75 Python tests and all eight compatibility profiles. This is part
of the pending Beammap Unity validation candidate, not a separately accepted
domain gate. Prior/reference and split-output fallback policies remain
unchanged and unresolved owner decisions are not inferred.

## Beammap Lifecycle Gate Accepted

Unity Beammap `redu03` was produced by `v4.0.0-3486-gb530e838` from the same
low-level configuration as accepted `redu02` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed 198 PTC chunks in 3,609.307 seconds with zero error-,
critical-, or fatal-level log records. The required
`citlali-beammap-provenance-v1` sidecar records one 5,234-detector/map
observation, three contiguous completed Beammap iterations, one mapmaking pass
per iteration, the expected source-aware RTC rerun on iteration one,
maximum-iteration termination, and exactly one required detector-TOD write at
iteration two with shape 5,234 detectors by 20 slots and 788 maximum samples.

Against `redu02`, the merged configuration is byte-identical. The accepted
Beammap profile reports the exact artifact-local detector-row/UID set and order
(not cross-observation UID persistence), flags, APT quantities, and all
good/bad signal, weight, and kernel maps. The strict full-depth comparison
excludes only volatile `citlali_profile.ecsv` timing, reads all 12 scientific
products including detector TOD and six split FITS files, and finds no missing,
extra, skipped, or changed records.

The standard reduction audit now recognizes and can require Beammap provenance.
It validates observation/iteration lifecycle, terminal state, convergence
accounting, detector-TOD cardinality and shape, and cross-checks observation
identity/map count against mapmaking plus iteration count against
post-processing fit contexts. This closes the pending lifecycle/provenance
validation checkpoint, but the Beammap authority domain remains partial until
observation-resolved prior/reference state and adjacent atomic
`beammap_source.*` handling are completed. No unresolved fallback policy is
inferred by this gate.

## Atomic Beammap Photometry State Accepted

The adjacent photometry safety cut removes the concrete
cross-observation source-flux hazard without changing successful numerical
behavior. `beammap_source.*` is parsed into a temporary observation value and
all required runtime-array fluxes are validated before any Engine state is
mutated. Successful installation replaces typed photometry and the legacy
mJy/beam map and clears the derived MJy/sr map; it never merges with an
earlier observation. Missing or invalid required flux retains the established
fatal reduction outcome, but now throws a typed invalid-config error instead
of calling `exit()` inside `Engine::get_photometry_config`.

Project-owner clarification (2026-07-15): source identity belongs to telescope
data and TolProj owns calibrator selection and flux estimation. Citlali must
not mirror source name or coordinates into this config domain. Beammap
provenance therefore advances to `citlali-beammap-provenance-v2` with
`telescope_data` named as the source-identity authority and only the installed
per-array flux/uncertainty recorded as Citlali photometry input. The reduction
audit accepts historical v1 sidecars and requires this ownership record for
v2.

Project-owner decision (2026-07-15): every runtime array requires a positive,
finite calibrator flux; missing or invalid required flux fails the reduction.
No fallback is permitted.

Unity Beammap `redu04` was produced by `v4.0.0-3489-g7e577c81` from the same
byte-identical low-level config as accepted `redu03` (SHA-256
`aa956b28465eaef8b23763e877857b5b8929e95ca4fbdc976db6d7b2a775636d`).
The run completed all 198 PTC chunks with zero error-level messages. Its valid
`citlali-beammap-provenance-v2` sidecar names telescope data and TolProj as the
respective source-identity and calibrator-flux authorities and records the
three required installed array fluxes. The strict full-depth comparison reads
all 12 scientific products, including detector TOD and six split FITS files,
and finds no missing, extra, skipped, or changed records. The dedicated Beammap
profile also reports the exact artifact-local detector-row/UID set and order
(not cross-observation UID persistence), flags, APT quantities, and good/bad
signal, weight, and kernel maps.

The total log interval is 3,661.793 seconds versus 3,609.307 seconds for
`redu03` (+1.45%). The dominant mapmaking interval is 0.53% faster; the
variation is concentrated in PTC chunk and diagnostics timing. This is within
the provisional 3-5% runtime budget and does not indicate a provenance-path
regression. Peak RSS remains unmeasured.

Both local targets build; all 24 focused Beammap/photometry tests, all 377
CTests, and all 49 reduction-audit tests pass. Full config preflight passes 75
tests, all eight compatibility profiles, 100% compact coverage, and every
authority audit.

## External KIDs And Config-Source Provenance Prepared

The bounded external KIDs checkpoint preserves Kidscpp as the numerical
execution authority while recording the exact bridge identity Citlali uses.
All four solved TOD representations (`xs`, `rs`, `is`, and `qs`) are supported.
The requested fitter/solver values, effective values, selected TOD type,
TolTEC data schema, and Kidscpp build version are separate fields in the
required atomic `citlali-kids-external-provenance-v1` sidecar. Historical
`solver.extra_output` behavior remains disabled and is now recorded explicitly
instead of being controlled by a header-level global.

The same successful CLI boundary now requires
`citlali-config-source-manifest-v1`. It records the ordered files actually
passed to Citlali, collision-safe copies, byte sizes, SHA-256 digests, and the
canonical merged YAML snapshot. TolTECA remains the owner of numbered
`NN*.yaml` discovery and upstream merge provenance; the record explicitly says
that TolTECA's complete ordered authoring-source list is not currently passed
to Citlali. Citlali does not guess or duplicate that merge.

Local CLI and test builds, all 382 CTests, 52 reduction-audit tests, and the
full 78-test config preflight pass. Unity point `redu59` identifies
`d016e1a64`, has zero serious log records, and passes semantic and digest
audits for both new records. Its low-level config is byte-identical to accepted
`redu58`; the strict full-depth comparison reads all 21 scientific products,
including complete RTC/PTC timestreams, with zero changed, skipped, missing, or
extra records. The external KIDs and Citlali CLI config-source checkpoint is
accepted. Complete upstream `NN*.yaml` provenance remains a future TolTECA
interface responsibility rather than a Citlali reconstruction task.

## Polarimetry Capability Disposition Accepted

The project owner intends Citlali to become the center of polarimetry
reductions, but not in the present refactor and not without an enabled
validation dataset. Phase 2 therefore preserves polarimetry as a planned
capability while mechanically rejecting `timestream.polarimetry.enabled: true`
before reduction execution. The exit condition is an approved polarimetry/HWPR
scientific contract plus an enabled end-to-end reference gate.

The frozen three-leaf request now has one direct typed reader, one immutable
request/effective capability plan, and one forward adapter into `RTCProc` and
`Calib`. The temporary legacy compatibility reader and reverse mirror are
removed. There is no separate `calibration.ignore_hwpr` YAML input; that name
was stale inventory text referring to the legacy adapter target. Disabled
reductions retain Stokes-I initialization and the established default values.

Successful reductions now require atomic
`citlali-polarimetry-provenance-v1`, recording the capability disposition,
requested/effective policy, accepted resolution, and realized non-execution.
The dedicated static audit freezes the boundary, while the reduction auditor
semantically rejects enabled or executed polarimetry in a successful run.
Local CLI and test builds, all 386 CTests, 54 reduction-audit tests, and the
full 82-test config preflight pass.

Unity point `redu60` identifies `db22bca1f`, completes all 12 PTC chunks in a
67.032-second total log interval, and has zero error-, critical-, or fatal-level
records. Its required v1 sidecar records the planned-unavailable capability,
an accepted disabled request, a disabled effective plan, completed reduction,
and no polarimetry or HWPR execution. The low-level input is byte-identical to
accepted `redu59`; the strict zero-tolerance comparison reads all 21 stable
scientific products, including complete RTC/PTC timestreams, with no changed,
skipped, missing, or extra records. The disabled capability boundary is
accepted. Enabled polarimetry remains planned but unavailable until its
scientific/HWPR contract and enabled reference gate are approved.

## Observation-Resolved Astrometry Candidate

The astrometry calibration-item loader now constructs the complete typed
pointing-offset request before touching observation runtime state. Structural
and finite-value validation runs on that temporary value, and a single forward
adapter then replaces both the typed request and the legacy Eigen vectors.
Invalid input throws the normal typed invalid-config error; the loader no
longer calls `exit()` or builds typed policy by mirroring partially mutated
runtime state. Legacy named axes, positional axes, one/two-value shapes, and
non-positive MJD sentinel normalization are preserved. The interpolation
kernel and its existing no-extrapolation behavior are unchanged. The remaining
interpolation failures now propagate as typed exceptions rather than terminating
the process from library code; successful numerical behavior is unchanged.

The project owner approved the legacy application contract. TolTECA selects
pointing support: two bracketing pointing observations produce interpolated
offsets, one pointing produces constant offsets, and no pointing observations
leave the explicitly configured offsets in force. Citlali applies the supplied
values. Positive MJD endpoints must remain strictly increasing, bracket the
whole observation, and are never extrapolated. Citlali does not receive the
upstream support-selection metadata, so it records that origin as unspecified
rather than inferring whether a constant came from one pointing or direct
configuration.

An observation-indexed execution plan now retains each immutable request,
effective application mode (`constant`, `observation-span-linear`, or
`explicit-mjd-linear`), observation number, installation/application counts,
and telescope sample count. Successful CLI completion requires atomic
`citlali-astrometry-provenance-v1`. Its authority record names TolTECA for
calibration selection and Citlali for application. A semantic reduction audit
and a static config-boundary audit reject incomplete lifecycle, malformed
offsets, inconsistent modes, authority drift, reverse mirrors, process exits,
or a missing required write.

The CLI and test targets build; all 398 CTests, 60 reduction-audit tests, and the
full 84-test config preflight pass. The combined astrometry/photometry domain is
still marked partial until Unity validates the new required sidecar and
scientific equivalence. The next gate should include a point reduction, then a
multi-observation OOF reduction because that fixture exercises observation
identity and stale-state isolation most directly. Beammap should follow before
the combined domain is marked complete.

## Astrometry Point Gate Accepted

Unity point `redu61` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level and canonical merged configuration as accepted
`redu60`. It completed all 12 PTC chunks in a 63.741-second total log interval
with zero error-, critical-, or fatal-level records. Every applicable required
provenance record passes semantic audit.

The new `citlali-astrometry-provenance-v1` sidecar records TolTECA as calibration-
selection authority and Citlali as application authority without claiming
unavailable support-origin metadata. Observation 152389 has one requested and
effective zero-valued az/alt correction, constant application mode, one atomic
installation, one application, and 7,697 telescope samples. The reduction is
complete.

The strict zero-tolerance comparison against `redu60` reads all 21 scientific
products and 2,041 records, including complete RTC/PTC timestreams, with zero
changed, skipped, missing, or extra records. The point checkpoint is accepted.
The combined astrometry/photometry domain remains partial until a multi-
observation OOF run validates observation identity and stale-state isolation,
followed by a Beammap run validating the adjacent accepted photometry contract.

## Astrometry Multi-Observation OOF Gate Accepted

Unity OOF `redu02` was produced by `v4.0.0-3496-g9ea6d7f0` from the same byte-
identical low-level configuration as accepted refactor `redu01`. It completed
all 18 PTC chunks for observations 152385-152387 in a 40.667-second total log
interval with zero error-, critical-, or fatal-level records. All applicable
required provenance records pass semantic audit.

The astrometry sidecar contains three contiguous observation identities. Each
was installed and applied twice, once during initial geometry and once during
the reduction iteration, with stable per-observation telescope sample counts.
This closes the multi-observation replacement and stale-state-isolation gate.
TolTECA supplied a constant zero-offset request for each observation, so this
fixture does not provide an end-to-end positive-MJD interpolation test; that
limitation is retained explicitly rather than overstating the evidence.

The strict zero-tolerance comparison against accepted refactor `redu01` reads
all 30 configured products and 1,941 records with zero changed, skipped,
missing, or extra records. Direct comparison against OG `redu00` reproduces the
same nine previously accepted inactive RTC-despike metadata differences; all
scientific numeric differences remain within the standard OOF tolerance. The
OOF checkpoint is accepted. Beammap remains the combined astrometry/photometry
gate, and science remains required for the final Phase 2 snapshot matrix.

## Astrometry Science Interpolation Gate Accepted

Unity science `redu20` through `redu23` was produced by
`v4.0.0-3496-g9ea6d7f0` from the same byte-identical low-level configuration as
accepted `redu16` through `redu19`. Final `redu23` completed 248 PTC chunks in a
711.330-second total log interval with zero error-, critical-, or fatal-level
records. Every science-applicable required provenance record passes semantic
audit.

The astrometry sidecar records observations 152390 and 152392 with distinct,
strictly increasing positive-MJD support pairs and `explicit-mjd-linear`
effective mode. Each observation was installed and applied five times, once
during initial geometry and once in each of four fruit-loop iterations, with
stable telescope sample counts of 151,535 and 151,941. Successful completion
also proves that each support pair bracketed its complete telescope timestream;
the unchanged application kernel forbids extrapolation.

Every retained fruit-loop iteration passes the standard strict science gate:
`redu16`-`redu19` versus `redu20`-`redu23` each has 27 common products and 1,478
comparison records, with zero missing, extra, skipped, or out-of-tolerance
records at `2e-8 + 1e-10 * abs(reference)`. A zero-tolerance probe sees only the
expected tiny OMP run-to-run drift. The science and explicit-MJD interpolation
checkpoint is accepted. Beammap is the final mode gate for the combined
astrometry/photometry authority domain.

## Astrometry And Photometry Beammap Gate Accepted

Unity Beammap `redu05` was produced by `v4.0.0-3496-g9ea6d7f0` from the same
byte-identical low-level configuration as accepted `redu04`. It completed all
198 PTC chunks with zero error-, critical-, or fatal-level records. Every
Beammap-applicable required provenance record passes semantic audit.

The version-two Beammap provenance is identical to `redu04`: one 5,234-map
observation, three completed iterations, 15,407 valid detector fits, exact
telescope-data source identity and TolProj flux authority, three atomically
installed array fluxes, and one required 5,234-detector by 20-slot TOD write.
The added astrometry record captures one constant zero-offset application over
383,699 telescope samples without changing that accepted photometry contract.

The zero-tolerance full-depth comparison reads all 12 products and 16,453
records, including complete detector TOD and six split FITS cubes, with zero
changed, skipped, missing, or extra records. The dedicated Beammap scientific-
equivalence profile reports the exact artifact-local detector-row/UID set and
order (not cross-observation UID persistence), flags, APT quantities, and
signal/weight/kernel maps for all 4,980 good and 254 bad detectors.

The 4,136.440-second total interval is 13.0% slower than `redu04`, but the
dominant mapmaking interval is 1.3% faster. The increase is concentrated in PTC
and diagnostics I/O before mapmaking, outside the astrometry change. Record the
variation without attributing it or treating one uncontrolled Unity comparison
as a performance conclusion; controlled performance/RSS certification remains
Phase 4 work.

The combined astrometry/photometry authority domain is complete. All 13 domains
in the original operational migration matrix have complete migration and
provenance disposition. The global F.1 leaf census and document/ledger
reconciliation remain before changing the active phase to Phase 3.

## F.1 Leaf Census Checkpoint

The owner approved the generated low-level Citlali YAML as Citlali's immutable
configuration/provenance boundary. TolTECA owns discovery, ordering, and merge
semantics for upstream `NN*.yaml` authoring files and must eventually record
that upstream provenance; Citlali records exact source bytes and ordered paths
from the generated low-level input onward. This is an explicit boundary
decision, not an inference that Citlali received unavailable source metadata.

The checked F.1 leaf contract resolves the union of `data/config.yaml` and the
four retained point, OOF, Beammap, and science low-level fixtures. It records
573 unique leaves, including 572 executable leaves and one explicitly ignored
deprecated leaf. Every record has a machine-readable authority, typed or
external owner, unit, allowed value-domain class, mode applicability, lifecycle
classification, resolution stage, and validation source. The preflight fails
on an uncovered leaf or drift from the resolved manifest.

This census exposed two real closeout omissions hidden by the earlier broad
subsystem grouping: 28 `timestream.learning` leaves executed from a legacy
options object populated in parallel with the typed request, and 14
`interface_sync_offset` leaves executed from an untyped mutable map with
permissive duplicate handling. They are now explicit `learning` and
`interface-sync` authority domains. Both are locally migrated through immutable
typed request, one-way adapter, validation, and versioned provenance. No
scientific algorithm or reduction behavior changed in either migration.

The learning omission is now locally migrated. All 28 leaves parse directly
into immutable `TimestreamLearningConfig`; one one-way adapter constructs the
unchanged `ReductionLearningState::Options` numerical input. The processed-
timestream requested/effective snapshots and versioned provenance now include
the complete learning policy. A frozen 28-path audit rejects reverse mirrors,
reader drift, incomplete adapter coverage, or missing serialization. Local CLI
and test builds plus focused reader/adapter tests pass. Because the standard
point fixture enables learning, an exact point Unity gate is the remaining
condition before marking this closeout domain complete.

The interface-sync omission is also locally migrated. All 14 TolTEC/HWPR
offsets parse atomically into immutable typed request state. Duplicate,
unknown, malformed, and non-finite entries are fatal; omitted interfaces retain
the established zero-second default with an explicit warning. One adapter
populates the unchanged alignment map. Raw-timestream provenance version 2
records requested and effective offsets with seconds as the explicit unit. A
frozen 14-path audit rejects reader, adapter, or provenance drift.

The F.1 startup gate is now operational rather than documentary. A generated
allowlist covers every normalized node in the checked 573-leaf contract and
the retained default configuration. Unknown nodes, including unknown empty
containers, enter fatal config diagnostics before execution. The `inputs`
subtree is deliberately excluded because its schema is owned by TolTECA; all
other low-level nodes are Citlali-owned. Typed validation errors now enter the
same fatal diagnostics instead of being logged as advisory mirror warnings.
The existing observation-scoped astrometry and photometry gates remain atomic.

The detailed [Phase 2 F.1 closeout](../handoff/PHASE2_F1_CLOSEOUT_2026-07-15.md)
maps every adopted checklist item to code, audit, and reduction evidence. Local
`citlali_cli` and test builds, all 410 CTests, all 96 config tests, eight compact
compatibility fixtures, 100% compact-surface coverage, and every boundary audit
pass. Unity point `redu62` closes the final gate as recorded below.

## Phase 2 Final Point Gate Accepted

Unity point `redu62` identifies `v4.0.0-3503-g9a3901e9` and the expected commit
`9a3901e91`. Its generated low-level input is byte-identical to accepted
`redu61`. It completed all 12 PTC chunks with zero error-, critical-, or
fatal-level records. Every required provenance sidecar passes semantic audit.

Processed-timestream provenance contains the complete 28-leaf requested and
effective learning policy exercised by the standard point fixture. Raw-
timestream provenance v2 contains all 13 TolTEC interface offsets plus HWPR in
requested and effective state, with unit seconds and exact equality. The
configuration-source manifest and canonical merged input are valid.

The strict zero-tolerance full-depth comparison reads all 21 stable products
and 2,041 records, including every RTC/PTC array. It reports zero changed,
skipped, missing, or extra records. The final F.1 gate is accepted; all 15
authority domains now have complete disposition and Phase 2 is complete.

The run took 176.435 seconds versus 63.671 seconds for `redu61`. The difference
is isolated to filesystem-facing stages: observation file setup increased from
1.758 to 28.723 seconds, raw/filtered output from 6.136 to 52.767 seconds, and
the 48 chunk-write calls averaged 4.172 rather than 2.482 seconds. Map
filtering, diagnostics, fitting, and other computational stages remained near
their prior timings. Treat this as an uncontrolled Unity/VAST I/O observation,
not a Phase 2 code-performance regression. A same-SHA rerun may characterize
the storage variance but is not required for scientific acceptance.

## Roadmap With Owner-Added Bridge Stages

### Phase 1 - Safety Stabilization

Repair output and run-success contracts, config parsing and finite-value
validation, output schema/cardinality checks, and ordered-writer cancellation.
Add injected failure and repeated-run tests without rewriting mature numerical
algorithms.

Exit gates:

- An injected required write failure returns a nonzero CLI status.
- Ordered output cannot deadlock after failure, and partial products have an
  explicit diagnosed disposition.
- A subsequent reduction in the same process starts with clean state.
- Invalid enums, NaN, and infinity fail with actionable config paths.
- The current point run has zero unexpected error-level messages and passes a
  strict complete-TOD and metadata comparison.

### Phase 2 - Config Authority And Provenance

Build the one-way flow from immutable requested config to effective execution
plan to realized observation metadata, with a temporary one-way legacy adapter.
Fix disabled-option provenance, atomic observation config, stale beammap flux
state, and typed/legacy parity checks. Validate real TolTECA overlay behavior
before compact config becomes operational.

Exit gates are the complete current-config definition of done in section F.1 of
the external review, including one authority per migrated field, no fallback to
raw YAML in migrated execution paths, correct provenance, and reviewed overlay
fixtures for each supported reduction mode.

### Phase 3 - Library, Session, And First Compiled Boundary

Introduce a minimal non-CLI reduction session/result boundary, remove reachable
library exits, give run/observation/scan state explicit owners, and freeze
`Engine` as a compatibility adapter. Add header-isolation and multi-translation-
unit checks, repair ODR hazards, and move one measured, coherent declaration and
validation tranche into `.cpp` files.

Exit gates:

- CLI policy is outside the library boundary.
- Sequential reductions in one process are clean and supported.
- Lifecycle state is reset by ownership rather than scattered cleanup.
- The first compiled boundary reduces dependency exposure without a material
  build or runtime regression.
- Further extraction has a named ownership or contract benefit; textual
  subdivision alone is not sufficient.

### Phase 4 - Validation, Performance, And Reproducible Build

Make strict comparison and active tests pinned CI gates. Add hermetic fixtures,
version/dependency provenance, current matched mode baselines, and controlled
performance diagnostics when triggered. Continue collecting timing and
peak-memory evidence during naturally required Beammap validation. Establish
polarimetry support or an explicit capability policy before release claims.

Exit gates are the broader structural definition of done in section F.2 of the
external review, with the project-owner performance proportionality exception:
strict scientific equivalence, zero unexpected errors, reproducible builds,
operational performance evidence with triggered controlled diagnostics, and
documented scientific conventions.

### Phase 4.1 - Four-Mode TolTECA Config Structure

Create a consistent numbered-YAML authoring kit for point, OOF, Beammap, and
science. Separate stable mode policy, site/runtime values,
observation/calibration selection, product choices, and user overrides. Keep
TolTECA as the merge owner and Citlali's generated low-level YAML as the
execution boundary.

Exit gates are defined in
[`PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md`](PHASE4_1_TOLTECA_CONFIG_STRUCTURE_PLAN_2026-07-16.md):
all four kits exist, overlay semantics are hermetically tested, accepted
low-level equivalence is explicit, and one TolTECA smoke run per mode passes.

The Citlali-owned kit and validation tranche is complete as of 2026-07-16.
`config/tolteca/` contains four hash-pinned five-file kits derived from the
accepted point, OOF, Beammap, and science snapshots. The hermetic merge tool
implements TolTECA/Tollan list semantics, reports effective authority and
override provenance, and participates in the full config preflight. That
preflight passes 107 tests and all four accepted policy hashes. It also exposed
and closed two existing science cleaner-grouping gaps in the resolved leaf
contract, which now covers 576 leaves. TolPROJ commit `a33d26a` vendors this
exact kit behind an opt-in `--refactor` setup path for pointing, automatically
selected OOF/science, and Beammap project setup. Its default commands retain
the established `70_reduce.yaml`/`72_reduce.yaml` behavior. The refactor path
hash-verifies every vendored file, rejects mixed numbered-config families,
generates `72_observation.yaml`, preserves operator-owned runtime and expert
overrides on same-kit reruns, and rejects in-place mode or kit changes. All 96
TolPROJ tests, Ruff, byte-compilation, and tracked-file audits pass. Phase 4.1
does not proceed to smoke reductions yet. Project-owner review found that the
V1 files still expose the full machine policy under generic names and do not
materially separate routine, advanced, and expert authoring. V1 remains a
mechanically exact reference and the TolPROJ path remains opt-in, but it is not
the accepted operator interface.

The V2 authoring structure, first reviewed through science, is now generalized
under `config/tolteca/v2/` for point, OOF, Beammap, and science. Every mode uses
seven mode-named files: generated internal policy, site runtime,
TolPROJ-generated observation binding, routine analysis defaults, product
choices, advanced overrides, and expert overrides. The ordinary surfaces are
bounded to 4 runtime leaves, 27-44 analysis leaves, and 5-30 product leaves.
Mode-inapplicable controls are excluded, fruit-loop controls are consolidated,
and source finding is visible but explicitly experimental and disabled.

All four unchanged V2 kits merge exactly to their accepted V1 policy hashes.
The preflight enforces classification, file-size, ownership-disjointness,
mode-scope, data-binding, and byte-for-byte regeneration gates; it passes 116
focused tests and every config-authority audit. Citlali commit `6b6be9f57` is
the canonical source. TolPROJ commit `8490f09` vendors that snapshot
byte-for-byte for all four modes and selects it only under `--refactor`; every
non-refactor command retains the legacy path. Its manifest-driven installer
generates mode-named observation files, preserves all five operator files on a
same-kit rerun, rejects mixed or in-place kit changes, and passes all 100
TolPROJ tests. A fresh Unity smoke reduction for each mode now completes Phase
4.1; no Citlali compilation is required for this YAML-only integration.

Project layout review found that TolPROJ science and OOF reductions live under
`<root>/<user>/<source>` while shared data live under `<root>/data`. Data input
and KIDs fit-report paths therefore belong to the generated observation/data
binding, not the reducer-edited runtime file. The canonical V2 generator places
those paths in the mode-specific generated observation file; TolPROJ supplies
`../../data` for nested science and OOF projects.

### Phase 4.2 - Technique And Performance Review

Review every active subsystem for scientific/numerical appropriateness and
real-workload efficiency. Produce evidence-labeled dispositions and a finite
backlog before broad remediation. Intentional science changes use successor
validation evidence rather than being forced to match OG.

Exit gates are defined in
[`PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md`](PHASE4_2_TECHNIQUE_PERFORMANCE_REVIEW_PLAN_2026-07-16.md):
all active components are covered, no unowned P0/P1 finding remains, dominant
runtime/memory contributors have evidence-backed dispositions, and accepted
changes receive proportionate tests and mode validation.

The comprehensive component census is complete as of 2026-07-16. The
[`technique and performance evaluation`](PHASE4_2_TECHNIQUE_PERFORMANCE_EVALUATION_2026-07-16.md)
and machine-readable
[`component review`](../validation/phase4_2_component_review.json) assign every
active component to one of 13 review units, reconcile the earlier correctness
and performance audits with current code, and record evidence-labeled
dispositions. The production RTC, PTC, naive/JINC mapmaking, fruit-loop,
point/OOF, Beammap, coadd, and Wiener techniques are retained. No wholesale
numerical rewrite is justified by the evidence.

The census found one P0 capability defect: experimental
`maximum_likelihood` mapmaking remained selectable even though it was not a
validated global noise-aware mapmaker and Beammap did not populate that method.
Typed validation now rejects it for production while preserving the research
implementation for an explicit future decision. It also found one P1 output
contract defect: required pointing and Beammap FITS metadata could silently
fall back to zero or omission. Those catches are removed so required write
failures propagate. Subsequent accepted point and Beammap reductions exercised
the supported output paths without unexpected errors, closing those
behavior-preserving production guards.

Current profiles identify three measured Beammap costs: PTC cleaning consumed
1,565.923 seconds, map population 1,250.498 seconds, and the PTC diagnostic
sidecar 344.554 seconds in accepted `redu06`. Science `redu28` spent 2,799.461
seconds in the aggregate TOD pipeline. These measurements provide bounded
targets if a future performance trigger occurs; the accepted run history does
not establish a sustained regression that justifies speculative changes to the
mature numerical or output paths.

Phase 4.2 closed on 2026-07-17. The production P0/P1 findings are owned and
repaired, later accepted point and Beammap runs exercise the supported paths,
and every active component has an evidence-backed disposition. Candidate
Beammap NetCDF lock narrowing and finer science-stage attribution remain
responses to a measured slowdown rather than mandatory optimizations. Dedicated
Beammap/noise-heavy peak-RSS and profiler-overhead campaigning is
trigger-deferred under retained-debt item D13. Source finding is explicitly
experimental and disabled in the accepted operator kits; enabling it requires
a scientifically owned injection/recovery matrix. The
[`evaluation completion addendum`](PHASE4_2_TECHNIQUE_PERFORMANCE_EVALUATION_2026-07-16.md#completion-addendum---2026-07-17)
records the final disposition.

Compilation-boundary and build-system work remains deferred pending the
TolTECA developer's current build design. Header changes still demonstrate the
cost of that debt: rebuilding the CLI translation unit and link took 60.02
seconds locally during this review.

### Phase 5 - Integration And Closeout

Consolidate canonical architecture and scientific-convention documentation,
the validation ledger, and the intended-science-change manifest. Mark or remove
legacy/stub paths, tag the forensic refactor branch, and integrate the exact
validated tree. Add install/export support only if external library consumption
is an accepted project goal.

Phase 4.2 may recommend bounded RTC/PTC or other algorithm work, but review does
not authorize a wholesale rewrite. R execution remains a follow-up until its
measured-channel prerequisites are explicitly approved.

## Stop And Defer Rules

- Stop splitting files when a split has no clear owner, contract, test seam, or
  dependency benefit.
- Do not broadly rewrite RTC/PTC, JINC, or Wiener-filter numerical kernels in
  this refactor.
- Do not make compact config authoritative before TolTECA overlay acceptance.
- Do not implement R execution before a measured-channel data contract exists.
- Do not add concurrent reductions as a requirement unless the project owner
  explicitly needs them; sequential same-process reentrancy is required.
- After the refactor, replace the flat fruit-loop `reduNN` iteration sequence
  with one atomically claimed run directory containing explicit nested
  iteration identities, for example `redu01/iterations/iter00` through
  `iterNN`. Treat `redu01` as the stable identity of one user-invoked reduction,
  not as an iteration number. Add a run manifest that records a stable execution
  ID, each child iteration ID, the selected final iteration, Citlali version and
  git revision, and the effective-config digest. Preserve TolTECA-facing final-
  product compatibility during migration. This is the preferred long-term
  replacement for coarse output-root exclusion, but it is not part of the
  bounded Phase 3 repair.
- Do not squash or rewrite the only validated branch history.

## Decisions Requiring Scientific Ownership

Ask the project owner when implementation first depends on an answer. Do not
silently choose among these:

- Which output products are required versus optional in each reduction mode.
- How disabled filters and extinction states appear in requested, effective,
  and realized provenance.
- The future scientific meaning of hardware-polarization controls and the
  contract required to make enabled polarimetry a supported capability.
- Allowed calibration or analysis fallbacks and their required diagnostics.
- Any future persistent measured-detector namespace and lifecycle. Canonical
  APT v1 already fixes its `uid` as artifact-local only and must not be read as
  a persistence claim.
- Future evolution of network/array mappings, coordinate-frame support, units,
  missing-value encodings, and table schemas outside already fixed versioned
  product contracts.
- OOF scientific intent and the acceptance tolerances for each mode.
- Whether any future caller needs concurrent reductions in one process.
- The measured-channel contract and missing-data policy for future R analysis.
- Whether Citlali must be installable and consumable as an external library.

## Durable Evidence

`validation/accepted_runs.json` is the machine-readable validation ledger. New
accepted checkpoints must record commit, binary version, mode, input/config
identity, comparator version, tolerances, error count, timing, available memory
evidence, and disposition. Run
`tools/baseline/validate_validation_ledger.py` after editing it.
`validation/intended_science_changes.json` is the separate source-to-evidence
ledger for intentional post-baseline scientific changes; validate commit
ancestry, patch identity, and evidence links with
`tools/baseline/validate_science_change_ledger.py`.
`doc/SCIENTIFIC_CONVENTIONS.md` is the canonical human reference for identity,
units, frames, indexing, validity, provenance states, and change-to-validation
routing. Product-specific executable requirements remain in
`validation/product_contracts.json`.
`doc/CANONICAL_APT_V1.md` is the normative human contract for the accepted but
unactivated canonical Beammap baseline APT schema, identities, raw relation,
field registry, ECSV encoding, and receipt-last publication.
`doc/ARCHITECTURE.md` is the canonical human reference for the active software
entry, component and dependency direction, lifecycle ownership, compatibility
boundaries, failure flow, source classification, and extension routing.
`doc/PHASE4_CLOSEOUT_CENSUS_2026-07-16.md` maps every adopted F.2 completion
criterion to evidence, an approved exception, a deliberate deferral, or a
finite remaining action.
`doc/RETAINED_DEBT.md` is the canonical owner/trigger/exit register for
deliberately retained limitations, and `doc/adr/README.md` indexes durable
architecture decisions.
`validation/validation_profiles.json` identifies the active immutable
validation epoch and one profile per supported reduction family; validate it
with `tools/baseline/validation_profiles.py --list`. Continue to update this
document and the dated handoff note at phase gates and material validation
checkpoints.
