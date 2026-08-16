# SCI-CAL — Detector Calibration, Atmospheric Extinction, And Signal Transfer Scope Brief

Status: `owner-approved`; scientific authorship authorized

Scientific owner: Grant Wilson

Version/date: `0.1`, approved `2026-08-16`

Approved source identifier: recorded in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md)

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager and Grant Wilson, `2026-08-16`
- Existing material adopted: CAL-D001--D004 as amended; BAND-001 passband
  identity and limitations; CAL-ATM-D006 quality-class distinction
- Existing material cited: frozen implementation-independent CAL core,
  content SHA-256
  `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`
- Existing material abstracted: the later owner's structural atmosphere,
  factor-lineage, variance/weight, reconstructibility, and response-basis
  decisions from mixed repair-disposition records
- Existing material superseded: the original universal-UID/row-is-never-
  identity clauses; automatic authority of legacy q-model anchors; general
  target-unit scope beyond initial top-of-atmosphere `mJy/beam`
- Existing material deferred or excluded: all implementation, audit, repair,
  re-audit, test, numerical-execution, Unity, conformity, and production-state
  evidence
- Genuinely new scientific work: reconcile the retained structural atmosphere
  operator with its authority limits; finish the once-only factor/lineage
  model at the selected-APT boundary; define minimum validity, uncertainty,
  response-basis, and provenance products; derive falsifiable predictions
- Owner-approved author references: this brief; the
  frozen independent core with a supersession cover sheet; the exact v1
  passband authority manifest; one sanitized CAL convention/ownership extract
- Author-packet exclusions: [`INTERNAL_DOSSIER.md`](INTERNAL_DOSSIER.md), all
  application source, current interfaces, audits, repairs, tests, validation,
  active ALIGN B3c, and historical conformity claims

Confirm that this opening was reviewed before launching scientific authorship:
`yes — Grant approved the brief and all five scope decisions on 2026-08-16`.

## 1. Package Name And Scientific Purpose

**SCI-CAL — Detector Calibration, Atmospheric Extinction, and Signal
Transfer** defines how an eligible detector sample and its conditional
statistical uncertainty acquire a scientifically declared top-of-atmosphere
`mJy/beam` point-source calibration.

The contract must make every physical and empirical factor explicit, preserve
its identity and validity, state the reference plane and response meaning,
separate conditional noise from calibration/response systematics, and make
the result reconstructible without treating present software behavior as the
answer.

## 2. Scientific Boundary

The operation begins after upstream packages have supplied:

- an admitted observation-local detector sample and conditional validity;
- its time/elevation/eligibility state under approved ALIGN/AST authority;
- the exact selected immutable APT and the admitted association from the
  target acquisition record to the measured Beammap source record;
- top-of-atmosphere calibration quantities and their lineage;
- an approved atmospheric input state and atmosphere-operator identity; and
- the requested calibration mode and declared target unit.

It ends with a calibrated detector sample in top-of-atmosphere `mJy/beam`, its
transformed conditional variance or inverse-variance weight where one exists,
an explicit calibration validity/quality state, and resolvable factor,
uncertainty, selected-APT, reference-plane, and response-basis provenance.

SCI-CAL owns the algebra, admissibility, causal factor composition, uncertainty
transfer, scientific labels, and minimum provenance of that transformation.
It does not own upstream source selection, APT matching, detector/beam
inference, pointing/astrometric estimation, mapmaking, filtering, or empirical
response validation.

## 3. Legitimate Inputs

The author must define the exact identity, unit, support, and validity needed
for each of these scientifically legitimate inputs:

1. **Signal state:** an admitted detector sample (or declared sample vector),
   its measured-channel identity, pre-CAL unit, any already-authorized
   offset/baseline convention, time, support, and conditional validity. CAL
   may consume the state but may not choose the upstream conditioning
   estimator. V0.1 is restricted to the ordinary `xs` detector stream.
2. **Acquisition identity:** observation/Tune identity, network/interface, and
   network-local tone or row slot. A global column is only a locator. A proven
   ordered-row contract or an explicit keyed mapping may bind the acquisition.
3. **Selected measured-APT state:** exact immutable artifact identity/digest;
   target acquisition key; selected source Beammap key; association
   method/version and match/abstention/quality state; per-detector calibration
   quantities, units, validity, and parent lineage.
4. **Calibration factors:** absolute `flxscale`; any already-embodied
   TolProj pointing-derived correction and its lineage; relative
   `responsivity` only where its distinct non-absolute role is scientifically
   relevant; Beammap `sens` only in its separately declared sensitivity or
   approximate-weight role; and no ambiguous total `fcf` authority.
5. **Atmosphere state:** zenith `tau225`; eligible sample elevation/airmass;
   time support when time-resolved; reference plane `X_ref=0`; exact
   atmosphere-operator identity, model/passband provenance, and declared
   support.
6. **Conditional uncertainty:** variance or inverse-variance weight, its
   signal unit, support, estimator identity, and missing/unavailable state.
7. **Calibration-systematic state:** named nuisance values or distributions,
   uncertainties, provenance, validity, and detector/array/observation/cohort/
   global correlation scope where known.
8. **Response basis:** originating Beammap APT beam/template identity,
   elliptical parameters when available, and the separately supplied realized
   mapmaker/kernel/filtering state needed to interpret the `mJy/beam` label.
9. **Instrument convention:** the selected content-bound TolTECA v1 modeled
   array passband set where the adopted atmosphere definition actually
   requires it, including its recorded unknowns and limits.

Unknown or unavailable information remains an explicit state; it is not
silently replaced with a zero, unity factor, guessed identity, or default
scientific convention.

## 4. Required Outputs

The contract must define, without prescribing a current file layout:

1. the calibrated signal and exact once-only composed multiplier, with each
   constituent factor's definition, role, unit, recipient/support, reference
   plane, and application stage;
2. top-of-atmosphere `mJy/beam` point-source peak meaning and the response
   conditions under which that label remains valid;
3. transformed conditional variance and/or inverse-variance weight on exactly
   the same valid support as the calibrated signal;
4. a named calibration validity state and one coherent calibration-quality
   classification for an observation or explicitly declared segment;
5. named calibration and response nuisance terms and correlation scopes where
   available, with total calibrated uncertainty explicitly unavailable when
   they are incomplete;
6. one resolvable canonical calibration-lineage record per coherent reduction
   package, including raw observation, selected APT and digest, target/source
   association, factor definitions, atmosphere/passband/operator identities,
   reference plane, target unit, quality/validity, and response basis; and
7. compact product links sufficient to resolve that canonical record, rather
   than unnecessary duplication of the entire APT or per-detector table in
   every product.

The contract must distinguish a scientifically invalid transformation, an
intentionally disabled or uncalibrated mode, unavailable uncertainty, an
engineering-only correction, and a science-qualified calibration.

## 5. Upstream And Downstream Responsibilities

- **TolProj** owns project/cohort APT seed selection, calibrator
  interpretation, use of matching, pointing-derived science APT flux
  correction, library curation, and binding the selected artifact to an
  observation. SCI-CAL consumes the declared result and must prevent omission
  or double application; it does not redefine TolProj's estimator.
- **TolAPT** owns design-to-measured matching and new immutable
  provenance-bearing outputs from immutable inputs. SCI-CAL does not require
  a perfect design match for ordinary measured Beammap calibration fields.
- **TolTECA's operational line** owns selected input and configuration
  delivery. SCI-CAL defines the scientific meaning and admission of delivered
  state, not delivery-time defaults or compatibility conversions.
- **BEAM/Beammap** owns source/beam inference, fitted calibration quantities,
  beam/template uncertainty, and empirical response characterization.
  SCI-CAL preserves their identities and uses only admitted products.
- **ALIGN/AST** owns sample time, eligibility, elevation/airmass input meaning,
  and pointing/astrometric semantics. This package must not choose or rederive
  them and must not inspect active ALIGN B3c work.
- **MAP/FLT** owns mapmaking, kernel/filter response, coaddition, and empirical
  downstream response fidelity. SCI-CAL states the response-basis provenance
  required to interpret calibration but cannot certify that response.
- **Downstream scientific consumers** may construct full covariance from the
  nuisance model when required. They may not call a conditional weight total
  uncertainty or statistical significance when required systematic terms are
  unavailable.

## 6. Externally Imposed Conventions

The following prior owner decisions are binding unless Grant explicitly
reopens them during scope review:

1. `tau225` is zenith optical depth; extinction uses full eligible-sample
   airmass with top-of-atmosphere pivot `X_ref=0`.
2. Transmission and correction factors remain finite and positive. The zero-
   opacity limit is unity; positive opacity must not occupy a finite unity
   plateau; interpolation is continuous in line-of-sight optical depth.
3. Initial SCI-CAL supports only top-of-atmosphere `mJy/beam` with
   point-source peak normalization. `MJy/sr`, `Jy/pixel`, Rayleigh--Jeans or
   thermodynamic temperature, extended-source calibration, and integrated
   photometry are separate future contracts.
4. Per-detector Beammap beam/template identity remains distinct from realized
   map/filter response. Elliptical parameters are retained when available;
   circularization is a labeled approximation.
5. Conditional measurement uncertainty is separate from calibration and
   response systematics. Required nuisance categories include detector
   `flxscale`, common absolute calibrator scale, any TolProj pointing-derived
   correction, WVR/atmosphere model, Beammap `sens` where used for approximate
   weights, and beam/template response. Missing uncertainty is unavailable,
   never zero.
6. Acquisition, measured-APT, cross-observation association, and design
   identities are distinct. Proven row order is an admissible binding, not a
   universal physical identity. Artifact-local UID, occurrence identity,
   semantic content identity, and byte transport identity are distinct;
   cross-artifact matching names occurrence-scoped endpoints.
7. Calibration correctness, atmosphere-representation fidelity, relative
   repeatability, and absolute flux performance are different claims. Prior
   provisional goals are no more than one-percent representation error over
   declared support, about five-percent relative repeatability, and about
   five-to-ten-percent absolute accuracy per TolTEC band; none is an
   unevidenced guarantee.
8. The exact selected passband reference is
   `toltec-passband-set-v1:sha256:5e6f38f14bcae93a29ffe8362c52b15209f51aee4e48373b23aaa5ec2f8a6433`.
   It is an array-named modeled response set. Detector/network aggregation,
   telescope-measured uncertainty/covariance, normalization, and the physical
   photon-versus-energy convention remain unestablished.
9. Opacity quality classes distinguish the science-qualification target
   (`0 <= tau225 <= 0.15`), engineering-availability target
   (`0.15 < tau225 <= 0.25`), and outside-supported calibration. This is a
   classification policy, not by itself an adopted operator or operational
   domain; a coherent observation/segment receives one class rather than a
   sample-by-sample class switch.

## 7. Questions The Contract Must Answer

The scientific author must answer the following without consulting Citlali
implementation:

### A. Estimator And Factor Causality

- What are the canonical signal equation and the exact order-independent
  factor decomposition for the initial `mJy/beam` transformation?
- Which factor(s) are already embodied in a selected TolProj-corrected APT,
  and what lineage is sufficient to prove once-only application?
- Where does relative responsivity affect CAL-related operations without being
  mislabeled as absolute calibration?

### B. Atmosphere Operator, Support, And Claims

- Which parts of the retained
  `am12_fixed_djf25_piecewise_linear_los_tau_v1` result are adopted as the
  v0.1 structural operator, and which physical/observational claims remain
  unavailable?
- What exact opacity/elevation/time support and interpolation/bracketing rules
  are scientifically authorized for a science-qualified product?
- What, if anything, is defined for the engineering-availability class up to
  `tau225=0.25` when no validated continuous extension is available?
- How do passband unknowns limit the meaning or uncertainty of the operator?

### C. Identity, Validity, Response, And Uncertainty

- What admission predicate proves the target acquisition-to-selected-source
  association without claiming perfect design identity?
- What contract rule prevents the v0.1 `xs` calibration factor from being
  relabeled for a physically distinct measured stream?
- What is the minimum calibration validity/quality state, and how do causes
  such as missing identity, unsupported unit, out-of-domain atmosphere,
  disabled calibration, and unavailable uncertainty differ scientifically?
- What response condition preserves point-source peak `mJy/beam`, and which
  response claims must remain conditioned on BEAM/MAP/FLT?
- Which nuisance parameters and correlations are required, optional, or
  explicitly unavailable, and when does linear propagation fail?

### D. Limiting Cases And Falsifiable Predictions

The contract must derive predictions for at least:

- zero opacity, increasing opacity, zenith/airmass behavior, model nodes and
  seams, invalid domains, and a bracketed versus unbracketed time series;
- unity/omitted/duplicated/inverted factors and a corrected-APT double-
  application challenge;
- row permutation, network reorder, missing/duplicate acquisition keys,
  matched/abstained/ambiguous association, and irrelevant design-ID changes;
- scalar and vector conditional covariance transfer, unavailable nuisance
  terms, and the fact that common systematics do not average down with sample
  count;
- an unresolved point source under the declared beam/response and after a
  response-changing kernel; and
- unsupported target units, engineering-only opacity, and outside-supported
  calibration without silent relabeling.

## 8. Non-Goals

SCI-CAL v0.1 does not:

- audit Citlali, determine implementation conformity, prescribe source paths,
  repair software, optimize an algorithm, run tests or reductions, request
  Unity work, or change production status;
- select calibrator catalogs, calculate TolProj pointing recovery, perform APT
  matching, infer a beam, estimate pointing, make maps, design filters, or
  validate empirical response;
- authorize units other than top-of-atmosphere point-source-peak `mJy/beam`;
- claim that modeled passbands are measured detector/network/telescope
  response or invent their unavailable uncertainty;
- require perfect design identity for measured Beammap calibration fields;
- define total calibrated uncertainty when required nuisance terms are
  missing; or
- absorb or interfere with active ALIGN B3c work.

## 9. Allowed References

The owner-approved author packet contains only:

1. this Scope Brief after owner approval and content hashing;
2. `SCI-CAL-001_INDEPENDENT_CORE.tex` from
   `codex/audit-sci-cal-001@27b0916e725696597c3ba84fb6a82bf6cf0ea356`,
   retrieved content SHA-256
   `106755520b048f601bc60fd04e7b6020e6fa470480ac3105fa7ba269c730a4fe`,
   with an owner-approved supersession cover sheet;
3. `SCI-CAL-001_PASSBAND_AUTHORITY_001.json` from
   `codex/register-sci-map-003-audit-disposition@8c581bfb26f01b187f4f1e0565f4457bcc25f099`,
   as the exact instrument-reference manifest only; and
4. one content-hashed, owner-approved CAL convention/ownership extract
   containing only the relevant identity, unit, immutable-artifact, and
   responsibility boundaries from the scientific conventions, TolProj/TolAPT
   documentation, and the identity principles of accepted-but-unactivated APT
   ADRs 0010 and 0011.

The supersession cover sheet must say that the later layered identity
contract replaces the independent core's stronger row/UID rule, that v0.1 is
restricted to top-of-atmosphere `mJy/beam`, and that no associated audit,
repair, validation, or implementation material is admitted.

No other reference may be opened without Grant's approval and an updated,
re-frozen Scope Brief or explicit owner answer. Exact packet identities and
hashes are recorded in [`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md).

## 10. Owner Decisions And Remaining Ambiguities

### Prior decisions confirmed

Grant confirmed the nine conventions in section 6, the layered identity
amendment, the once-only factor principle, conditional variance/weight
scaling, package-level reconstructibility, and the separation of structural
atmosphere correctness from physical and observational claims.

### Five scope decisions approved on 2026-08-16

1. **CAL-SCOPE-D001 — included endpoint.** SCI-CAL ends at
   calibrated detector samples plus their conditional uncertainty, quality,
   and canonical lineage; downstream maps are consumers, not CAL estimators.
2. **CAL-SCOPE-D002 — atmosphere authority.** The retained fixed
   line-of-sight-optical-depth operator enters authorship as structurally
   adopted but physically/observationally qualified, with exact operational
   support still to be stated by the scientific contract.
3. **CAL-SCOPE-D003 — engineering-only behavior.** No calibrated SCI-CAL
   output is authorized for `0.15 < tau225 <= 0.25` until a continuous
   engineering operator is separately adopted. The contract must represent
   the outcome truthfully as unavailable/uncalibrated rather than silently
   extrapolating or relabeling it.
4. **CAL-SCOPE-D004 — reference packet.** The four-item author packet in
   section 9 is approved, including preparation of the sanitized cover and
   convention/ownership extract.
5. **CAL-SCOPE-D005 — measured-channel scope.** V0.1 calibrates only `xs`.
   No other measured detector stream inherits the same scientific meaning.

These decisions approve scientific scope, not an implementation choice.

## 11. Independence Statement

This brief defines the scientific problem, prior decisions, boundaries,
questions, and expected predictions without prescribing the current Citlali
implementation as the answer. It was sanitized from the separate internal
dossier. The proposed scientific author will receive only the owner-approved
brief, the exact references in section 9, and later owner answers.

The author will not receive Citlali source, current interfaces, audits,
findings, repairs, tests, numerical execution evidence, Unity material,
validation state, or active ALIGN B3c context. Grant's 2026-08-16 approval
authorizes the bounded author packet and implementation-blind author dispatch;
it does not itself approve the resulting scientific contract substance or any
implementation.
