# SCI-RTC — Raw-Timestream Conditioning And Temporal Response Scope Brief

Status: owner-approved Stage A author input

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-17`

Approved source identifier: `RTC-SCOPE-D001--D016`, owner-approved
`2026-08-17` with owner modification to `RTC-SCOPE-D004`

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md) and the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager and scientific owner, `2026-08-17`
- Existing material approved for adoption: the independent RTC mathematical
  core; owner decisions D001--D004; the phase-zero downsampling amendment;
  and the owner-approved learned-sampling design
- Existing material abstracted or excluded: current shared conventions and
  adjacent package interfaces are sanitized; implementation, audits,
  handoffs, repairs, tests, validation, Unity, and production status are
  excluded
- Genuinely new work: reconcile product-role signal domains; consolidate the
  prior derivation under current package boundaries; define selected-policy,
  response, availability, atomic-output, and learned-mode obligations; and
  expose remaining numeric owner choices
- Approved author references: this brief; the exact independent core plus
  [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md); and
  [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
- Author-packet exclusions: all unlisted repositories and files, including
  implementation, audit, repair, test, validation, and production material

Confirm that this opening was reviewed before launching scientific authorship:
`yes — Grant Wilson, 2026-08-17`.

## 1. Package Name And Scientific Purpose

**SCI-RTC — Raw-Timestream Conditioning And Temporal Response** defines how an
admitted aligned detector stream is transformed into a conditioned detector
stream whose signal meaning, unit, temporal and detector-mixing response,
support, influence, validity inputs, uncertainty availability, and provenance
remain scientifically interpretable.

The package must account for selected calibration application where applicable,
despike detection and replacement, source-protection controls, finite-window
FIR/IIR/notch filtering, phase-zero sampling, fixed or learned sampling plans,
and the atomic output needed by PTC, VAL, MAP, BEAM, NOI, and response-tracer
consumers. It must not treat a finite conditioned array as sufficient by
itself.

## 2. Proposed Scientific Boundary

V0.1 begins with:

- an immutable observation and admitted detector/acquisition identity;
- an aligned primary `xs` detector stream with declared raw signal quantity,
  sign, reference/baseline, unit, ordered sample grid, support, and validity;
- exact conditional ALIGN/AST identity, time, scan, coordinate, frame, and
  synthesis/support state required by the selected operation;
- an imported CAL operator and calibration lineage only for a product role
  that authorizes calibration;
- a selected RTC policy with requested, effective, observation-resolved, and
  realized state; and
- declared input covariance, nuisance, and response information where
  available.

V0.1 ends with an atomic role-specific RTC bundle containing conditioned TOD,
ordered detector/sample/scan/support identity, complete realized response or
honest availability, cause-preserving influence and flags, validity inputs,
conditional uncertainty/covariance status, exact state/provenance, and required
diagnostics.

RTC owns the scientific consequences of the realized conditioning order. It
does not own the physical meaning of upstream timing, coordinates, source
calibration, atmosphere, detector beam, or eligibility policy. It does not own
PTC cleaning and weights, MAP estimation, map filtering, source inference, or
fruit-loop recurrence.

## 3. Legitimate Inputs

1. **Primary signal.** The ordinary v0.1 input channel is `xs`, with its
   physical observable, raw unit, sign, reference/baseline, finite domain, and
   product-role meaning explicitly supplied. No other channel inherits that
   meaning.
2. **Identity.** Observation, Tune, network/interface, detector occurrence,
   selected APT/binding, ordered column, native row, aligned sample slot,
   scan, science interval, context interval, and output slot are distinct
   identities.
3. **Aligned state.** A declared assigned time grid, cadence/rate, phase,
   mapping, gaps, synthesis/origin state, support, scan partition, and
   validity supplied conditionally by ALIGN. Physical detector event timing
   may remain unavailable.
4. **Coordinate state.** AST-supplied coordinate, frame, topology, detector
   binding, validity, and uncertainty only when a selected mask or response
   operation requires it.
5. **Calibration state.** An immutable SCI-CAL-selected operator, unit target,
   source/target atmosphere lineage, detector binding, factor state,
   uncertainty, and availability only for roles that authorize calibration.
6. **Selected conditioning policy.** Declared despike detector and donor rule,
   replacement mapping and any separately authorized donor-to-target transfer,
   mask policy, ordered FIR/IIR/notch stages, exact coefficients,
   normalization, precision, state/reset, boundary/edge and non-finite rules,
   and phase-zero sampling state.
7. **Sampling plan.** Fixed or learned request; allowed factors and realizable
   filter family; native cadence; and, for learned mode, admitted beam,
   telescope speed/support, scan, tolerance, fallback, and immutable plan
   identity.
8. **Statistical state.** Input mean/covariance and nuisance covariance when
   supplied, with correlations and limitations. Absence is explicit and does
   not authorize diagonal, white, stationary, or independent assumptions.
9. **Product role.** Pointing, OOF, Science, Beammap, diagnostic-only, or other
   approved role, with the exact signal domain and required consumers stated.

## 4. Required Outputs

The contract must require, without prescribing a current storage schema:

1. conditioned `xs` TOD in the product-role signal unit, shape, and order;
2. immutable observation, detector occurrence, input/output sample grid, scan,
   interval, support, stage, and parent identity;
3. the exact factorized/local realized response including all enabled
   response-changing stages and detector mixing, or a typed unavailable state;
4. cause-preserving acquisition, synthesis, replacement, mask, edge, filter,
   non-finite, and phase-zero influence/support state;
5. separate numerical computability, descriptive flags, operator masks,
   coordinate validity, response status, and consumer-eligibility inputs;
6. conditional moments/covariance or weight semantics only when supported,
   with selection, nuisance, response, and model uncertainty availability
   stated separately;
7. requested, effective, observation-resolved, learned/resolved where
   applicable, and realized policy/state with exact coefficients, phase,
   rate, state, edge, and output lineage;
8. distinct immutable outer, inner, full, mini, diagnostic, simulated, and
   processed stage identities whenever those roles are produced;
9. explicit completion and required-output failure state; and
10. scientifically named diagnostics with units, support, estimator, validity,
    and a statement of whether they are inert observations or affect a selected
    policy.

## 5. Upstream And Downstream Responsibilities

- **ALIGN** owns native-to-assigned sample mapping, grid/cadence, scan and gap
  meaning, origin/synthesis state, mapping response/covariance, and physical
  timing availability. RTC consumes these facts and propagates their influence.
- **AST** owns coordinate values, frame/topology, detector binding, coordinate
  validity, and astrometric uncertainty. RTC may use them for an approved mask
  or response operator but may not infer missing coordinates.
- **SCI-BEAM and SCI-CAL** own calibration-factor production and calibration
  meaning. RTC may apply an admitted CAL operator in a declared role and own
  its ordering/response consequences; it may not derive `flxscale`, source
  flux, atmosphere physics, beam meaning, or promotion.
- Frozen SCI-BEAM assigns legacy `responsivity` no canonical scientific role.
  For donor detector `q` and target detector `d`, RTC may instead use valid,
  compatible `flxscale` values as the raw-domain transfer. Under the declared
  multiplicative convention `z_i = flxscale_i x_i`, the replacement scale is
  `flxscale_q / flxscale_d`. Both values must be valid for the exact detector
  occurrences in the same calibration convention and domain, and the target
  value must be nonzero. Otherwise this transfer is unavailable unless a
  separate donor-to-target authority is admitted.
- **SCI-BEAM** separately requires its primary standardized detector input in
  raw `Delta f/f`. RTC must preserve that role and may not silently replace it
  with a calibrated map or timestream.
- **RTC** owns the selected conditioning transformation, phase-zero sampling,
  complete conditioned response, local support/influence, and atomic output
  bundle.
- **PTC** is optional after RTC and owns correlated-mode cleaning, fitted
  state, analysis coefficients, processed-sample meaning, and covariance
  status. PTC may not strengthen an unavailable RTC response or treat
  influenced samples as independent science.
- **VAL** owns reusable sample/detector eligibility and cause precedence. RTC
  supplies exact causal inputs; it does not collapse them into a universal
  eligibility bit.
- **MAP** owns sample-to-map estimation, map response/support/validity, and
  coaddition. It may not infer missing RTC response or input eligibility.
- **BEAM, NOI, MAP-003, FLT, SRC/MODE, and FRUIT** own their respective
  inference, empirical noise, response-tracer, map-filter, fit, and recurrence
  science. They must bind the exact RTC parent and preserve unavailable states.

## 6. Externally Imposed Conventions

- V0.1 is Stokes I on primary `xs`; enabled polarimetry and measured R-channel
  execution are excluded until separately approved.
- Primary detector timestreams are samples by detectors; detector indices are
  local positions, not stable external identities.
- Time is seconds on the admitted assigned grid. The supported approximately
  8-ms rate relationships and `0.5x`, `2x`, and `4x` family are conditional
  software-grid authority, not physical integration-event proof.
- Pointing/OOF/Beammap coordinate-dependent operations use a declared AltAz
  tangent plane; Science uses its declared equatorial J2000 TAN relation.
  RTC performs no implicit frame conversion.
- Beammap's primary standardized signal is raw `Delta f/f`. A CAL-authorized
  path may instead produce `mJy/beam`; units and response cannot be inherited
  across roles.
- Missing, disabled, automatic, invalid, rejected, failed, and unavailable are
  semantic states rather than undocumented numerical sentinels.
- Requested, effective, observation-resolved, learned/resolved, and realized
  state flow one way. Realized state never rewrites the request.
- Required output failure propagates. Optional provenance detail is
  scientifically inert.

## 7. Questions The Contract Must Answer

1. What is the factorized ordered RTC operator for each admitted product-role
   signal domain, and which upstream operators are imported rather than owned?
2. What exact equivalence is required if calibration and cross-detector
   replacement are ordered differently?
3. How are despike selection, donor reuse, masks, filter state, edge treatment,
   non-finite handling, and phase-zero sampling represented in the realized
   operator?
4. When does an output require a local detector-time Jacobian rather than a
   scalar LTI transfer function?
5. How are amplitude, phase, group delay, aliasing, response uncertainty, and
   incomplete response represented?
6. How do ALIGN synthesis and RTC replacement influence propagate into
   scientific ineligibility without erasing their numerical effects?
7. Which conditional moments and covariance can be supported, and how are
   selection, calibration, response, nuisance, and model uncertainty kept
   distinct or unavailable?
8. What are the exact phase-zero output identity, representative time,
   support, flag/influence aggregation, edge, and cardinality rules?
9. How do fixed and learned sampling share one contract, and which numerical
   decisions are required before a learned plan may be resolved or applied?
10. What atomic bundle and availability states are required by each downstream
    consumer?
11. Which diagnostics may only observe, which may advise, and which may enter
    a declared selected policy?
12. What limiting cases and deterministic predictions falsify normalization,
    response, state, influence, alias, identity, and reset errors?

## 8. Non-Goals

V0.1 does not:

- derive ALIGN timing, AST coordinates, BEAM/CAL calibration, PTC cleaning,
  VAL policy, MAP/FLT estimators, source fitting, or fruit-loop recurrence;
- authorize physical integration-event timing, absolute timing correction,
  or astrometric placement;
- redesign mature despike, FIR, IIR, notch, or other RTC numerics merely to
  make a cleaner implementation;
- select universal current configuration defaults or numerical thresholds
  from code;
- authorize per-array or per-scan applied cadence, heterogeneous downstream
  time grids, noise-aware sampling optimization, or continuous adaptation;
- activate measured R, polarimetry, or an unapproved signal channel;
- prescribe current classes, files, storage schemas, or product names;
- inspect implementation for conformity, repair code, run validation or
  reductions, access Unity, or change production status; or
- claim implementation correctness, response fidelity, observational
  performance, or production readiness.

## 9. Approved Allowed References

The implementation-blind author may receive only:

1. this Scope Brief;
2. [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md) together
   with exact independent core
   `3319d7424c732c1c9fc300c336e4d428e6f91068:doc/audits/packages/SCI-RTC-001_INDEPENDENT_CORE.tex`;
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md).

No raw owner-decision file, audit, handoff, repair, re-audit, test, validation,
configuration document, current source file, sibling repository, or external
reference is admitted unless the owner amends the exact packet.

## 10. Approved Owner Decisions And Remaining Ambiguities

The following scope decisions were approved together by Grant Wilson on
`2026-08-17`; `RTC-SCOPE-D004` includes the owner's approval modification.

| ID | Approved decision |
| --- | --- |
| `RTC-SCOPE-D001` | V0.1 owns the conditioning transformation from an admitted aligned primary stream through the atomic RTC bundle; ALIGN/AST/CAL meanings remain imported. |
| `RTC-SCOPE-D002` | V0.1 is Stokes-I primary `xs`; measured R and enabled polarimetry are excluded. |
| `RTC-SCOPE-D003` | Product-role signal domains remain distinct: frozen Beammap uses raw `Delta f/f`; a separately CAL-authorized role may use `mJy/beam`; neither silently substitutes for the other. |
| `RTC-SCOPE-D004` | Where calibration is admitted, CAL owns factor science while RTC owns exact ordered application. Calibration precedes cross-detector replacement unless an alternative proves full unit/factor/offset/response equivalence. For raw donor `q` and target `d`, valid compatible factors under `z_i = flxscale_i x_i` authorize donor scale `flxscale_q / flxscale_d`; both factors must be valid for the exact detector occurrences under the same convention/domain, and the target factor must be nonzero. Legacy Beammap `responsivity` is not required. |
| `RTC-SCOPE-D005` | Any output influenced by ALIGN synthesis or RTC replacement is scientifically ineligible; compact causes and support remain mandatory. |
| `RTC-SCOPE-D006` | Every response-changing RTC stage, detector mixing, state, edge, mask, and sampling phase appears in the realized response, or the required response is unavailable. |
| `RTC-SCOPE-D007` | Outer, inner, full, mini, diagnostic, simulated, and processed products have distinct immutable identities and parents; requested/effective/observation-resolved/realized state is one-way. |
| `RTC-SCOPE-D008` | Phase-zero point selection is the only authorized v0.1 downsampling operator; arithmetic-mean downsampling is excluded. |
| `RTC-SCOPE-D009` | Exact filter/multirate coefficients, normalization, precision, state, edge, phase, rate, support, and alias semantics are recorded once per coherent segment; v0.1 does not redesign mature DSP. |
| `RTC-SCOPE-D010` | V0.1 includes fixed sampling and the already-approved optional learned-sampling authority, but no learned plan may apply without its required numeric policy. |
| `RTC-SCOPE-D011` | Learned sampling uses maximum safe reduction, the smallest admitted beam, maximum valid in-scan speed, analytical beam-times-filter response, exact phase-zero alias accounting, and one common immutable observation plan. |
| `RTC-SCOPE-D012` | Noise-aware, per-array, per-scan, heterogeneous-cadence, and continuously adaptive execution are deferred successor scopes. |
| `RTC-SCOPE-D013` | Coordinate-dependent masks require admitted AST identity/frame/validity; invalid or unavailable coordinates are not outside-source values and fail the affected operation. |
| `RTC-SCOPE-D014` | Despike/donor, source-mask, FIR/IIR/notch, edge, and recovery details are selected versioned policies constrained by the contract; current defaults are not universal scientific constants. |
| `RTC-SCOPE-D015` | The atomic RTC bundle contains signal plus identity, complete-response status, support/influence, typed flags/causes, validity inputs, uncertainty availability, provenance, and diagnostics; TOD alone is insufficient. |
| `RTC-SCOPE-D016` | Algebraic contract, implementation conformity, representation fidelity, observational performance, and production readiness remain separate claims. |

The contract may retain open numeric and policy decisions in the companion
owner ledger when their unavailable consequences are exact. Owner approval of
this Scope Brief does not answer those later questions or approve any current
implementation.

## 11. Independence Statement

This brief defines the problem, scientific boundaries, already-approved prior
decisions, and genuinely new work without prescribing current Citlali behavior
as the answer. The eventual author receives only the exact owner-approved
packet named above. It does not receive the internal dossier, recovery record,
source, audit findings, repairs, tests, validation, Unity evidence, current
production state, or unlisted model-memory context.

If the packet is insufficient, the author must return a precise scientific
question. It may not inspect implementation or search for a convenient answer.
