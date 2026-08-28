# SCI-JINC — Signed-Coefficient JINC Gridding And Response Scope Brief

Status: Stage A owner-review candidate; not owner-approved

Scientific owner: Grant Wilson

Version/date: `v0.1`, `2026-08-28`

Starting source identifier:
`codex/scientific-contract-library@731f821954d4321509765720c6ba1838c95eff3d`

Approved source identifier: unavailable until owner approval

## Program Adherence And Prior-Work Recovery

This package follows the
[Citlali Scientific Contract Library Program](../../../README.md), the
[pilot process review](../../../PILOT_PROCESS_REVIEW_2026-08-16.md), and the
[owner-approved downstream roadmap](../../../DOWNSTREAM_CONTRACT_ROADMAP_2026-08-26.md).
It starts after, and does not alter, the
[frozen SCI-MAP v0.1/r0.7.1 authority](../../SCI-MAP/v0.1/SCIENTIFIC_OWNER_FREEZE_R0.7.1.md).

- Prior-work record: [`PRIOR_WORK.md`](PRIOR_WORK.md)
- Recovery reviewed by: Codex manager on `2026-08-28`; scientific-owner review
  pending
- Existing material adopted: the accepted signed `N/C` estimator and distinct
  `N`, `C`, `Q`; the eight approved D003 support, subpixel, conditioning,
  admission, mask, coverage, kernel, and provenance decisions; the frozen
  ordinary-MAP exclusion; and the bounded destination-ownership invariant
- Existing material cited: the frozen implementation-independent
  `SCI-MAP-002_INDEPENDENT_CORE.tex`, content SHA-256
  `2c1f9ff95f65422a098846f747ed165d5aeddc5bedd854678bfa7faeebba4e24`
- Existing material abstracted: stable upstream quantity, identity, coordinate,
  unit, validity, lifecycle, response/covariance, and producer/transformer/
  consumer boundaries; no source or schema mechanics
- Existing material superseded: the core's radial-cutoff and pixel-area-
  integrated response branches, replaced by owner-approved square-cache and
  point-phase conventions
- Existing material deferred or excluded: implementation, source audit,
  findings, repairs, re-audits, tests, Unity, reductions, validation,
  achieved-performance, integration, production status, and the unidentified
  “memo” behind the March alignment note
- Genuinely new scientific work: reconcile the retained JINC authority with
  frozen upstream boundaries, close only the listed owner choices, and render
  one shared JINC authority in the library's two-view form without repeating
  the estimator derivation
- Proposed author references: this Scope Brief; the exact frozen independent
  core paired with [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md);
  and [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md)
- Author-packet exclusions: the complete recovery record, internal dossier,
  decision log, raw owner-decision files, adjacent packages, implementation,
  audits, repairs, tests, validation, reductions, Unity, current status, and
  every unlisted source

Confirm that this opening and the exact packet were reviewed before launching
scientific authorship: **no**. Stage B is blocked on explicit owner approval.

## 1. Package Name And Scientific Purpose

SCI-JINC defines a signed spatial-kernel estimator that turns an admitted,
coordinate-bearing detector timestream into a normalized observation map and
its method-specific response, support, conditional uncertainty, and lineage.

The central physical distinction is cancellation. A JINC footprint contains
positive lobes, analytic zeros, and negative lobes. The estimator therefore
cannot be described by the ordinary positive-coefficient language used for
SCI-MAP. It needs its own denominator, conditioning, response, support,
covariance, and validity meanings.

The package exists to make those meanings durable and usable by scientists and
engineers without making current Citlali behavior the scientific answer.

## 2. Scientific Boundary

The v0.1 operation begins after an upstream producer has supplied one exact
observation occurrence bundle containing:

- an admitted transformed detector sample and its quantity/unit identity;
- a declared JINC coefficient-weighting input or a typed unavailable state;
- the stable detector, array, sample, product-generation, and segment identity;
- the sample's AST-owned coordinate in the exact target JINC WCS;
- producer-owned retention, validity, response, covariance, and cause state;
  and
- the requested/effective/resolved state needed to select one JINC operator.

It ends after one atomic observation-level JINC result has either been
published with complete identity and required companions or declared
unavailable/failed with cause.

The in-scope transformer owns:

- the analytic JINC coefficient convention and array-specific parameter
  selection;
- square-cache footprint support and map-edge cropping;
- point-phase subpixel response;
- signed accumulation and `N/C` normalization;
- unit-invariant cancellation conditioning;
- conditional diagonal weight and off-diagonal covariance equations;
- the JINC-transformed, processing-filtered source-template response;
- coefficient-squared effective integration time;
- formal-support validity and any named JINC-local science policy;
- grouping, destination, and atomic publication identity; and
- requested/effective/resolved/realized JINC provenance.

Observation coaddition, map filtering, empirical noise/significance, source
fitting, Beammap inference, Pointing/OOF interpretation, and fruit-loop
feedback are outside this observation-estimator boundary unless the owner
explicitly adds a separately identified interface. In particular, no ordinary
SCI-MAP coadd rule is imported here.

## 3. Legitimate Inputs

For an admitted sample occurrence `i` and target pixel `p`, the contract may
use only explicitly bound inputs:

- signal `d_i` (or successor symbol), with finite payload, physical quantity
  role, unit, CAL/PTC ancestry, and immutable product generation;
- a finite positive producer-supplied coefficient `q_i` with exact family,
  unit, normalization, support, uncertainty meaning, and covariance
  assumptions, or a typed unavailable state;
- stable observation, detector occurrence/UID, array identity (`a1100`,
  `a1400`, or `a2000`), sample identity, PTC segment, and grouping identity;
- an AST-owned coordinate valid for the same occurrence and exact target JINC
  WCS; row, column, shape, time, or numerical equality alone cannot establish
  the join;
- producer-owned retention and cause-preserving validity/policy results;
- upstream response and covariance objects, each with exact domain, codomain,
  parents, support, approximations, unavailable terms, and lifecycle;
- finite strictly positive JINC shape parameters `a`, `b`, and `c`, `r_max`,
  pixel size, and array scale, plus integer effective `subpixel_n >= 1`;
- a declared processing-filtered source-template timestream when a response
  product is requested; and
- finite positive effective processed-timestream sample frequency `f_s,i`
  when coefficient-squared effective integration time is requested.

The input coefficient's statistical interpretation is not inferred from its
name or inverse-square unit. If the exact family does not establish
inverse-variance meaning and applicable covariance assumptions, the contract
must preserve the algebraic estimator while typing formal precision and
covariance claims unavailable or conditional as appropriate.

## 4. Required Outputs

For every requested and supported observation-level JINC map identity, the
contract must define an atomic result containing or explicitly typing
unavailable:

1. normalized signal `m_p = N_p/C_p` in the admitted signal unit;
2. the distinct normalization and quadratic accumulators or sufficient exact
   identities to reproduce and audit their roles;
3. conditional diagonal variance/weight and the associated off-diagonal
   covariance model, with assumptions and omitted terms;
4. the realized processing-filtered source-template response projected
   through the same JINC operator and normalized by `C_p`;
5. coefficient-squared effective integration time
   `T_c2,p = sum_i c_ip^2/f_s,i` in seconds;
6. an authoritative formal-support validity state distinct from temporal
   support, hits, finiteness, or empirical policy;
7. exact JINC map, array/group, WCS/frame, unit, operator, parameter, support,
   subpixel, conditioning, and parent identity;
8. explicit availability/cause for every required or optional companion;
9. requested/effective/resolved/realized provenance and immutable product
   joins; and
10. failure state that suppresses realized success when any required output or
    join fails.

Names and storage layouts are not selected by this Scope Brief. The contract
must state meanings independently of current FITS, YAML, C++, or container
representations.

## 5. Upstream And Downstream Responsibilities

### Upstream producers

- **SCI-ALIGN/SCI-AST** own occurrence/time identity, sample coordinates,
  frame/WCS realization, coordinate validity, and coordinate uncertainty.
- **SCI-RTC** owns raw-timestream conditioning and its response, causal
  influence, validity, and lineage.
- **SCI-CAL** owns calibrated quantity/unit meaning, calibration response and
  uncertainty, quality, and lineage.
- **SCI-PTC** owns the transformed signal, retention, cleaning realization,
  coefficient family and QC when supplied, response/covariance state, and
  application generation. SCI-JINC cannot infer a coefficient family or
  precision role left unavailable by PTC.
- **SCI-VAL** registers and evaluates exact versioned rules. It does not author
  JINC admission, formal support, response use, publication, or consumer
  policy.

An exact `SCI-PTC_TO_SCI-JINC` boundary and JINC-owned admission profile do not
yet exist. The ordinary `SCI-PTC_TO_SCI-MAP v0.1/r0.1` boundary is informative
predecessor evidence only; it is not silently renamed or inherited.

### SCI-JINC transformer

SCI-JINC owns only the declared signed transformation and its local
conditioning, response, covariance, support, validity, destination, product,
and provenance facts. It preserves producer meaning and causes without
reclassifying them.

### Downstream consumers

- **SCI-NOI** owns empirical noise realizations, empirical covariance/weight,
  and significance calibration.
- **SCI-FLT** owns deterministic or inference-bearing filter transfer,
  filtered response/covariance/support/validity, and immutable raw-JINC
  parentage.
- **SCI-BEAM/SRC/MODE** own Beammap, source-fit, Pointing, and OOF
  interpretation and their consumer-specific qualification.
- **SCI-FRUIT** owns recurrence, map-to-TOD feedback, learning, iteration,
  convergence, and restart identity.

No consumer may reconstruct the JINC response from defaults, relabel
coefficient-squared time as exposure or precision, promote formal-invalid
support, or turn a conditional formal weight into empirical significance.

## 6. Externally Imposed Conventions

- Array identity is the stable name `a1100`, `a1400`, or `a2000`, never a
  container or map position.
- In-memory indices are zero-based; persisted FITS/WCS pixel coordinates are
  one-based.
- FITS/WCS, when used, is the persisted coordinate authority; memory order
  does not define axis sign, handedness, orientation, or wrapping.
- Requested, effective, observation-resolved, and realized states are
  distinct, one-way lifecycle stages.
- Missing, unavailable, unsupported, invalid, and failed are explicit states,
  not undocumented numeric sentinels.
- A required product or publication failure propagates and prevents a
  realized-success record.
- The JINC analytic coefficient is dimensionless; signal/response, coefficient,
  covariance, support-time, and logical-state units must be declared at their
  own boundaries.
- Enabled polarimetry and measured-R execution remain outside the active
  authority.

## 7. Questions The Contract Must Answer

The retained core and owner decisions already answer the estimator,
signed-lobe, support-family, subpixel-family, conditioning principle, formal
support, coefficient-squared time, response role, and provenance-stage
questions. Stage B must reuse those answers.

Before author dispatch, the owner must classify each question below as an
owner-supplied answer, an explicit deferral with a typed unavailable
consequence, or a bounded question the Stage B author is authorized to analyze.
The author may not choose a task-changing branch silently:

1. **Upstream quantity and coefficient.** What exact PTC product quantity and
   JINC-facing coefficient family enter the estimator, and under what evidence
   may `q_i` be called inverse variance?
2. **Complete analytic identity.** What exact content-bound analytic JINC
   function, parameter ordering, array scale, envelope, zero-limit, and
   amplitude convention complete the generic core without relying on current
   source or the unrecovered memo?
3. **Coordinates and admission.** Which AST coordinate role and exact
   JINC-owned VAL profile govern sample admission? Does JINC require its own
   exposure product, or only `T_c2`?
4. **Observation coaddition.** Is v0.1 strictly observation-level, or must it
   define a separate signed-estimator-compatible coadd boundary? Ordinary MAP
   coaddition is not a default answer.
5. **Grouping and destination cardinality.** Which array/network/detector
   groupings are scientifically supported, and how is one unique destination
   product identity established for each worker/population route?
6. **Product availability.** Which companions are required for every JINC
   product, which are optional, and what exact cause vocabulary applies when
   inputs or consumers are unavailable?
7. **Response and covariance composition.** Which fixed-state, full-procedure,
   or re-resolved response families and which covariance approximations may be
   published, and how are omitted selection/nuisance/parameter terms typed?
8. **Numerical realization.** What summation policy, contributor-count error
   bound, phase tie/bin rule, convergence tolerance, and deterministic
   sequential/parallel acceptance policy are contract parameters rather than
   implementation defaults?
9. **Finite-map edge rule.** What happens when the rounded sample center is
   outside the map but square support would overlap, and how are truncated
   response and covariance recorded?

The contract must also explain limiting cases: constant input, a single
contributor, equal coefficients, analytic zero, negative lobe, exact and near
cancellation, rescaling of signal units, truncated edges, missing array
identity, non-finite parameters/coefficients, unavailable upstream covariance,
and ambiguous destination identity.

## 8. Non-Goals

SCI-JINC v0.1 does not:

- alter or reopen frozen SCI-MAP, PTC, RTC, CAL, ALIGN, AST, or VAL authority;
- import ordinary positive-coefficient admission, one-hot projection, F010
  availability, exposure aliases, coaddition, or validity by analogy;
- tune `r_max`, `a`, `b`, `c`, `subpixel_n`, thresholds, or defaults;
- audit, repair, optimize, or refactor Citlali or assess an implementation
  candidate;
- run tests, simulations, reductions, Unity work, transfer-function studies,
  or observational validation;
- define empirical noise, significance, filtering, source fitting, Beammap,
  Pointing/OOF, or fruit-loop feedback science;
- assert achieved response fidelity, photometric accuracy, covariance
  fidelity, numerical reproducibility, performance, readiness, or production
  suitability; or
- draft or freeze Stage B artifacts under this Stage A authorization.

## 9. Allowed References

The proposed implementation-blind author may open only the exact logical
items content-bound in
[`AUTHOR_PACKET_MANIFEST.md`](AUTHOR_PACKET_MANIFEST.md):

1. the owner-approved bytes of this Scope Brief;
2. the exact frozen independent core at
   `fe201b69be2764dc47dc0a1957bfc8e493f2905a:doc/audits/packages/SCI-MAP-002_INDEPENDENT_CORE.tex`,
   paired with
   [`AUTHOR_SUPERSESSION_COVER.md`](AUTHOR_SUPERSESSION_COVER.md); and
3. [`AUTHOR_CONVENTIONS_AND_OWNERSHIP.md`](AUTHOR_CONVENTIONS_AND_OWNERSHIP.md).

No web or external paper is proposed. If the owner later supplies the missing
underlying JINC memo or a replacement analytic authority, it requires exact
identity, classification, a renewed Scope Brief review, and a new manifest.

## 10. Owner Decisions And Remaining Ambiguities

The approved historical JINC decisions and current open questions are listed
with stable identity, authority, consequence, and affected artifacts in
[`SCIENTIFIC_OWNER_DECISION_LEDGER.md`](SCIENTIFIC_OWNER_DECISION_LEDGER.md).

For Stage A approval, the owner must decide whether to:

1. approve this observation-level boundary and its explicit exclusions;
2. approve the proposed use of the core plus supersession cover rather than a
   new derivation;
3. approve the sanitized upstream/downstream ownership extract;
4. confirm the abstract destination-ownership invariant while leaving all
   historical evidence outside authorship;
5. disposition any open question that would materially change the author's
   task; and
6. approve the exact content hashes in the author manifest.

Owner approval of Stage A will authorize only later implementation-blind
authorship. It will not approve the returned scientific substance or any
implementation/evidence claim.

## 11. Independence Statement

This brief defines a scientific problem and ownership boundary without
prescribing current Citlali source, storage schemas, defaults, or observed
products as the answer. The proposed author packet contains only this brief,
one implementation-independent predecessor core under an explicit
supersession cover, and one sanitized conventions/ownership extract.

The implementation, audit, repair, re-audit, test, validation, Unity,
integration, achieved-performance, readiness, and production materials used
for Stage A discovery are not author inputs. If the packet is insufficient,
the future author must return a precise scientific question rather than search
the repository or infer from model memory.
