# MAP-SPACE-HORIZONTAL-SCIENTIFIC-CONTRACT-AUDIT-001 Findings, Repairs, And Owner Decisions

Status: completed under owner disposition; shared-source repair remains required

Recommended disposition: `ACCEPT WITH BOUNDED CONTRACT REPAIR`

The original stop-and-escalate action was correct.  On 2026-09-03 the owner
resolved `MSP-OD-001` in favor of frozen SCI-MAP v0.1/r0.7.1 and SCI-JINC
v0.1/r0.3.  That decision establishes scientific/package coherence for this
audit; it does not erase the contradictory bytes in
`doc/SCIENTIFIC_CONVENTIONS.md`.  The four findings therefore remain MAJOR and
each is recorded as `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`.

This artifact specifies the required clause-level repair but performs none of
it.  No shared-conventions or frozen-package file is edited by this audit.

## Finding counts and states

| Severity | Count | Current state |
| --- | ---: | --- |
| CRITICAL | 0 | none |
| MAJOR | 4 | `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED` |
| MODERATE | 0 | none |
| MINOR | 2 | `OPEN / MANAGER-ONLY-CORRECTION-DEFERRED` |

Explicit typed unavailable states `MSP-U-001`--`MSP-U-011` in the conformance
matrix are contract states, not findings.  The owner disposition does not make
any numerical route available.

## Owner-resolved MAJOR findings

### MSP-F-001 — MAJOR — ordinary MAP physical-component identity

State: `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`

Classification: contradiction; product identity and signal meaning.

Affected product/routes: MSP-P004; MSP-E001, MSP-E009, MSP-E010, MSP-E012,
MSP-E013 and MSP-E023 insofar as they inherit MAP quantity identity.

Exact conflicting shared-conventions clauses:

- `doc/SCIENTIFIC_CONVENTIONS.md:330-332` formally labels the single current
  component as Stokes `I` and reserves `Q` and `U`.
- `doc/SCIENTIFIC_CONVENTIONS.md:413-416` calls ordinary naive observation and
  coadd maps Stokes-I.
- The shared-conventions file has SHA-256
  `affe9c5fa144fd2fe196b8cccaf4dc9bc9ec9970634ef7db9386ac9c5e2a1f53`.

Exact frozen authority supporting the disposition:

- `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-PTC_TO_SCI-MAP_BOUNDARY.md:45-52`,
  SHA-256
  `a499c59afb69eefda74a5b131ad37afd165213b99324c8b67a6de1d20793c9b7`,
  defines the ordinary signal as a nonpolarimetric total-intensity-equivalent
  detector-time quantity in the inherited top-of-atmosphere point-source-
  equivalent `mJy/beam` convention and says neither the unit nor a STOKES token
  establishes formal Stokes I.
- `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex:4`,
  SHA-256
  `68acf81d6c27788495cc680a1819da5f269616493b9aba5a6e0a1d1058ba5fa7`,
  requires that identity.
- `SCIENTIFIC_OWNER_FREEZE_R0.7.1.md:20-46`, SHA-256
  `91801005ba2f2bce6471a9f6f4ed0b79806c893f498b4f3cca9e81e26df39ce1`,
  freezes SCI-MAP v0.1/r0.7.1.

Owner disposition: preserve the frozen nonpolarimetric total-intensity-
equivalent MAP identity.  The generic Stokes-`I` language is superseded only
to the extent that it assigns formal Stokes-`I` physical identity to this MAP
product.

Necessary clause-level repair, not performed: revise lines 330-332 and
413-416 so that legacy/grouping index or label `I` does not establish formal
Stokes identity; state the frozen nonpolarimetric total-intensity-equivalent,
top-of-atmosphere point-source-equivalent `mJy/beam` identity and nominal-beam
lineage; leave any future formal Stokes-I product to separate authority.

### MSP-F-002 — MAJOR — MAP coadd coefficient and normalization

State: `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`

Classification: contradiction; normalization and statistical meaning.

Affected product/routes: MSP-P005; MSP-E004, MSP-E010 and MSP-E013.

Exact conflicting shared-conventions clauses:

- `doc/SCIENTIFIC_CONVENTIONS.md:446-454` defines `u` as realized `weight_I`
  after observation normalization and possible empirical rescaling.
- `doc/SCIENTIFIC_CONVENTIONS.md:698` assigns an inverse-square signal unit and
  conditional precision interpretation to the generic MAP gridding/
  normalization coefficient.

Exact frozen authority supporting the disposition:

- `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-MAP_COADD_PROFILES_R0.7.md:9-26`,
  SHA-256
  `d93c04488925931676b02dff433774ff2cda9846fdd1d3f34bff29d76efdd702`,
  fixes the domain to admitted observation-output row `(o,p)`, fixes
  `u_op=1` dimensionless, and defines equal-observation arithmetic averaging.
- `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex:41-44`,
  SHA-256 as in MSP-F-001, requires the same rule and separate covariance
  typing.

Owner disposition: preserve one dimensionless `u_op=1` coefficient for each
admitted MAP observation row.  This observation-level rule does not replace or
flatten authorized sample-, pixel-, numerator-, denominator-, validity-, or
coverage-level information.  It is not a JINC coaddition rule; base SCI-JINC
v0.1 authorizes no cross-observation coaddition.

Necessary clause-level repair, not performed: replace the `weight_I`/
empirical-rescaling definition in lines 446-454 and the applicable unit claim
at line 698 with the observation-row domain, dimensionless `u_op=1`, exact
admitted-row normalization, and prohibited precision, inverse-variance,
empirical-weight, exposure, and significance meanings.  Preserve all lower-
level information as separately typed, and state explicitly that the rule
does not extend to JINC.

### MSP-F-003 — MAJOR — MAP original-footprint exposure versus signal membership

State: `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`

Classification: contradiction; coordinates, support, exposure and lifecycle.

Affected product/routes: MSP-P003, MSP-P004 and MSP-P005; MSP-E003 and
MSP-E004.

Exact conflicting shared-conventions clauses:

- `doc/SCIENTIFIC_CONVENTIONS.md:470-475` makes retained exposure share signal
  membership and integer embedding.
- `doc/SCIENTIFIC_CONVENTIONS.md:491-492` defines upstream-eligible and retained
  exposure through eligibility, contribution, normalization and support
  decisions.
- `doc/SCIENTIFIC_CONVENTIONS.md:503-505` aliases coverage to retained exposure
  and policy support.
- `doc/SCIENTIFIC_CONVENTIONS.md:700` gives the generic exposure unit statement
  without the frozen unique-original coordinate identity.

Exact frozen authority supporting the disposition:

- `doc/scientific_contracts/packages/SCI-MAP/v0.1/src/common/requirements.tex:9,18,31-32`,
  SHA-256 as in MSP-F-001, requires unique-original occurrence accounting and
  separation from descendant/operator/support meanings.
- `doc/scientific_contracts/packages/SCI-MAP/v0.1/SCI-AST_TO_SCI-MAP_ORIGINAL_FOOTPRINT_COORDINATE_BOUNDARY.md:23-68`,
  SHA-256
  `f7be703d17320de6f6ecfc3a3974d54799540fff1d8b2d9491c941a7ea3b45a4`,
  binds each stable original acquisition occurrence to its own exact AST
  ALIGN-grid coordinate and target MAP WCS.

Owner disposition: MAP exposure is a geometric quantity based on unique
original AST-coordinate occurrences.  It is not defined by processed signal
membership, filtering footprint, interpolation, operator support, response
support, or statistical weight.  Base JINC has no inferred physical-exposure
or standalone-support product.

Necessary clause-level repair, not performed: revise lines 470-475, 491-492,
503-505 and 700 so MAP exposure places each deduplicated stable original at
that original's exact AST ALIGN-grid coordinate in the target WCS; separate
that placement from all descendant signal/contribution/filter/interpolation/
operator/response/statistical-support facts; remove any alias that changes
physical-exposure identity; and state that these clauses create no JINC
exposure or standalone-support role.

### MSP-F-004 — MAJOR — JINC bundle, support, response and uncertainty roles

State: `OWNER-RESOLVED / SHARED-SOURCE-REPAIR-REQUIRED`

Classification: contradiction; product roles, response and statistical claims.

Affected product/routes: MSP-P006; MSP-E005, MSP-E011, MSP-E020 and MSP-E024.

Exact conflicting shared-conventions clauses:

- `doc/SCIENTIFIC_CONVENTIONS.md:497-501` makes JINC F010 unavailable but then
  claims separately approved formal-support products under `SCI-MAP-002`.
- `doc/SCIENTIFIC_CONVENTIONS.md:510-515` promotes `C^2/Q` to a conditional
  formal mapmaker-weight role.
- `doc/SCIENTIFIC_CONVENTIONS.md:533-548` publishes formal support,
  coefficient-squared coverage, response kernel and empirical-downgrade
  products.
- `doc/SCIENTIFIC_CONVENTIONS.md:697-702`, only insofar as those generic unit
  and role rows are applied to JINC, implies signal/kernel, weight, noise,
  exposure, count, support, validity or standardized roles beyond the frozen
  five-role bundle.

Exact frozen authority supporting the disposition:

- `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/common/requirements.tex:193-231,290-309`,
  SHA-256
  `207a85acb31a4f381b289781706c9f14058d330ff847e99023e9e5714c4d4dff`,
  requires exactly the five numerical roles and forbids base response,
  variance, formal-weight, covariance, empirical-noise, significance,
  physical-exposure, standalone support/availability, diagnostic and
  generalized-provenance roles.
- `doc/scientific_contracts/packages/SCI-JINC/v0.1/src/common/assumptions.tex:101-107`,
  SHA-256
  `15b811ab6ace92aa2d1713ae19b92454cb865e8862b82a599f94eca1003a1765`,
  preserves the relevant conditional assumptions.
- `FREEZE_AUTHORITY_MANIFEST_R0.3.md:11-36,87-100`, SHA-256
  `ff4b79e7cca3950831eda95a16ec6a535597f543c4676378d2fc2f01d50faed2`,
  binds the six shared modules that control SCI-JINC v0.1/r0.3.

Owner disposition: preserve exactly five base numerical roles:
`jinc_signal_numerator`, `jinc_signed_normalization`,
`jinc_quadratic_accumulator`, derived `jinc_map` with local support/validity,
and `jinc_coefficient_squared_time`.  The compact generative record is
information state, not a sixth numerical role.  Additional weight, support,
response, covariance, exposure, diagnostic, or generalized-provenance roles
are not implicit, optional, or downstream-inferable base-v0.1 products.

Necessary clause-level repair, not performed: revise lines 497-501, 510-515,
533-548, and the application of generic rows 697-702 to JINC so the section
enumerates only the exact five-role atomic bundle; preserves conditional
mathematics as nonproduct mathematics; characterizes coefficient-squared time
only as method-specific temporal accounting; removes claims of base formal
weight, support, response, covariance, physical exposure, diagnostic,
empirical-downgrade or generalized-provenance products; and states that base
JINC authorizes no cross-observation coaddition.  Any route requiring an
excluded role must record `NOT_AUTHORIZED`, `UNAVAILABLE`, or
`NOT_APPLICABLE`, with exact authority evidence, rather than construct one.

## Minor documentary findings

### MSP-F-005 — MINOR — stale candidate/status wording inside frozen sets

State: `OPEN / MANAGER-ONLY-CORRECTION-DEFERRED`

MAP retains “freeze-only errata draft” wording in
`src/common/notation.tex:8` and `src/scientific-rationale.tex:53`, while
`SCIENTIFIC_OWNER_FREEZE_R0.7.1.md:20-46` explicitly freezes the exact set.
JINC boundary/profile files retain candidate/approval-required wording at
`SCI-PTC_TO_SCI-JINC_BOUNDARY.md:5-7`,
`SCI-AST_TO_SCI-JINC_BOUNDARY.md:5-6`, and
`SCI-JINC_UPSTREAM_ADMISSION_PROFILE.md:5-18,77-82`, while the controlling
manifest at `FREEZE_AUTHORITY_MANIFEST_R0.3.md:11-36,55-57,83-100,140-171`
supersedes those phrases.  Scientific meaning is unambiguous under the
manifests.  Frozen bytes remain untouched; any future correction needs its own
authority.

### MSP-F-006 — MINOR — manager records predate the accepted audit base

State: `OPEN / MANAGER-ONLY-CORRECTION-DEFERRED`

`doc/scientific_contracts/POINT_NOI_FLT_FIXED_INTEGRATION_CANDIDATE_2026-09-03.md:5,31-40`
and `doc/REFACTOR_STATUS.md:3429-3436` say the local line has not moved and the
horizontal audit has not begun.  At preflight, local
`refs/heads/codex/refactor-mainline` and the detached checkout both resolved to
commit `5f0fc20042b88fb6cd883c92d1b59b7f22832901`, tree
`97a4d908061e51418f93afc1d97d27433af441b8`.  These manager-only records do
not change scientific authority and are outside this work order.

## MSP-OD-001 — owner decision recorded

On 2026-09-03 the owner accepted the stop-and-escalate finding, retained all
four conflicts as MAJOR, and selected the frozen SCI-MAP v0.1/r0.7.1 and
SCI-JINC v0.1/r0.3 meanings.  The decision applies to MSP-P003--MSP-P006,
MSP-E001, MSP-E003, MSP-E004, MSP-E005, and every downstream route that
inherits those meanings.

The result has two deliberately separate axes:

- Scientific/package coherence: established under the recorded owner
  disposition and frozen authorities, subject to all existing conditional,
  unavailable, not-authorized and not-applicable route states.
- Repository-documentation coherence: not established because the exact
  clauses above remain in `doc/SCIENTIFIC_CONVENTIONS.md`; shared-source repair
  is required, so an unqualified `PASS` is prohibited.

## Recommended separate follow-on work order

Commission a narrowly scoped shared-conventions repair against an exact owner-
specified base.  Authorize edits only to `doc/SCIENTIFIC_CONVENTIONS.md` and
only to the affected spans identified above: 330-332, 413-416, 446-454, 470-475,
491-492, 497-501, 503-505, 510-515, 533-548, and 697-702 to the extent those
generic rows apply to the affected MAP/JINC meanings.

The follow-on should:

1. implement the four clause-level repairs exactly as specified here;
2. preserve all unrelated shared-conventions meanings and cross-references;
3. cite the exact MAP r0.7.1 and JINC r0.3 frozen authorities and record that
   the supersession is product-scoped;
4. verify the repaired clauses no longer assign formal Stokes-I identity,
   empirical/inverse-square MAP coadd meaning, descendant-based exposure, JINC
   coaddition, or extra JINC base roles; and
5. receive independent review before integration.

Exclude SCI-MAP, SCI-JINC, every other frozen package, application code,
validation products, canonical refs, FRUIT, ALIGN, the two manager-only minor
findings, dependency installation, and Unity activity unless separately
authorized.  The follow-on is a documentation-authority repair only and must
make no implementation, validation, performance, readiness, production, or
activation claim.

## Read-only completion checks

- Exact commit/tree/parent/detached state and local-ref equality were checked.
- All 71 admitted source paths and SHA-256 identities were checked.
- Package-local identity/completeness/internal-consistency verification and
  ECS-to-frozen-core representation fidelity were retained for all six
  packages.
- Every numerical path remains fail-closed through the explicit negative
  states and representative traces; unauthorized JINC roles were not created.
- FRUIT was limited to its excluded boundary envelope; ALIGN was not reopened.
- No additional consequential authority conflict appeared after applying
  `MSP-OD-001`.

No shared-conventions repair, integration, commit, push, rebase, cleanup, ref
movement, application change, frozen-package change, validation run, dependency
installation, or Unity action was performed.
