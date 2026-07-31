# SCI-MAP-001 bounded repair and re-audit handoff — 2026-07-31

## Authority and disposition

The project owner accepted the `SCI-MAP-001` `amend` verdict on 2026-07-31
and supplied the F009 and F010 scientific-policy decisions recorded below.
This authorizes a bounded repair proposal; it does not authorize repair in the
audit worktree, numerical-algorithm broadening, production expansion, or a
precision/significance claim.

- Governing implementation assessed by the audit:
  `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Audit branch: `codex/audit-sci-map-001`.
- Decision-record parent:
  `681aeec99c4d1c4762ecb8c4dae63c8599638678`.
- Frozen independent core:
  `doc/audits/packages/SCI-MAP-001_INDEPENDENT_CORE.tex`, SHA-256
  `13dd5922bd492e381afcc3b015284216dde1ccc2199ece3d070ee577c7324381`.
- The final decision-record commit is the commit that contains this handoff
  and the corresponding amended final audit; the coordinator must record the
  exact returned commit and artifact digests.
- Contract status becomes `approved`. Implementation remains
  `nonconformant`, validation remains `in_progress`, production remains
  `existing_use_only`, verdict remains `amend`, and re-audit remains
  `required`.
- F009 and F010 are `addressed_pending_reaudit`, not `closed`. Their policy
  choices are resolved, but implementation, product, provenance, fixture, and
  fresh re-audit gates remain.
- `remediation_branch` and `remediation_commit` remain null until a separate
  repair stage actually exists.

The repairer must work in a fresh worktree and a separate branch, suggested as
`codex/repair-sci-map-001`, based on the coordinator-selected exact application
SHA. The repair must not be made on `codex/audit-sci-map-001`.

## Approved F009 contract

### Admission and registration

Every observation is admitted as one immutable bundle before any coadd-owned
state is mutated. Admission covers the complete ordered map bundle, not one
pixel or map slot at a time. The admitted tuple includes:

- grouping and ordered map-slot identity, including array/network/detector or
  group identity, Stokes identity, and frequency identity where applicable;
- signal unit and estimator identity;
- response/kernel contract identity and required-companion inventory;
- coordinate frame, projection, source epoch, orientation, pixel scale,
  reference world coordinate, and reference pixel;
- observation and coadd shapes; and
- the versioned validity, coefficient, support, and non-finite policies.

Strict identity must be formed from authoritative full-precision inputs. The
legacy `MapBuffer::WCS` stores several WCS values as `float`, so equality of
that narrowed adapter is not sufficient: distinct source centers must not be
allowed to alias after float conversion. A full-precision admitted identity
must be serialized losslessly and may feed the legacy WCS only through a
one-way adapter.

The only approved geometric transform is the current centered integer
common-grid embedding, denoted by `$J_o$`. For coadd and observation dimensions
`(R_c, C_c)` and `(R_o, C_o)`, require

```text
R_c >= R_o, C_c >= C_o,
(R_c - R_o) and (C_c - C_o) are even,
delta_row = (R_c - R_o) / 2,
delta_col = (C_c - C_o) / 2.
```

The observation block must be exactly in bounds. Its full-precision WCS must
describe the same world coordinate at `(row + delta_row, col + delta_col)` as
the observation WCS at `(row, col)`. Shape and the corresponding reference-
pixel offset are the only permitted WCS differences. A source-center, frame,
projection, unit, scale, orientation, response, map-order, or odd/negative
shape-offset mismatch rejects the entire observation before mutation.

General reprojection, interpolation, fractional shifts, implicit source
recentering, and a best-effort WCS match are not approved. Centered integer
embedding is a geometric placement only. The signal-centering operator is
explicitly `$L_o = I$`: no mean subtraction, null-mode removal, or source
recentering is implicit in coaddition.

### Estimator and finite policy

Retain the normalized weighted mean

```text
N_c[p] = sum over admitted numerical contributions o of u[o,p] * m[o,J_o^-1(p)]
Q_c[p] = sum over the same observations of u[o,p]
m_c[p] = N_c[p] / Q_c[p] when Q_c[p] is finite and positive
```

with the kernel/response tracer accumulated through the same admitted
observation set, numerical-contribution mask, embedding, and coefficient. The coefficient
`u` is a nonprecision gridding coefficient by default. It is not inverse
variance merely because the current plane is named `weight_I`.

At the governing implementation, the coadd coefficient is the observation
`weight_I` plane after map normalization and, when enabled, optional global
empirical rescaling. The bounded repair preserves that arithmetic stage and
records it exactly in provenance; it does not convert the coefficient into a
precision or approve the empirical scale.

An explicitly invalid pixel is skipped before its signal or companions are
evaluated; it must never be included through multiplication by zero. For a
declared numerical contribution, signal, `u`, and every required numerical
companion must be finite, and `u` must be positive for the accepted ordinary
estimator. An unexpected non-finite required value or nonpositive contributing
coefficient is a required failure before any coadd mutation. A bad later map
slot must not leave an earlier slot partially committed.

Coadd consumes the F010 normalization, science-policy, and authoritative
validity states; it must not derive a second version of any one from
`weight > 0`, `coverage_bool_I`, retained exposure, or finite signal. The
in-memory states and their persisted authorities must be byte-identical. A
normalization-retained but science-policy-rejected pixel may remain
numerically contributable to the normalized mean. It must remain explicitly
low-confidence, and the coadd's own science-policy support and authoritative
validity must be evaluated and persisted separately. An explicitly invalid
pixel never contributes.

Precision interpretation requires `SCI-PTC-001` proof that the applicable
coefficient is a marginal precision and that the required independence or
covariance conditions hold. Correlated GLS is deferred. No coadd inverse-
variance, uncertainty, standardized-signal, S/N, or significance claim is
authorized while covariance is unresolved.

## Approved F010 contract

### Eight distinct persisted facts

The owner fixed eight separate logical products. The FITS-style names below
are a bounded serialization proposal, not additional owner decisions; the
repair must freeze the final spellings, units, dtypes, applicability, and
absence rules in the product registry and realized provenance.

| Logical fact | Proposed canonical identity | Required meaning |
| --- | --- | --- |
| Geometric hits | `geometric_hits_I` | Count of finite, in-bounds sample/detector projections into the pixel before upstream eligibility and mapmaker contribution selection. |
| Contributing hits | `contributing_hits_I` | Count of terms actually admitted to the numerator/denominator by the estimator's named, versioned contribution predicate. For ordinary naive mapping and the accepted coadd, the coefficient is finite and positive. A JINC-specific signed-coefficient predicate is outside this repair and must not be inferred silently. |
| Coadd observation count | `coadd_observation_count_I` | Count of admitted observation maps that contribute to the coadd pixel; not sample hits, exposure, coefficient sum, or validity. |
| Upstream-eligible exposure | `upstream_eligible_exposure_I` | Detector-seconds projected from samples explicitly eligible under the upstream validity contract, before the mapmaker's contribution and normalization-retention decisions. |
| Retained exposure | `retained_exposure_I` | Detector-seconds retained after the accepted contribution and normalization-support decisions; coadd values use the same admitted observation membership and integer embedding as signal. |
| Normalization support | `normalization_support_I` | Boolean result of the separately versioned numerical-normalization rule. It authorizes division/population only, not science use. |
| Science-policy support | `science_policy_support_I` | Boolean result of the separately versioned science-support threshold/policy. It does not by itself establish finite values or admitted identity. |
| Authoritative science validity | `science_valid_I` | `normalization_support AND science_policy_support AND finite signal and declared required companions AND admitted product identity`. This is the only authoritative raw science-valid mask. |

The exact contribution predicate and required-companion set are part of the
realized contract. For each enabled product profile, provenance must enumerate
the required companion identities. A missing required companion, an undeclared
companion set, or a non-finite required companion makes `science_valid_I`
false or causes required publication failure according to the versioned
failure policy; it may never be silently promoted to valid.

`coverage_I` is retained only as a documented compatibility alias for
`retained_exposure_I`. It must be bitwise equal to that canonical plane, have
the same detector-seconds meaning, and must not be described as wall-clock
integration time, inverse variance, support, confidence, or validity.

`coverage_bool_I` is deprecated as a validity authority. If it remains for
compatibility, it must be explicitly described as a deprecated threshold
diagnostic or exact compatibility alias of `science_policy_support_I`, with
the relationship tested and serialized. It is never an authority for
`science_valid_I`, and no downstream operator may substitute it for that
mask.

### Separate threshold algorithms and realized provenance

Normalization support and science-policy support remain separate. The bounded
repair preserves the currently traced numerical roles rather than selecting a
new threshold:

- ordinary normalization uses the finite-positive coefficient plane and the
  current `coverage_cut / 10` threshold role;
- science-policy support uses the current full `coverage_cut` threshold role;
  and
- both predicates require an explicit finite-positive coefficient check and
  use `>=` at the realized threshold. IEEE `!(w < threshold)` is prohibited.

The current order-statistic rule selects strictly positive values, lets `N` be
their count, and uses zero-based ascending index
`floor((floor(0.75*N) + N) / 2)` before multiplying by the applicable cut.
The repair must give the order-statistic rule and each role-specific support
rule stable algorithm/version identifiers. Any change to that numerical rule,
quantile convention, threshold ratio, comparison, or coefficient-stage choice
is outside this bounded repair unless separately approved.

For every observation map and coadd map, realized provenance must preserve,
with lossless round-trip precision:

- both algorithm/version identifiers;
- the exact input coefficient product and lifecycle stage;
- the requested and realized cut values;
- both realized threshold values, including the zero/empty-input case;
- the positive-value count and zero-based selected order-statistic index;
- finite, positivity, and `>=` comparison conventions;
- counts for geometric, eligible, contributing, retained, each support state,
  and final validity;
- the declared required-companion set and admitted bundle identity; and
- the exact raw-parent/product digest needed by a downstream operator.

Header values alone do not replace the realized provenance record. Tests must
prove numeric or hexadecimal round-trip recovery of every double without
rounding to display precision.

### Downstream rule

A later operator receives raw `science_valid_I` and the exact raw-parent
identity as immutable inputs. It must preserve raw science validity separately
from its own numerical-computability, stencil/window-support, response,
covariance, and output-validity masks. Numerical population cannot promote a
raw-invalid pixel. This handoff requires that interface but does not choose or
repair any NOI, FLT, source, or fruit-loop estimator.

## Bounded repair work packages

The following is the maximum authorized repair surface. Exact touched files
must be recorded on the repair branch, and unrelated cleanup is excluded.

| Work package | Findings | Bounded outcome |
| --- | --- | --- |
| Parallel ordinary accumulation | F001 | Remove shared-pixel races while preserving the accepted sequential estimator and a declared deterministic/equivalence policy. |
| Raw selection and finite failure | F002, F005 | Exclude explicitly invalid contributions, fail on unexpected non-finite valid inputs, and produce finite-positive support predicates without changing upstream flag policy. |
| Atomic coadd admission | F003, F009 | Add full-precision identity/WCS/response/unit preflight and two-phase centered-integer admission/commit; preserve the normalized weighted mean with `L = I`. |
| Same map-operator boundary | F004 | Make signal, kernel, and map-boundary realization accumulation use the same admitted membership/support operator. Do not change realization generation or empirical-noise policy. |
| Weight/covariance labeling | F006 | Label coefficients as normalization/nonprecision gridding quantities by default; represent precision as conditional/unavailable and preserve the `SCI-PTC-001` restriction. |
| Product hierarchy and metadata | F007, F010 | Persist the eight distinct facts, compatibility aliases, deprecation, units, identities, and explicit absence rules. |
| One-way lifecycle/provenance | F008 | Persist requested, effective, observation-resolved, and realized admission, identity, coefficient, support, threshold, membership, and product-inventory facts at full precision. |
| Local truth suite | F011 | Add deterministic equation, finite-state, identity/WCS, product, provenance, alias, and sequential/OpenMP gates at the exact repair SHA. |
| External evidence | F012 | After local repair gates pass, issue and audit a new exact-repair-SHA `SCI-MAP-001-UNITY-001` request. Codex does not run Unity. |
| Upstream conditions | F013 | Keep CAL/AST/PTC/VAL conclusions explicitly conditioned; do not invent their contracts in the repair. |

Likely implementation touchpoints include the `MapBuffer` storage and
normalization implementation; ordinary naive accumulation and merge paths;
`observation_coadd_accumulation.h` and its setup/call boundaries; map/coadd
execution plans and provenance; FITS name, metadata, product-registry, and
writer helpers; reduction auditors; focused C++ and Python tests; and the
config boundary audit. The repairer must prefer a pure admission object owned
by the coadd plan or lifecycle rather than adding cross-cutting mutable state
to `Engine`.

The governing-source trace identifies the following specific seams:

- `observation_coadd_accumulation.h` currently truncates unchecked centered
  offsets and mutates full matrices by positional slot;
- `observation_setup_impl.h`, `todproc_map_count_impl.h`,
  `observation_output_config.h`, and `observation_exposure_time.h` currently
  mutate coadd WCS/grouping, membership, observation numbers, or exposure too
  early;
- `map.h`, `map_buffer_allocation.h`, `todproc_allocation_impl.h`,
  `naive_mm.h`, and `map.cpp::normalize_maps` own the missing support state and
  the pre-floor information currently overwritten;
- `map_image_output_helpers.h` currently recomputes a policy mask in the
  writer instead of consuming one typed realized authority;
- the FITS unit/description helpers, `validation/product_contracts.json`, and
  the diagnostics dashboard currently retain unconditional inverse-variance
  or integration-time wording that must become nonprecision/retained-exposure
  wording without changing arithmetic; and
- mapmaking/coadd provenance, reduction auditors, focused tests, and the coadd
  config-boundary audit must advance together and reject tampered identity,
  coefficient-stage, threshold, or product-inventory facts.

On the separate repair branch, the accepted successor must also receive a
durable ADR, the corresponding `doc/SCIENTIFIC_CONVENTIONS.md` and
`doc/REFACTOR_STATUS.md` updates, and a new validation-epoch/product-contract
record. Historical accepted snapshots and their provenance are retained, not
rewritten. None of those production-document changes belongs on the audit
branch.

The shared product representation must be capable of carrying the eight F010
facts for JINC without reopening JINC mathematics. If SCI-MAP-002 has not yet
approved a method-specific signed-contribution predicate, the JINC product
must declare the corresponding fact unavailable rather than silently changing
the ordinary `q > 0` definition.

All observation-level mutations must occur only in the commit phase after the
complete bundle passes admission. This includes numerical accumulators,
coadd WCS/grouping, membership and observation numbers, exposure/count state,
product inventory, and realized provenance.

## Prohibited scope

The repair must not add or alter:

- general reprojection, interpolation, fractional registration, implicit
  centering, a GLS solver, covariance regularization, or a new map estimator;
- RTC/PTC numerical algorithms or an inferred PTC independence model;
- jackknife/noise-realization construction, N versus N-1 policy, variance
  floors, empirical calibration, or filtered variance;
- convolve, Wiener/low-pass, source fitting, Pointing/OOF, Beammap inference,
  or fruit-loop behavior;
- JINC internal mathematics or a signed-contribution convention;
- accepted production profiles, configuration defaults, or production
  authorization; or
- repair, merge, push, or canonical-ledger edits on the audit branch.

## Required local repair gates

At an exact repair SHA, with no required-data skip and zero unexpected
error-level record, the repairer must return at least:

1. One-pixel, identity-projection, uniform/unequal-weight, masked,
   zero-support, boundary, and multiple-detector fixtures tied directly to the
   numbered audit equations.
2. Exact or pre-registered sequential/OpenMP agreement for every supported
   ordinary profile, plus ThreadSanitizer evidence for the repaired primitive.
3. A two- and three-observation unequal-coefficient coadd that independently
   reconstructs `N_c`, `Q_c`, normalized signal, kernel, retained exposure,
   contributing-observation count, and all validity states.
4. Odd/negative shape offsets and individual map-slot, unit, response,
   source-center, frame, projection, scale, orientation, and WCS mismatches
   that fail before mutation. A mismatch in the final map slot must leave all
   coadd bytes, membership, counts, exposure, and provenance unchanged.
5. A no-centering fixture proving constants and offsets are neither mean
   subtracted nor recentered, and a source-center mismatch proving geometric
   centered placement cannot repair an identity mismatch.
6. NaN and both infinities in valid signal, coefficient, kernel, and each
   required companion that fail atomically; the same payload behind an
   explicitly false validity input must be skipped without evaluation or
   poisoning.
7. Pixel-truth fixtures for all eight F010 products, including pixels that are
   geometrically hit but upstream-ineligible, eligible but noncontributing,
   contributing but normalization-rejected, retained but science-policy
   rejected, numerically populated but science-invalid, and fully valid.
   Threshold-boundary cases include `nextafter` below, exactly equal to, and
   `nextafter` above each realized threshold, plus zero, negative, NaN, both
   infinities, empty input, and a constant coefficient map.
8. FITS/product-registry round trips proving names, dtypes, units, WCS,
   estimator type, alias equality, deprecated `coverage_bool_I` metadata,
   required-companion inventory, and explicit grouping/profile absences.
9. Provenance tamper tests for algorithm version, coefficient stage, cut,
   full-precision thresholds, comparison convention, admitted identity,
   observation membership/offset, response identity, and raw-parent digest.
10. Static and runtime assertions that no GLS/reprojection/interpolation path,
    implicit centering, inverse-variance metadata, coadd uncertainty, or
    significance claim has been introduced.
11. Direct small-matrix `A`, mean, and `A C A^T` comparisons, with covariance
    conclusions remaining conditional on `SCI-PTC-001`.
12. The repository's focused CTests, baseline-tool tests, full config
    preflight, and all touched provenance/output validators.
13. A no-broadening control in which every contribution is authoritative-valid
    and compatible: preserve observation order, centered offsets, and the
    existing `Q += u`, `N += u*m`, `K += u*k` operation order, with
    bitwise-identical numeric coadd planes apart from metadata/provenance.
    Any numeric delta in mixed-validity fixtures must be confined to an exact
    inventory of pixels whose newly authoritative validity excludes a term.

## Fresh re-audit and external gate

A fresh re-auditor, in a fresh worktree and suggested branch
`codex/reaudit-sci-map-001`, must assess the exact repair SHA rather than the
working tree or audit branch. The re-audit must:

- verify every F001-F013 gate and preserve F009/F010 as historical finding
  IDs;
- recompute observation and coadd products independently from persisted
  admitted validity, centered embedding, and coefficient planes;
- verify all eight F010 facts and both threshold algorithms from lossless
  provenance;
- prove that raw science validity survives as a separate downstream input and
  is never replaced by operator-local numerical support;
- retain the no-precision/no-significance restriction until `SCI-PTC-001` and
  applicable covariance evidence close it;
- audit the complete same-repair-SHA Unity request returned by the external
  evidence owner; and
- issue an updated production disposition. No repair is accepted merely
  because output resembles a historical map.

Until that re-audit succeeds, the four-profile `existing_use_only` disposition
and every audit fail-closed consumer restriction remain in force.
