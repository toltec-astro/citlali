# SCI-MAP v0.1 Post-Freeze Scientific-Owner Decisions

Date: `2026-08-28`

Scientific owner: Grant Wilson

Status: approved scientific-owner input to a bounded SCI-MAP/PTC successor;
not incorporated into the frozen SCI-MAP v0.1/r0.7.1 or SCI-PTC v0.1/r0.5
authorities

## Purpose And Change Control

This record preserves the decisions reached after the SCI-MAP v0.1/r0.7.1
freeze while walking the eight remaining MAP-local questions and the external
`PTC-OD-010` coefficient gate one by one. It prevents those decisions from
being reconstructed from conversation or silently inferred from an
implementation.

The frozen r0.7.1 MAP packet and frozen r0.5 PTC packet remain unchanged. Their
ledgers therefore continue to report the states that were true at their
respective freezes. This record does not retroactively change a frozen product,
source manifest, requirement, prediction, PDF, VAL evaluation, or scientific
claim. Incorporation requires an explicitly versioned successor, new exact
source binding, and renewed cross-document and horizontal review.

No decision below asserts implementation conformity, validation, achieved
response, achieved covariance, observational performance, readiness, or
production authorization.

## MAP-Local Decisions

### `SCI-MAP-OD-001` — Purpose Of The Support Rule

**Resolved.** The adopted threshold construction is an operational
relative-support policy. The permissive numerical-normalization threshold
`Q_star coverage_cut / 10` and the default science-support threshold
`Q_star coverage_cut` serve distinct roles. `Q_star` is a reference drawn from
the upper well-supported part of the exact declared policy population. The
rule manages relative coverage and edge support; it is not, by itself, an
exposure, precision, significance, response, completeness, or optimality
statement. Every realization records the exact population and policy identity.

### `SCI-MAP-OD-002` — Change And Extension Authority

**Resolved.** Anyone may inspect, modify, run, distribute, study, or propose an
improved Citlali algorithm. Independent development is encouraged. The
scientific contract governs product identity and claims, not permission to
experiment. A changed support algorithm is not conforming SCI-MAP v0.1 unless
it is mathematically equivalent under the contract. Official adoption uses a
versioned successor and evidence appropriate to the claims being made. No
successor or independently developed result retroactively relabels an earlier
MAP product.

### `SCI-MAP-OD-003` — Response Disclosure

**Resolved.** Every MAP product reports its MAP-local realized operator
response; that operator is known from the exact realized MAP state. An
upstream-conditioned effective response is reported as available, limited, or
unavailable according to the bound upstream response information. A complete
whole-chain response remains unavailable unless separately established by an
authorized complete study. Response meaning, domain, assumptions, omissions,
and limitations are explicit. Incomplete response knowledge does not invalidate
the numerical map or prohibit later scientific analysis. Later response
estimates or response-corrected maps are new versioned products bound to the
original MAP product and processing identity.

### `SCI-MAP-OD-004` — Uncertainty And Covariance Persistence

**Resolved.** MAP reports what uncertainty or covariance information it
actually provides, together with its meaning, domain, assumptions,
limitations, omissions, and exact lineage. A complete covariance matrix or
operator is not mandatory for the base map. Absence of a complete covariance
model does not invalidate the map or prohibit later scientific analysis.
Normalization, hits, exposure, and diagnostic weights are not relabeled as
precision. Later covariance estimates may be attached as immutable, versioned
products without altering the original MAP product's claims.

### `SCI-MAP-OD-005` — Observation Maps And Coadd-Only Operation

**Resolved.** A complete logical observation-map bundle is required before an
observation can participate in coaddition. Persisted per-observation map
products are the normal scientific expectation and are required whenever the
effective output plan requests them. Physical persistence remains
plan-controlled so a fully in-memory pipeline is also permitted. A later
coadd-only entry point may consume persisted conforming observation-map
bundles and perform coaddition plus post-MAP filtering without rerunning the
upstream pipeline. Persisted and in-memory routes use the same admission,
arithmetic, product identity, and provenance rules. A required publication
failure propagates according to the effective plan; optional nonpublication
does not imply that the logical observation product did not exist.

### `SCI-MAP-OD-006` — Pointing And OOF Reuse

**Resolved.** Pointing and OOF may be registered consumers of the exact
ordinary MAP gridding operator when their inputs satisfy the operator's
contract. Reuse grants no mode-specific scientific meaning. The respective
mode contracts own target WCS and frame selection, source model and fitting,
response interpretation, mode outputs, and mode-specific validation. A mode
may select a different registered PTC coefficient family without changing the
MAP operator's ownership boundary.

### `SCI-MAP-OD-007` — Numerical Domain Of `coverage_cut`

**Resolved.** `coverage_cut` is finite and nonnegative. Exact zero is valid and
requests no relative cut beyond the independently required finite,
strictly-positive normalization domain. Values in `0 < coverage_cut <= 1` are
the ordinary recommended range. Values greater than one are permitted only as
an explicit expert request and may intentionally produce empty scientific
support. A negative, non-finite, missing, or unrepresentable value fails before
support rows are constructed or a required product is mutated. The requested,
effective, observation-resolved, and realized exact state/value are retained.
No universal numerical default is established here; a mode policy may own an
explicit versioned recommendation or default.

### `SCI-MAP-OD-009` — Common-Grid Preparation And Future Reprojection

**Resolved.** A canonical target grid may be requested before observation-map
construction. Compatible observation maps may then be placed into a common
coadd grid by exact centered-integer embedding without modifying their
scientific samples. Post-hoc cropping or resampling does not make an existing
map compatible. A crop, when scientifically requested, creates a new
row-selected product with its own identity and lineage. Fractional shifting,
interpolation, reprojection, and mosaicking require a separate future
scientific contract and are not ordinary SCI-MAP v0.1 coaddition operations.

`SCI-MAP-OD-008` was already resolved in the frozen packet by the one-hot,
half-open containing-pixel projection and is not reopened here.

## External PTC Coefficient Decision

### `PTC-OD-010` — Versioned MAP-Facing Coefficient Registry

**Resolved for successor architecture.** PTC shall support a versioned
registry of MAP-facing analysis/gridding coefficient families rather than one
mandatory universal formula. The user selects an allowed family from the
effective option set. A mode-owned policy may provide an explicit, versioned
default when the user supplies no selection. Requested, effective-policy,
observation-resolved, and applied/realized family identities remain distinct
and are recorded.

PTC produces the selected coefficient payload and owns each registered
family's scientific definition. For every use it supplies:

- exact family, version, generation, and owner identity;
- detector or detector-time index and any exact broadcast relation;
- compatibility with the transformed PTC product and immutable parentage;
- coefficient availability/QC and its causes, distinct from sample validity;
- statistic, factors, unit, normalization operator and domain, estimation
  population, support, lifecycle, uncertainty, and prohibited meanings; and
- the exact PTC-owned profile and requested/applicable/eligible/realized
  decision state.

MAP consumes the resulting typed coefficient values with the transformed
detector samples and their separately typed validity. MAP does not reproduce,
select, or infer the family's generating formula. It verifies exact registered
identity and binding, performs its own positive/zero/negative/non-finite/
unrepresentable numerical classification, and applies an admissible finite
strictly-positive value only to the matching occurrence. No family is treated
as inverse variance, precision, sensitivity, residual scatter, or another
statistic except where that meaning is explicitly established by the
registered PTC family.

An absent selection with no authorized mode default, an unregistered family,
a missing or mismatched payload, or an unavailable required coefficient/QC
decision makes the affected numerical route unavailable. There is no hidden
unity or alternate-family fallback.

Candidate families to recover and define include uniform/constant,
APT-sensitivity-based, and residual-scatter-based coefficients. A hybrid or
validated learn--apply family may be added later. These names are discovery
seeds, not registrations: existing implementation labels and behavior are
evidence to inspect, not scientific authority. Each usable entry still needs
an exact, versioned PTC definition satisfying SCI-PTC-REQ-052--055. Mode
recommendations likewise remain explicit versioned policy; ordinary faint
science, Pointing, and OOF need not select the same family.

## Successor Work Required

The owner-decision queue addressed in this review is closed, but the frozen
contracts are not changed by that closure. A bounded successor shall:

1. incorporate these eight MAP dispositions without renumbering existing
   normative IDs merely for editorial convenience;
2. replace the universal-family reading of `PTC-OD-010` with the registry and
   exact runtime-selection architecture above;
3. recover prior weighting work before defining the initial PTC registry
   entries, and explicitly adopt, supersede, defer, or exclude each candidate;
4. update the PTC-to-MAP boundary, MAP decision register, source manifests,
   crosswalks, and any affected VAL source bindings as one versioned set;
5. retain sample validity, coefficient availability/QC, MAP admission, and
   numerical contribution as separate typed propositions; and
6. repeat deterministic source, parity, boundary-equality, and horizontal
   contract review before any successor freeze.

Until at least one exact coefficient family is registered and the successor
authority is complete, the existing frozen r0.7.1 numerical-route limitation
remains an honest statement. No implementation work or validation result is
authorized or claimed by this record.
