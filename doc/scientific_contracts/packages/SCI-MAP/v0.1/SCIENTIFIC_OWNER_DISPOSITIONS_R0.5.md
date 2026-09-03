# SCI-MAP v0.1/r0.5 Scientific-Owner Dispositions

Date: `2026-08-27`

Status: targeted owner-review dispositions; no implementation, validation,
performance, readiness, or production claim

## `SCI-MAP-R05-D001` — Physical Quantity

Select option B from the r0.5 directive. The ordinary input and base MAP role
are the calibrated-`x`-derived nonpolarimetric total-intensity-equivalent
quantity. They are not called Stokes I. The active unit is the inherited
top-of-atmosphere, point-source-equivalent `mJy/beam` convention and the
quantity identity includes the exact originating fixed-nominal-beam/template
and spectral/calibration lineage. A token cannot promote the quantity.

## `SCI-MAP-R05-D002` — PTC MAP-Facing Coefficient

Select option B. Frozen PTC r0.5 leaves `PTC-OD-010` open, so no ordinary
numerical MAP route is available until the PTC/MAP scientific owner selects
and versions an exact coefficient family satisfying SCI-PTC-REQ-052--055 and
the PTC-to-MAP boundary. MAP infers no unity, loading, sensitivity, scatter,
inverse-variance, or precision coefficient. Cross-generation use requires an
explicit compatibility record and cannot alter an earlier PTC product.

## `SCI-MAP-R05-D003` — Initial Projection Family

Resolve `SCI-MAP-OD-008`. Authorize
`SCI-MAP:one_hot_containing_pixel@1`. A finite in-grid coordinate belongs to
the unique half-open lower-inclusive/upper-exclusive pixel cell; `G_pi=1`
there and zero elsewhere. The outer upper boundary and out-of-grid coordinates
contribute nowhere. `sum_p G_pi=1` only on the finite in-grid domain.
Fractional projection is deferred. MAP owns the request and semantics; AST may
materialize the relation only with exact AST-coordinate-parent and MAP-plan
bindings.

## `SCI-MAP-R05-D004` — Exposure Carriage

Adopt the package ownership in the directive. ALIGN owns `e_acq` and `e_vo`
seconds on stable original occurrences. RTC/CAL/PTC preserve them and causes.
MAP owns upstream-eligible, retained, one-hot-projected, and coadded exposure
under exact admission. MAP uses original-occurrence union and deduplication;
replacement, synthesis, overlapping support, and repeated coadd parents create
no exposure. Exposure is not inferred from duration alone, cadence, count,
hits, `Q`, coefficient, or precision. Exposure is required as an explicitly
typed bundle role; missing lineage makes exposure unavailable and blocks only
an exposure-qualified required product or claim unless the effective plan
declares exposure a required complete-bundle companion.

## `SCI-MAP-R05-D005` — Coadd Coefficient And Admission

Authorize the MAP-owned family
`SCI-MAP:uniform_observation_coadd_coefficient@1`. It is dimensionless,
observation-row indexed, exactly one on the support of every atomically
admitted bundle, fixed by the effective coadd plan, and represents equal
observation averaging. Its normalization domain is the exact per-pixel set of
admitted observation rows. It is not inverse variance, precision, a covariance
summary, or a claim of equal noise.

Authorize `SCI-MAP:observation_coadd_admission@1` over complete observation
bundles. It binds compatible quantity and nominal beam, realized PTC route and
product role, response source/class, complete AST WCS/grid, support policy,
coefficient family, covariance disclosure, null/additive-reference state,
required companions, lifecycle, and parentage. Missing/conflicting required
compatibility rejects the entire observation before mutation.

An observation with unavailable response may enter the named
response-independent base-coadd role. The numerical signal coadd remains
permitted; coadd response is typed unavailable if any signal member lacks an
exact compatible response, and no hidden subset or zero response is formed.
Incomplete covariance is honestly disclosed and is never zero-filled.

## `SCI-MAP-R05-D006` — Upstream Admission

Authorize `SCI-MAP:map_upstream_admission@1` exactly as registered by SCI-VAL.
MAP owns the policy; VAL Registry binds it and VAL Core evaluates it. An
eligible occurrence is only a route candidate. Projection and final numerical
contribution remain MAP actions.

## `SCI-MAP-R05-D007` — Response And Bias Notation

Remove the undefined `mu_0` from the authoritative expectation equation. Use
the conditional perturbation response `delta E[mhat|Theta]=A H delta s`.
Carry PTC `lambda`, fitted correlated removed component, total removed
component, removed subspace, fixed-state null space, full-procedure
invariant/unidentifiable modes, and any separately authorized residual
conditional bias as distinct typed facts. `H` always names one exact response
family. A realized PTC-grid companion begins at MAP.

## Blocking Triage

| Class | State after r0.5 |
| --- | --- |
| A — any numerical single-observation map | **Open:** exact PTC coefficient family (`PTC-OD-010`); admitted numerical `coverage_cut` value/domain (`SCI-MAP-OD-007`). **Closed by r0.5:** MAP admission profile, exact same-`n` signal/coordinate join, one-hot `G_pi`, exposure role and missing behavior. Therefore no generally authorized ordinary numerical route is claimed. |
| B — response/uncertainty claims | **Open:** `SCI-MAP-OD-003`, `SCI-MAP-OD-004`, and PTC full-procedure response inputs/domain. Base signal validity remains separable from typed response/covariance availability. |
| C — coadd | **Closed for the centered-integer base branch:** uniform observation coefficient, coadd admission profile, and unavailable-response policy. **Open:** `SCI-MAP-OD-005` publication abstraction and `SCI-MAP-OD-009` canonical crop/pad or future reprojection ownership. |
| D — optional/future | `SCI-MAP-OD-006` Pointing/OOF reuse, fractional projection, correlated GLS, and mosaicking remain deferred. |
