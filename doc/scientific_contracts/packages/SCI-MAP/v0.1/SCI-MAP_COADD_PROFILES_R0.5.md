# SCI-MAP v0.1/r0.5 Coadd Coefficient And Admission Profiles

Status: targeted owner-review MAP policy; no implementation or validation
claim

## `SCI-MAP:uniform_observation_coadd_coefficient@1`

- **Owner:** SCI-MAP scientific owner.
- **Index/domain:** exact observation-output row `(o,p)` after centered-integer
  placement and atomic coadd admission.
- **Value/unit:** `u_op = 1`, dimensionless.
- **Normalization:** per output pixel over the exact set `O_p` of atomically
  admitted observation rows; `Q_p^c = sum_o u_op`.
- **Support/lifecycle:** fixed by the immutable effective coadd plan and valid
  only for the exact admitted observation/product generation and row domain.
- **Meaning:** equal-observation arithmetic averaging.
- **Prohibited meanings:** inverse variance, precision, covariance summary,
  equal noise, empirical weight, exposure, or scientific significance.
- **Uncertainty relation:** covariance propagates separately through
  `B_out C_obs B_out^T`; the coefficient is not derived from `C_obs`.

## `SCI-MAP:observation_coadd_admission@1`

- **Owner/use:** SCI-MAP; atomic admission of one complete observation bundle
  to the centered-integer base coadd.
- **Object:** immutable observation MAP bundle with exact product and row
  identities.
- **Required compatibility:** nonpolarimetric total-intensity-equivalent
  quantity; `mJy/beam` plus exact nominal-beam/template convention; realized
  PTC route/product role and compatible application generation; response
  source/domain/class state; complete AST frame/WCS and identical grid;
  centered-integer shape/reference-pixel relation; support policy; exact
  uniform coadd-coefficient family; covariance representation/disclosure;
  PTC null/removed-component/additive-reference state; exposure convention;
  required companions; requested/effective/resolved/realized lifecycle; and
  immutable parentage.
- **Decisive exclusions:** different quantity/beam/grid/frame, fractional
  shift, crop/pad not already authorized in a parent product, reprojection,
  mosaic, incompatible PTC or policy generation, missing required companion,
  or conflicting identity.
- **Missing/conflict:** reject the entire observation before any coadd state
  changes and preserve exact cause.
- **Response-unavailable policy:** a bundle with an honestly unavailable
  response may enter the named response-independent base-coadd role. If any
  admitted signal member lacks an exact compatible response, the coadd signal
  may remain available while coadd response is typed unavailable. No hidden
  subset or zero response is formed.
- **Covariance policy:** incomplete or unavailable covariance is carried
  honestly and limits covariance-dependent claims; unknown blocks are never
  zero or independence. It does not alone invalidate the signal coadd.
- **Consumer action:** after atomic admission, MAP applies centered-integer
  placement, exact row support, equal-observation accumulation, exposure union,
  and MAP-local coadd validity. Admission itself performs no arithmetic.
- **Supersession:** any changed field requires a new immutable profile version
  and cannot rewrite prior admission decisions or products.
