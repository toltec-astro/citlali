# WP-7 CAL Quality-Token Compatibility Proposal

Status: **approved binding scientific-owner authority**

Prepared: `2026-08-25`

Scientific owner: Grant Wilson

If the scientific owner replaces the status above with
`approved binding scientific-owner authority`, every proposed rule below
becomes binding for the exact sources named in Section 1. No separate semantic
inference is authorized.

## 1. Proposed bound scope

This proposal applies only to the exact following sources:

| Source | SHA-256 |
| --- | --- |
| WP-7 approved scientific-authority addendum | `7a0a92a411f4d93f321257fba5cdbc561a4249f5d49cc49b8fe974f87e77d577` |
| Frozen SCI-CAL definitions | `2a9c91f485ea7d41ba6d5b13c77f77b8314612da4bc4b59eb1228235374b71b5` |
| Frozen SCI-CAL requirements | `80054fbd526d6a0878f6724c620024955062d41fca1273b85836ead3ee9b5f74` |
| Frozen SCI-CAL scientific rationale | `f780cef579cb39ac1ed748f021a0024d9f1576d7960fe3b4363557c10bbff318` |
| Continuing SCI-VAL source-binding register | `ff5402b71c40f31daac1f7c820a705a5a23eb64688f70955fac76e10e2916430` |
| Continuing SCI-VAL profile registry | `5a5a96a283ab6bd3aa6176548b11a9798ec6a12a0b430277eecd7c2caf752893` |
| Frozen SCI-PTC requirements | `f2047600cc06c234a78aa3ddf6a575abf2f9592b3e3da810491f6db0150fe21c` |

It changes no threshold, classifier precedence, numerical-support rule,
profile consequence, or scientific claim. It defines exact vocabulary
compatibility while keeping observation, sample, and operator-regime scopes
distinct.

## 2. Observation-wide opacity quality class

The canonical field is the observation-wide output class of
`cal_wvr_observation_quality_mean_peak_v1`. Its canonical machine tokens are:

```text
invalid_opacity_input
opacity_quality_unavailable
outside_supported_opacity
science_qualification_eligible
engineering_only
```

Only the following frozen SCI-CAL spellings are exact aliases in that
observation-wide class field:

| Frozen spelling | Canonical token |
| --- | --- |
| `science-qualification-eligible` | `science_qualification_eligible` |
| `engineering-only` | `engineering_only` |
| the defined term `engineering-availability` | `engineering_only` |

The `engineering-availability` alias is limited to the frozen SCI-CAL term for
an observation retained for engineering use under the supported operator. It
does not alias the separately named atmosphere-operator regime in Section 4.

The existing SCI-VAL and SCI-PTC profile spelling `engineering-only` is
therefore the exact legacy spelling of the canonical observation-class token
`engineering_only`. Every existing profile consequence is preserved: the
class remains a producer fact, does not itself prohibit the named PTC
mathematics, and never creates CAL science qualification.

The other three canonical observation-class tokens have no spelling-only
legacy alias. Their meanings are governed directly by the approved successor
classifier.

The frozen observation-level quality label `outside-supported calibration`
is superseded, not aliased, for successor replay. The exact successor
classifier inputs and causes determine whether the canonical result is
`outside_supported_opacity`, `opacity_quality_unavailable`, or
`invalid_opacity_input`. If those inputs and causes are not available, the
successor class is unavailable; a raw frozen label alone must not be converted
by spelling. This cause-preserving split resolves the earlier label's combined
domain/input role without changing any sample-local state.

## 3. Sample-local CAL availability and validity

The sample-local CAL state is not the observation-wide opacity class.

| Frozen or rendered spelling | Canonical sample-local state |
| --- | --- |
| `outside-supported-calibration` | `outside_supported_calibration` |
| `outside supported calibration` | `outside_supported_calibration` |
| `invalid_atmosphere` | `invalid_atmosphere` |

`outside_supported_calibration` means that the affected sample lacks an
admissible numerical CAL result under its operator/input/domain authority.
`invalid_atmosphere` is its sample-local invalid-input state. Neither is an
alias for an observation-wide class.

In particular:

- `outside_supported_calibration` is not an alias for
  `outside_supported_opacity`;
- `invalid_atmosphere` is not an alias for `invalid_opacity_input`; and
- `opacity_quality_unavailable` does not erase independently supported
  sample-local CAL results.

When a frozen record contains the prose or hyphenated phrase “outside
supported calibration,” its owning field and object scope must determine
whether it is the sample-local state above or the superseded observation-level
label governed by Section 2. A raw token without that scope is insufficient
and fails closed; it must not be guessed into either state.

## 4. Atmosphere-operator regimes

The exact machine-contract regime labels remain:

```text
science_qualification_regime
engineering_availability_regime
```

They describe the numerical atmosphere operator's opacity-support regimes.
They are not observation-wide quality-class tokens and are not renamed by this
crosswalk. In particular, `engineering_availability_regime` is not an alias
for `engineering_only`; the classifier must still use its complete
observation-wide inputs and precedence rules.

## 5. Canonicalization and provenance

1. Preserve every raw source token and its owning field as provenance.
2. Canonicalize only after establishing the exact bound source, field, and
   object scope.
3. Apply only the explicit aliases in Sections 2 and 3.
4. Never compare or join an observation class, sample-local CAL state, and
   operator regime merely because their names or thresholds appear related.
5. An unknown spelling, missing scope, conflicting class, or source digest
   mismatch fails closed and remains unavailable.

## 6. Proposed precedence and limitations

Approval would supersede only the missing exact token compatibility and scope
crosswalk among the bound sources. The approved successor classifier remains
the authority for observation-wide class computation. Frozen SCI-CAL remains
the source of the preserved profile consequences and sample/operator meanings
except where the successor authority explicitly supplies the resolved
observation classifier or sample-local unavailable-opacity behavior.

Approval would not establish implementation conformity, observational
validation, achieved calibration performance, production readiness, or a
downstream science-quality decision.
