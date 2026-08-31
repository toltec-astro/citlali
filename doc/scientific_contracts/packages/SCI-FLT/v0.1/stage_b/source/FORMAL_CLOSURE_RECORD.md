# SCI-FLT-FIXED v0.1 r0.3 Formal-Closure Record

Document identity: `SCI-FLT-FIXED-FORMAL-CLOSURE v0.1/draft-r0.3`

Status: implementation-blind Stage B closure draft; scientific-owner review required

Scientific owner: Grant Wilson

Normative authority remains the shared core. This record makes the r0.2/r0.3 owner
dispositions and cross-package boundaries easy to review; it does not add a
scientific rule beyond the core.

## 1. Fixed-family scope disposition

The v0.1 disposition is the preferred narrow rule:

- `FLT-FIXED-CONV` is the only numerically admitted base family.
- `L_Theta` is the complete matrix representation of one exact realized
  sampled convolution, not an arbitrary selectable dense operator.
- `FLT-FIXED-CONV-LOWPASS` is only a qualified subtype.
- One product applies one resolved sampled convolution exactly once.
- A final kernel may be constructed elsewhere; FLT claims no intermediate
  transformation, composition, selector collapse, or reordering.

## 2. Conditional-row-selector amendment

`J_full` is resolved once before payload arithmetic from the plan, immutable
parent membership, bundle and row admission, availability, finiteness,
`K_req`, and required predicates. The complete applied operator is
`A_Theta,J = J_full L_Theta`.

Strict linearity is conditional on that exact frozen parent membership and
selector. Response perturbations, covariance draws, noise realizations, and
NOI members reuse `A_Theta,J`. A member that cannot supply a frozen required
footprint is unavailable there. Selection and support uncertainty remain
excluded unless separately supplied as typed uncertainty.

## 3. Required-footprint and zero-operator disposition

- `K_geom_science` is representation-invariant scientific geometry.
- `K_store` is nonauthoritative serialization footprint.
- `K_nonzero` is exact canonical nonzero support; exact zero never uses a
  tolerance.
- The ordinary method uses `K_req = K_nonzero`.
- Dense, sparse, cropped, and zero-padded representations preserve all
  scientific support, output, response, covariance, and identity facts.
- Identity uses `K_req = {0}` and preserves the exact admitted finite parent
  row domain.
- Zero inherits an explicit admitted finite parent-support row domain and does
  not gain rows from a vacuous empty-footprint predicate.

Geometric, storage, nonzero, required-dependency, signed, absolute, and squared
support are separate objects. A required exact-zero offset needs a separately
named method and a scientific reason independent of storage.

## 4. Publication-lifecycle amendment

The lifecycle is:

```text
requested
  -> effective
  -> resolved
  -> applied
  -> complete_publication_candidate
  -> publication_decision
  -> realized | failed | not_produced.
```

Publication policy evaluates the complete candidate and defines disposition
and prescribed action. VAL may produce a decision artifact. Only the FLT
publisher performs or declines publication and creates realization. Disabled
is not produced. Identity and zero follow the same sequence.

## 5. Immutable NOI compatibility amendment

The actual `NOI-UNC[FLT-SIG]` product is outside FLT atomic completion.
`FLT-NOI-COMPATIBILITY` contains exact FLT/operator/row identity, boundary and
profile compatibility, fixed-state semantics, publication-time request, and
typed compatibility or unavailability. It contains no future NOI identity. A
later NOI child references FLT; an optional reverse relation is separately
versioned and never changes FLT completeness or realization.

## 6. Profile records and VAL status

The exact draft bytes are in `POLICY_RECORDS.json`:

- `SCI-FLT-FIXED:input_bundle_admission@1`;
- `SCI-FLT-FIXED:input_parent_row_admission@1`; and
- `SCI-FLT-FIXED:output_publication@1`.

They distinguish base signal, response-qualified, covariance-qualified, and
jointly qualified requests. Parent-row decisions feed FLT-owned construction
of `J_full` and `S_out`. Policy defines publication disposition; VAL may return
a decision artifact; the FLT publisher acts. They are not owner-approved
Registry entries, and no Registry evaluation is claimed.

## 7. Response-family crosswalk

| Family | Applied relation | Retained state | v0.1 status |
| --- | --- | --- | --- |
| Fixed-state linear parent response | `R_out^fixed = A_Theta,J R_parent^fixed` | Frozen plan and selector | Admitted when compatible |
| Realized parent-grid response companion | Apply `A_Theta,J` exactly once | Exact companion identity | Admitted when compatible |
| Parent full-procedure finite difference, FLT fixed | `Delta y_parent-FP = A_Theta,J Delta m_parent-FP` | Exact compatible frozen domain and parent state-change record | Admitted when compatible; otherwise affected rows unavailable without re-selection |
| FLT re-resolved procedure response | No SCI-FLT-FIXED relation | Re-resolved FLT state | Outside v0.1 |

The zero operator establishes a local zero derivative and zero parent-payload
conditional covariance contribution. A complete source-domain response may
remain unavailable, and total systematic uncertainty remains separate.

## 8. Covariance authority and representation table

`COVARIANCE_COMPATIBILITY_TABLE.md` binds the exact five-row compatibility
decision. Complete covariance permits exact two-sided propagation; an explicit
independent-diagonal model permits full model-conditional covariance including
induced cross rows; marginal-only authority does not infer independence or
exact mixed-row marginals; structured/partial authority permits only proved-
exact operations; unavailable authority remains unavailable. The general
variance identity retains every weighted parent cross term.

## 9. WCS, low-pass, and transfer decision

Every convolution identifies pixel-index, affine tangent-plane angular, or
another exact coordinate-domain method. `LOWPASS_TRANSFORM_CONVENTION.md`
binds every required sign, normalization, unit, origin, ordering, signed and
Nyquist treatment, grid, response quantity, attenuation, band geometry, phase,
anisotropy, and WCS relation. The sampled-kernel transfer remains separate from
complete finite `A_Theta,J`.

## 10. Operator-composition decision

The preferred narrow rule is adopted: one product applies one exact resolved
sampled convolution. Base v0.1 admits no ordered multi-operator composition and
no identity such as collapsing intermediate selectors into one final selector.

## 11. Numerical-conformance policy disposition

`NUMERICAL_CONFORMANCE_POLICY.md` is a draft future-evidence policy. It freezes
an independent oracle, comparison regimes, cancellation and conditioning,
covariance, sequential and parallel agreement, non-finite handling, row-level
decisions, lifecycle, and provenance before candidate results. It supplies no
validation, numerical-adequacy, or performance finding.

## 12. Exposure, terminology, and ownership disposition

`FLT-EXPOSURE-LINEAGE` records exact parent exposure identity or typed absence.
FLT creates no physical exposure; influence is not exposure; convolving an
exposure plane is not authorized as filtered-signal physical exposure.

`confidence` is either an exact named upstream quality state with preserved
meaning or `not_defined`. It is not inferred from support, finiteness,
covariance, weight, or downstream eligibility.

"Externally resolved" means complete before application under one exact FLT-
owned plan. FLT retains policy, selector, application, product, publication,
and failure ownership.

## 13. Source-packet closure

`BUILD_BINDING.json` binds the author packet, all 17 admitted objects, both
owner directives, every authored source and tool, embedded fonts, and PDFs by
exact byte count and SHA-256. `AUTHORITY_MANIFEST.json` supersedes manual
combination by binding the complete final authority, report, source, tool, and
PDF set. Its external digest is the single proposed-freeze entry point.

## 14. Nonclaims

This closure record makes no implementation-conformity, achieved-response,
achieved-covariance, numerical-adequacy, validation, calibration, observational
performance, readiness, scientific-freeze, production-suitability, production-
authorization, or Unity claim.
