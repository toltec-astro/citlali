# SCI-FLT-FIXED v0.1 Policy and Action Architecture

Record identity: `SCI-FLT-FIXED-POLICY-ACTION-ARCHITECTURE v0.1/draft-r0.4`

Status: unregistered Stage B profile architecture draft; scientific-owner review required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

## Policy domains

```text
SCI-FLT-FIXED:input_bundle_admission@1
  evaluates one complete parent bundle and resolved plan

SCI-FLT-FIXED:input_parent_row_admission@1
  evaluates each exact parent row for the named FLT use

SCI-FLT-FIXED:output_publication@1
  evaluates one complete candidate and defines disposition plus action
```

`J_full` and `S_out` are deterministic FLT-owned constructions from parent-row
decisions, `K_req`, parent domain, availability, finiteness, and predicates.
Profiles do not perform convolution or author output-row arithmetic support.

For a requested nonzero convolution with empty `S_out`, application records
`applied_no_scientific_output_support`, retains its evidence, and publication
prescribes `not_produced` with cause `no_full_footprint_output_rows`. Identity,
zero, disablement, and publication failure retain their distinct branches.

`not_requested_at_FLT_publication` is historical FLT provenance and does not
block a later independently requested compatible NOI child. The child owns its
lifecycle, and no child or reverse relation mutates FLT.

SCI-VAL may bind and evaluate a future owner-approved immutable profile and
produce a decision artifact. The FLT publisher performs or declines the
prescribed action. FLT owns realization and FLT-local validity. A policy or VAL
evaluation does not perform publication.

## Registration state

All three r0.3 profile records remain drafts, not owner-approved Registry
entries, not registered, and not Registry-evaluated. This r0.4 closure binds
their exact action semantics but creates no approval by name.
