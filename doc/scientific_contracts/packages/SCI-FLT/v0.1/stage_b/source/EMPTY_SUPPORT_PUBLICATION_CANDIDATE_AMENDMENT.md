# SCI-FLT-FIXED v0.1 Empty-Support Publication-Candidate Amendment

Record identity: `SCI-FLT-FIXED-EMPTY-SUPPORT-PUBLICATION-AMENDMENT v0.1/freeze-candidate`

Status: implementation-blind conditional scientific-owner freeze-candidate amendment; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

The shared normative core controls.

## Candidate object

`complete_publication_disposition_candidate` has exactly two variants:

- `product_candidate`, containing the complete atomic FLT product bundle; and
- `no_output_support_candidate`, containing the exact request, parent, plan,
  operator, `K_req`, attempted output-domain construction, proof that `S_out`
  is empty, exact row/cause accounting, application generation,
  `applied_no_scientific_output_support`, and the prescribed publication cause.

The no-output-support variant contains no realized `FLT-SIG` and is not an
atomic FLT product bundle. `SCI-FLT-FIXED:output_publication@1` evaluates either
exact variant.

## Publication-use axes

For a requested nonzero convolution with empty `S_out`:

```text
request       = requested;
applicability = applicable;
eligibility   = ineligible;
realization   = not_produced;
cause         = no_full_footprint_output_rows.
```

The earlier input/application use may have been eligible and successfully
applied. The publication use is ineligible because no scientific signal rows
exist. It is not not-requested, disabled, execution failure, decision
unavailable, zero, or a realized empty product.

## Nonclaims

This amendment supplies no implementation, numerical-adequacy, validation,
readiness, production, or Unity finding.
