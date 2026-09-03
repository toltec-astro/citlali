# SCI-FLT-FIXED v0.1 Scientific-Versus-Representation Identity Amendment

Record identity: `SCI-FLT-FIXED-IDENTITY-LAYERS-AMENDMENT v0.1/freeze-candidate`

Status: implementation-blind conditional scientific-owner freeze-candidate amendment; owner signature required

Scientific owner: Grant Wilson

Stage B date: `2026-08-31`

The shared normative core controls.

## Scientific operator identity

Scientific identity binds the canonical coordinate-domain
offset-to-coefficient relation; exact coefficient values and units;
`K_geom_science`, `K_nonzero`, and `K_req`; center, orientation, handedness,
phase, and subpixel convention; normalization; coordinate/WCS metric;
transfer qualification; edge rule; input and output scientific domains; and
support, response, covariance, lifecycle, and product science. Its digest is
independent of serialization.

## Representation identity

Representation identity separately binds `K_store`; dense, sparse, cropped,
or padded encoding; field and byte ordering; container or compression; its
digest; and its representation generation.

A representation-only change may create a new representation artifact or
generation but retains the scientific operator identity, FLT product identity,
and scientific generation and changes no `S_out`, arithmetic, response,
covariance, influence, lifecycle, or claim. A canonical scientific coefficient
map or other scientific-fact change creates a new scientific transformation
and product generation.

## Exact invariance fixture

```text
dense encoding != sparse encoding in representation identity
scientific operator identity is identical
FLT product identity and scientific generation are identical
```

Any serialization choice that changes scientific identity or behavior fails
this fixture.

## Nonclaims

This amendment supplies no implementation, numerical-adequacy, validation,
readiness, production, or Unity finding.
