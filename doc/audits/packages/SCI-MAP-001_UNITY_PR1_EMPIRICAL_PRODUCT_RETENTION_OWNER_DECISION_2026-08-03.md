# SCI-MAP-001 MAP-UNITY-PR1 empirical-product retention owner decision — 2026-08-03

Status: owner approved; applies to interpretation and presentation of the
returned empirical-product inventory; it authorizes no implementation change

Package: `SCI-MAP-001`

Evidence request: `SCI-MAP-001-UNITY-001`

Decision ID: `MAP-UNITY-PR1`

Authority: project owner

## Decision

The differing placement of empirical-noise products is intentional product
policy, not evidence that empirical realization generation was skipped.
Keeping a separate realization ensemble for every individual observation when
a coadd is requested would impose an unacceptable accumulating disk-storage
cost. The accepted retention policy is therefore:

- In a successful empirical coadd reduction (`S-X`), retain the compact
  per-observation empirical planes in the ordinary signal FITS products:
  `noise_variance_I`, `sig2noise_I`, and `sig2noise_pixel_I`.
- Retain the separate 64-realization ensemble product for the coadd only.
- In explicit non-coadd empirical mode (`S-E`), retain the separate
  realization products per observation as the requested product form.

This decision applies to the product's location and cardinality, not to the
definition, generation, numerical validity, or acceptance of the empirical
noise realization. A missing separate per-observation realization file in
`S-X` must not by itself be reported as a missing empirical calculation.

## Consumer and provenance rule

A product inventory or provenance manifest for an empirical reduction must
state whether a realization ensemble is retained per observation or only at
the coadd, and must identify the compact per-observation empirical planes.
A consumer that genuinely needs a per-observation realization ensemble must
request explicit non-coadd empirical mode or return for a separately scoped
product-policy decision; it must not silently infer that ensemble from a
coadd-mode output.

The returned `S-E`/`S-X` inventory comparison is sufficient to establish this
placement policy. No processed-time-chunk capture is required merely to decide
whether the absence of per-observation ensemble files in coadd mode is a
failure. Such a capture remains governed separately by `MAP-UNITY-ED2` if it
is needed for a different primitive-level claim.

## Unchanged boundaries

This decision does not:

- modify application source, build configuration, numerical behavior, or the
  frozen seven-case protocol;
- establish the correctness of the empirical estimator, realizations, F010
  semantics, F011 semantics, WCS, coaddition, or sequential/OpenMP behavior;
- close any MAP finding or dependency, supply or accept the complete Unity
  evidence package, integrate the repair, launch a re-audit, or expand
  production use; or
- replace the `MAP-UNITY-ED1` bounded evidence policy or the `MAP-UNITY-ED2`
  full/all capture and temporary-retention policy.

## State effect

`MAP-UNITY-PR1` resolves only the product-retention interpretation raised by
the returned `S-E` and `S-X` product inventory. It records a durable storage
and consumer-policy choice while all broader SCI-MAP-001 evidence and
conformance gates remain unchanged.
