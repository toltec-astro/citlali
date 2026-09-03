# SCI-JINC-ODQ-110 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

## Approved Center-Admission Rule

SCI-JINC v0.1 admits an occurrence to a finite destination map only when the
resolved rounded sample center used for JINC square-cache placement lies in
the finite destination pixel domain. The center-domain test occurs before any
sample-pixel footprint evaluation.

If that rounded center is outside the finite map, the occurrence contributes
nowhere, even when part of its JINC square would overlap valid destination
pixels. Equivalently,

```text
center_i outside finite destination map
  => I_ip = 0 for every destination pixel p
  => Delta N_p = Delta C_p = Delta Q_p = Delta T_p^(kappa^2) = 0
     for every destination pixel p.
```

No footprint-overlap admission is permitted for an out-of-map center. The
authoritative center for this gate is the same single-valued rounded center
used by the ODQ-109-compliant point-phase/cache realization; an independent
containing-pixel or alternate center is not substituted.

## In-Map Centers And Ordinary Edge Crop

For an admitted in-map center, SCI-JINC evaluates the resolved fully populated
square support and omits only square pixels outside the finite destination
map. Retained pixels use their actual admitted membership. No wrap,
reflection, full-interior renormalization, footprint completion, edge-response
correction, or replacement contribution is introduced.

Rejecting an out-of-map center is an ordinary map-domain no-contribution
result. It does not invalidate the complete per-array observation bundle,
make a required product role unavailable, require a placeholder, create a new
detailed cause vocabulary, or require per-occurrence, per-pixel, provenance or
diagnostic products. Existing operational logging may record the decision for
debugging without becoming scientific output.

## Scientific Consequence

Constructing a larger JINC map and then cropping it need not be exactly
equivalent to constructing the smaller finite map directly. JINC-then-crop
equivalence is not a scientific requirement of SCI-JINC v0.1.

The rule is pragmatic: partially overlapping footprints from occurrences
centered outside the requested map are expected to provide little scientific
benefit near the outermost boundary, while center admission gives the finite
map one clear occurrence boundary and avoids additional edge-specific
accumulation semantics. This rationale does not establish achieved weighting,
response, performance, or validation.

## Stage Consequence

`SCI-JINC-ODQ-110` is closed for Stage A. No unresolved numbered SCI-JINC
scientific-scope ODQ remains in the current ledger. The next scientific-owner
decision is `SCI-JINC-STAGE-A-Q002`, exact-byte approval or revision of the
complete successor Stage A packet and firewall. A versioned SCI-VAL registry
binding for `SCI-JINC:jinc_map_contribution@1` remains a separate Stage B
dispatch prerequisite.

This disposition changes sanitized Stage A author-control bytes and remains
subject to renewed exact-byte approval under `SCI-JINC-STAGE-A-Q002`. It does
not launch Stage B, alter frozen authority, prescribe implementation
representation, create edge-response/provenance/diagnostic products, perform
validation, or establish conformity, achieved performance, readiness, or
production status.
