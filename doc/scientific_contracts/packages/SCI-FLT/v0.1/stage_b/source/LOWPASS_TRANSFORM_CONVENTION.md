# SCI-FLT-FIXED v0.1 Low-Pass Transform Convention

Record identity: `SCI-FLT-FIXED-LOWPASS-TRANSFORM-CONVENTION v0.1/draft-r0.3`

Status: required convention fields, not a realized low-pass plan

Scientific owner: Grant Wilson

Stage B date: `2026-08-30`

## Required content binding

Every `FLT-FIXED-CONV-LOWPASS` plan binds the exact transform sign and
normalization; coordinate and spatial-frequency units; zero-frequency origin
and ordering; positive, negative, and Nyquist treatment; exact frequency
sample grid; declared response quantity; linear-ratio or decibel attenuation;
passband, transition, and stopband geometry; phase branch or unwrapping;
anisotropy; WCS metric; DC gain; finite-grid limits; sampled kernel; and
parameter provenance.

For a pixel-index kernel, a permitted reference form is

```text
H(nu) = sum over r of k(r) exp[-2 pi i nu dot r].
```

This expression is not a default. The adopted convention must be content-bound.
PRED-024 uses the identical convention; complete metadata under a different
convention fails. Sampled/interior transfer remains distinct from the complete
finite row-restricted `A_Theta,J`.

## Nonclaims

This record supplies no plan values, achieved transfer, validation, numerical
adequacy, performance, readiness, or production claim.
