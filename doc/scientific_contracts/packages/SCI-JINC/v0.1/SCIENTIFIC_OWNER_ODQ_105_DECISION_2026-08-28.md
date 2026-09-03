# SCI-JINC-ODQ-105 — Scientific-Owner Disposition

Status: owner approved; bounded Stage A disposition

Scientific owner: Grant Wilson

Decision date: `2026-08-28`

Owner-decision source: supplied scope-control attachment, SHA-256
`c7c0760ef1a51ee549e8033806245bfdadb481a0fc986964906dfd2fbdb7b28b`.

## Approved Scientific Disposition

SCI-JINC v0.1 defines the estimator and its complete product bundle for one
observation. It defines and authorizes no cross-observation JINC combination
or coadd semantics.

This does not assert that JINC bundles from different observations cannot be
combined. Any future combination requires a separately authorized scientific
boundary whose inputs are complete observation-level JINC bundles. That
boundary must decide eligible realization identity, geometry, calibration,
admission, coefficient, parameter, normalization, conditioning, response,
covariance, support, validity, provenance and failure compatibility, and it
must define the exact object and algebra being combined. It may not import
ordinary SCI-MAP coaddition or infer that adding accumulator planes or
combining normalized maps is authorized.

“Observation” is the scientific grouping boundary, not an implementation,
streaming, chunking, process or memory boundary. Samples or processing chunks
from the same observation may contribute incrementally to the one complete
observation bundle when they share the exact observation, stable array, JINC
plan and realization, target WCS, admission/parameter/coefficient state and
lifecycle generation required by the contract. Processing-chunk identity does
not create a JINC product, coadd or independent scientific grouping.

## Stage Consequence

`SCI-JINC-ODQ-105` is closed for base-v0.1 observation/coadd scope. The
decision changes sanitized Stage A author-control bytes and remains subject to
renewed exact-byte approval under `SCI-JINC-STAGE-A-Q002`. It does not launch
Stage B, define a JINC coadd, prescribe implementation or memory architecture,
modify implementation, or establish conformity, validation, achieved
performance, readiness or production status.
