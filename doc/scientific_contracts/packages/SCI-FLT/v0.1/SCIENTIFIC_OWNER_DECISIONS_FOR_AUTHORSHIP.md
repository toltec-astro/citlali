# SCI-FLT-FIXED v0.1 Scientific-Owner Decisions For Authorship

Decision identity: `SCI-FLT-FIXED_OWNER_DECISIONS v0.1/r0.1`

Scientific owner: Grant Wilson

Decision date: `2026-08-30`

Status: sanitized content-bound Stage A author input; exact packet approval and
Stage B launch remain pending

## ODQ-101 — Package Split And Name

`SCI-FLT` is the tranche. The first contract is `SCI-FLT-FIXED`. Do not use
`SCI-FLT-DET` because of the detector-namespace collision. `SCI-FLT-INF` is a
non-authoritative holding tranche only; do not launch one combined
inference-bearing contract. Wiener, matched/template-amplitude,
source-learned, data-derived spectral/mode selection, automatic method
selection, and per-member relearning remain separately pending.

## ODQ-102A — Strict Linear Base

SCI-FLT-FIXED v0.1 admits only

\[
  y=L_\Theta m,
  \qquad c_\Theta=0.
\]

Affine transformation is outside v0.1 and requires a versioned successor with
separate additive-term parent, unit, support, uncertainty, reference-mode,
response, covariance, NOI, lifecycle, and failure semantics. No offset,
background, template subtraction, or additive correction is inferred.

## ODQ-102B — Fixed Low-Pass Scope

Fixed convolution is the concrete family. A fixed-low-pass-convolution
subtype is admitted only when the exact resolved plan supplies spatial-
frequency domain and WCS metric, DC gain, passband, transition, stopband or
attenuation, phase, anisotropy, finite-grid/edge limitations, exact kernel and
normalization, and parameter provenance. Low-pass is a qualified claim, not a
synonym for smoothing or a second generic family. Missing transfer facts make
the low-pass claim unavailable while fixed-convolution identity may remain.

## ODQ-103 — Parent And Ordering

One method binds exactly one MAP observation, MAP coadd, or JINC observation
parent role. FLT does not coadd. Filtering an observation and filtering a
coadd are different products. No filter/coadd commutation is inferred. A
missing/unavailable numerical MAP or JINC parent remains unavailable.

## ODQ-104 — Full-Footprint-Only Method

Full-footprint-only convolution is the sole v0.1 edge/missing/non-finite
method. An output row is scientific only when every required kernel location
lies in the exact parent domain, is admitted, is finite, and passes all
required predicates. Other rows are unavailable, not zero. No boundary
extension, periodic wrap, truncated convolution, support renormalization,
clamp, mirror, edge completion, or missing-value replacement is admitted.
Those methods require separately named successors.

## ODQ-105 — Transformed Amplitude And Response

The output is the transformed parent-map amplitude in units derived from the
exact operator. It retains the originating nominal-beam identity and publishes
the filter-composed response or honest absence. It is not automatically a
flux, matched amplitude, filtered-beam calibration, preserved source peak,
integrated-flux product, extended-source-fidelity product, or target PSF.

## ODQ-106 — Covariance And NOI

SCI-FLT-FIXED owns deterministic propagation of an available declared parent
covariance through the exact operator, with complete, diagonal-input,
marginal, structured, partial, and unavailable states distinguished. Unknown
cross terms are not zero. SCI-NOI owns empirical uncertainty, covariance
inference, inverse conditional scale, and standardized signal. Historical
FLT-D003 placement of an empirical scale inside FLT is superseded.

## ODQ-107 — Inference-Bearing Methods And Failure

Inference-bearing families remain separate unless future Stage A work proves
one scientific identity. A method requiring unavailable state fails or is a
separately requested/named method; no silent fallback retains the requested
identity. Fixed-state and per-member-relearned methods cannot mix.

## ODQ-108 — Consumer And VAL Boundary

SCI-FLT-FIXED defines exact transformed products but authorizes no generic
Beammap, Pointing, OOF, source-fit, catalog, NOI, or FRUIT use. Each consumer
owns its exact use policy. FLT owns FLT input/publication policy; VAL binds and
evaluates without authoring facts or policy. NOI applies the exact FLT
operator to compatible members but does not choose it.

## Preserved Limits

These decisions authorize fresh implementation-blind authorship only after
separate exact-byte packet approval and explicit launch. They select no
numerical kernel, cutoff, WCS, parameter, implementation, or evidence state
and make no conformity, validation, calibration, response/covariance fidelity,
performance, readiness, freeze, production, or Unity claim.
