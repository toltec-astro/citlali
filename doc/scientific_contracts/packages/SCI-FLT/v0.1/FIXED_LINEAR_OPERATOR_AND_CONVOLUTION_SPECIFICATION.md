# SCI-FLT-FIXED v0.1 Linear Operator And Convolution Specification

Status: sanitized Stage A author candidate awaiting exact-byte owner approval

## Scope

Base v0.1 defines one fixed linear same-grid map-domain transformation:

\[
  y = J_{\rm full} L_\Theta m.
\]

`m` is one exact admitted MAP observation, MAP coadd, or JINC observation
parent. `L_Theta` is completely resolved and frozen before application.
`J_full` selects the exact full-footprint scientific output-row domain. There
is no additive term.

## Complete Realized Operator

For parent vector `m` on exact domain `S_in`,

\[
  y_p = \sum_{q\in S_{\rm in}} L_{pq} m_q,
  \qquad p\in S_{\rm out}.
\]

The method identity binds:

- input and output row domains;
- exact parent and output WCS, frame, topology, grid, metric, shape, and pixel
  area;
- operator identity/version and complete sampled coefficients;
- coordinate domain, coefficient units, orientation, and handedness;
- kernel center, finite support, even/odd extent, tie, phase, and subpixel
  conventions;
- normalization and any qualified transfer claim;
- exact edge/crop/full-footprint rule;
- numerical coefficient representation;
- response, covariance, null-space, and influence state;
- requested/effective/resolved/applied/realized lifecycle generation; and
- causes, failure behavior, and immutable provenance.

A discrete convolution kernel is a structured construction of `L_Theta`. The
complete finite `L_Theta`, not a friendly kernel name or continuous ideal, is
the scientific transformation.

## Fixed Convolution Family

For finite kernel support `K_Theta`,

\[
  (L_\Theta m)_p
    = \sum_{r\in K_\Theta} k_\Theta(r)m_{p-r}.
\]

The kernel, cutoff/width if present, sampled coefficients, normalization,
support, and all discretization facts are externally resolved and frozen. No
kernel, cutoff, threshold, support, or normalization is learned from the
parent or from individual NOI members.

## Qualified Fixed-Low-Pass-Convolution Subtype

Low-pass is an optional qualified claim on an exact fixed convolution, not a
second generic operator class. It is available only when the resolved plan
binds:

- spatial-frequency domain and WCS metric;
- DC gain;
- passband;
- transition region;
- stopband or attenuation criterion;
- phase;
- isotropic/anisotropic state;
- finite-grid and edge limitations;
- exact kernel and normalization; and
- parameter identity, source, and provenance.

If any required transfer fact is unavailable, the operator may remain a fixed
convolution while its low-pass claim is unavailable.

## Full-Footprint Scientific Domain

The sole v0.1 output domain is

\[
S_{\rm out}=\{p:\ p-r\in S_{\rm in},\ m_{p-r}\text{ is admitted and finite,
and every required predicate passes, for all }r\in K_\Theta\}.
\]

Rows outside `S_out` are scientifically unavailable, not zero. Stored shape
and WCS may remain parent-shaped while the scientific vector is restricted to
`S_out` and unavailable rows retain explicit causes.

Base v0.1 performs no boundary extension, periodic wrapping, truncation,
support renormalization, inpainting, reflection, clamping, or replacement of
missing/non-finite values.

## Units, Response, Covariance, And NOI

Output units follow the exact operator. For an available compatible parent
response and covariance,

\[
  R_{\rm out}=J_{\rm full}L_\Theta R_{\rm parent},
\]

\[
  C_{\rm out}=J_{\rm full}L_\Theta C_{\rm parent}
  L_\Theta^{\mathsf T}J_{\rm full}^{\mathsf T}.
\]

Unknown response/covariance remains unavailable. Parameter, beam, WCS,
selection, and model uncertainty remain separate from fixed-state conditional
propagation.

For every compatible admitted NOI member,

\[
  M_b^{\rm out}=J_{\rm full}L_\Theta M_b.
\]

The same operator and row domain apply to signal, response, covariance, and
NOI members.

## Excluded Methods

Base v0.1 excludes affine offsets, reprojection, resampling, deconvolution,
fixed or learned boundary extension, truncated convolution, support
renormalization, data-derived kernels/cutoffs, automatic method selection,
Wiener inference, matched/template-amplitude estimation, source-learned
subtraction, and per-member re-resolution.

## Required Predictions

The future contract must state falsifiable predictions for:

- identity and zero kernels;
- unit scaling;
- constant input under every authorized normalization;
- impulse and source-response composition;
- signed and zero-sum kernels;
- full-footprint admission and unavailable edge rows;
- rejection of every deferred edge method, including support-conditioned
  renormalization;
- missing and non-finite parent input;
- output covariance and induced cross-pixel covariance;
- unavailable parent response/covariance;
- parent WCS mismatch;
- observation/coadd noncommutation;
- exact fixed-state NOI parity and per-member re-resolution rejection;
- disabled versus identity versus zero transformations; and
- upstream unavailable parents.
