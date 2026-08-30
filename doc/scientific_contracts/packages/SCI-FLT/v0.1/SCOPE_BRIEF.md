# SCI-FLT-FIXED v0.1 Stage A Scope Brief

Status: final repaired scientist-readable Stage A candidate; exact-byte owner
approval required; Stage B not authorized

## Program Adherence And Prior-Work Recovery

This Scope Brief follows the Scientific Contract Library Program, pilot
process, and owner-approved downstream roadmap. Prior-work and implementation
recovery preceded the brief. Implementation, configuration, schemas, tests,
audits, repairs, validation, reductions, generated products, defaults,
historical behavior, and production state remain outside the future author
packet.

The raw historical Convolve record is withheld. Only sanitized fixed-linear
mathematics appears in
[`AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md`](AUTHOR_DETERMINISTIC_TRANSFORMATION_EXTRACT.md).
Exact MAP, JINC, and NOI boundaries are supplied as dedicated compact objects.

## Package And Tranche Decision

`SCI-FLT` remains the recovery tranche. The first scientific contract is
`SCI-FLT-FIXED v0.1`. The name `SCI-FLT-DET` is rejected because of the
detector-namespace collision.

`SCI-FLT-INF` is a non-authoritative holding tranche only. No combined
SCI-FLT-INF Stage B contract is authorized. Wiener, matched/template-amplitude,
source-learned, data-derived spectral/mode selection, automatic method
selection, and per-member relearning remain separate pending Stage A work.

## Base Scientific Object

Base v0.1 defines one fixed linear same-grid map-domain transformation:

\[
  y = J_{\rm full}L_\Theta m,
\]

where `m` is one exact admitted parent, `L_Theta` is the complete externally
resolved fixed operator, and `J_full` selects exactly the rows whose complete
kernel footprint is admitted and finite. There is no additive term.

Fixed convolution is the concrete transformation family. A fixed-low-pass-
convolution subtype is a qualified scientific claim only when its complete
transfer specification is bound. It is not a second generic operator class.

## In Scope

- one immutable MAP observation, MAP coadd, or JINC observation parent role;
- exact parent package/revision/product/application generation and lineage;
- strict linearity with `c_Theta = 0`;
- a fixed same-grid finite operator and fixed convolution construction;
- a qualified fixed-low-pass-convolution subtype;
- exact WCS/frame/topology/grid/metric/shape/row-domain identity;
- sampled kernel/coefficient identity, units, pixel-area factors, center,
  extent/tie, phase, orientation, support, normalization, and provenance;
- full-footprint-only output admission and typed unavailable edge rows;
- transformed signal, output units, originating nominal-beam identity, local
  transfer, transformed response, null/mode state, influence, support,
  FLT-local validity, covariance state, causes, lifecycle, and failure;
- deterministic transformation of an available exact parent covariance;
- exact fixed-state application to compatible NOI realizations; and
- separate observation, coadd, and JINC product identities with no assumed
  filter/coadd commutation.

## Explicitly Deferred Or Excluded

- affine offsets, template/background subtraction, and additive correction;
- reprojection, resampling, mosaicking, and deconvolution;
- fixed boundary extension, periodic wrapping, truncated-unrenormalized
  convolution, support renormalization, inpainting, and edge completion;
- data-derived kernel/cutoff/support/normalization or automatic selection;
- Wiener signal/noise-model or prior-based transformation;
- matched/generalized-least-squares template-amplitude estimation;
- source-learned templates, positions, masks, morphologies, or subtraction;
- data-derived spectral/mode selection and map-domain destriping;
- per-member operator re-resolution or relearning;
- SCI-FLT coaddition;
- RTC temporal/timestream filtering;
- source-fit, Beammap, Pointing, OOF, catalog, or FRUIT use policy; and
- SCI-NOI ensemble design or empirical uncertainty inference.

## Ownership

- MAP/JINC own the parent estimand and parent facts.
- CAL owns absolute calibration, passband/color correction, and calibration
  covariance.
- SCI-FLT-FIXED owns the exact local transformation, output product, response,
  support/validity, deterministic covariance state, lifecycle, and failure.
- SCI-NOI owns realization ensembles and empirical uncertainty and applies the
  exact FLT transformation to compatible members.
- SCI-BEAM and future source/mode contracts own physical source/beam and
  Pointing/OOF interpretations.
- SCI-FRUIT owns iterative feedback science.
- FLT owns FLT-use policy; VAL binds/evaluates but does not author it.

## Parent And Numerical Availability

MAP observation and MAP coadd roles are distinct. JINC is observation-only.
An unavailable MAP or JINC numerical parent remains unavailable; this contract
does not manufacture one from algebra. Frozen MAP/JINC response/covariance
absence remains honest absence in the transformed product.

## Required Stage B Deliverable

After exact-byte owner approval and explicit launch, one fresh
implementation-blind author may produce the SCI-FLT-FIXED normative core,
scientist rationale, engineering-conformance view, requirements, predictions,
traceability, and draft PDFs from only the exact manifest objects. The author
must return one precise question and stop if the packet is insufficient.

## Current Gate And Nonclaims

The Stage A decisions are complete, but the repaired bytes and hashes have not
yet been owner-approved. Stage B has not been launched. This brief selects no
numerical kernel, cutoff, WCS, or implementation and establishes no algorithm
change, conformity, validation, calibration, response/covariance fidelity,
performance, readiness, freeze, production, or Unity authority.
