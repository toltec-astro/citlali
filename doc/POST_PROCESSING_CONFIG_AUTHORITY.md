# Post-Processing Config Authority

This document fixes the bounded Phase 2 contract for map filtering, source
finding, and source fitting. It changes no numerical behavior.

## Frozen Surface

The machine-readable manifest is
`tools/config/post_processing_legacy_paths.json`. It freezes 35 leaves across
two historical prefixes:

- 24 `post_processing.*` leaves;
- 11 `wiener_filter.*` leaves.

The top-level Wiener section is part of this domain because it controls filter
template construction and convergence. Leaving it outside the boundary would
make the typed map-filter snapshot incomplete.

Current families are 11 map-filter activation/edge controls, one histogram
control, four source-finding controls, eight source-fitting controls, and 11
legacy Wiener controls.

## Characterized Starting State

The current boundary is intentionally recorded as mixed, not accepted as the
target architecture:

- `WienerFilter::get_config` parses 21 filtering leaves before copying most of
  them backward into `PostProcessingConfig`.
- Existing direct or mirrored readers populate 13 other typed leaves.
- The initial typed request omitted `post_processing.source_fitting.model` and
  `wiener_filter.kernel_template_tail_mode`; both are now represented by closed
  enums in the complete 35-leaf direct request reader.
- Mapmaking-disabled policy mutates the typed request after parsing.
- Source-finding settings are copied from the observation map buffer to the
  coadd map buffer.

These facts are migration inputs. They are not endorsements of reverse mirrors
or mutable requested configuration.

The direct request reader is currently a verified, behavior-neutral boundary:
it is not yet wired into `Engine` execution. Focused tests prove complete
default parsing, disabled-value preservation, and invalid-enum diagnostics.
The next checkpoint will compare this request against the existing parser as a
read-only shadow before authority changes.

## Target Contract

The domain will follow one direction:

```text
merged YAML -> immutable typed request -> effective post-processing plan
                                      -> one-way numerical adapters
                                      -> realized product record
```

The request must preserve all 35 supported leaves, including values belonging
to disabled sections. Effective state resolves mapmaking, reduction mode,
coaddition, noise-map requirements, template availability, and requested
activation without changing the request. The Wiener filter, map fitter, and map
buffers may remain mature numerical targets, but they must not populate or
override typed policy.

Required provenance will distinguish requested filtering/finding/fitting,
effective activation and resolution reasons, and realized observation/coadd
filtering, source-table, and fit cardinality.

## Validation

Fast validation uses matched OG/refactor pointing overlays with identical
merged low-level configuration. Cases should independently cover filtering
disabled, filtering enabled, and source finding/fitting activation. A science
gate is required if migration changes coadd-specific filtering or output
routing. Numerical algorithms and accepted tolerances are unchanged.
