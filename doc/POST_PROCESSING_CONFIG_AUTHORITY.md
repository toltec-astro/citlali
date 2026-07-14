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

The characterized starting boundary was intentionally recorded as mixed, not
accepted as the target architecture:

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

The direct request reader first ran as a fail-fast, read-only shadow during
`Engine` config loading. Focused tests prove complete default parsing,
disabled-value preservation, invalid-enum diagnostics, active-field parity,
and useful mismatch diagnostics. The shadow compares activation and histogram
unconditionally, but compares detail fields only where the legacy path
actually loads them. This prevents disabled legacy defaults from masquerading
as requested-value mismatches.

The first target-contract checkpoint now constructs a separate
`PostProcessingExecutionPlan` from that request. The plan preserves every
requested value, resolves mapmaking-dependent filtering and finding into an
effective snapshot, records why activation changed, derives fitting need from
reduction type and requested downstream work, and clears realized state for a
new reduction. At that checkpoint it was intentionally parallel to the
existing execution boundary: no production consumer read the effective
snapshot, and no numerical implementation changed.

The map-filtering consumer cutover is now complete locally. Both duplicated
serial/OpenMP `WienerFilter::get_config` YAML parsers and the reverse
Wiener-to-typed mirror are retired. One adapter copies the effective typed map-
filter snapshot into either numerical implementation, including the legacy
conditional FWHM loading and arcsecond-to-radian conversion. Filter activation,
required-output policy, runtime dependency checks, and map-diagnostic edge-
guard metadata consume the effective plan. The mature filtering algorithms are
unchanged. The remaining mixed boundary is source finding and source fitting.

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

Consumer migration is ordered map filtering, source finding, then source
fitting. Map filtering is complete and accepted by Unity point `redu54`.
Source finding is accepted exactly by Unity point `redu55`; source fitting
remains on the mixed boundary. Each slice
replaces a reverse mirror or policy read with one one-way typed adapter while
keeping the mature numerical object as the execution target. A consumer
cutover, unlike plan construction alone, requires the matched enabled-filtering
point gate, with accepted `redu55` now serving as the immediate baseline.

The source-finding adapter projects the effective threshold, angular window,
and finder mode directly into observation and optional coadd map buffers. It
does not copy coadd policy from realized observation state. Source detection,
Gaussian fitting, and table production remain mature numerical consumers.

## Validation

Fast validation uses matched OG/refactor pointing overlays with identical
merged low-level configuration. Cases should independently cover filtering
disabled, filtering enabled, and source finding/fitting activation. A science
gate is required if migration changes coadd-specific filtering or output
routing. Numerical algorithms and accepted tolerances are unchanged.
