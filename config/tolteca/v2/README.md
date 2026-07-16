# Science Authoring Prototype V2

This is a science-only prototype for review. TolPROJ does not install it yet,
and it has not replaced the accepted four-mode V1 kit.

The prototype remains ordinary TolTECA YAML. `tolteca reduce` merges the files
directly in numeric order; no translator or generated intermediate step is
required before a reduction.

## Files And Audiences

| File | Audience | Purpose |
| --- | --- | --- |
| `60_science_internal_policy.yaml` | Citlali maintainers | Complete accepted policy. Generated, hash-checked, and not normally edited. |
| `71_science_runtime.yaml` | Site operator | Executable, input/output paths, thread count, output layout, and verbosity. |
| `72_science_observation.yaml` | TolPROJ | Observation selection, APTs, calibrator fluxes, and pointing support. |
| `81_science_defaults.yaml` | Reducer | Routine mapmaking, calibration, cleaning, weighting, and iteration choices. |
| `82_science_products.yaml` | Reducer | Coadd, noise, filtering, fitting, and retained TOD product choices. |
| `90_science_advanced_overrides.yaml` | Advanced reducer | Additional supported user-facing controls omitted from the short defaults. Empty by default. |
| `99_science_expert_overrides.yaml` | Citlali expert | Detailed algorithm or diagnostic overrides. Empty by default and requires validation rationale. |

The normal operator surface is 62 low-level leaves: five runtime values, 27
analysis defaults, and 30 product values. Every one is classified
`user-facing`. The complete 404-leaf policy remains available in the clearly
marked internal file, rather than being presented as a normal editing surface.

All routine fruit-loop choices are consolidated in `81_science_defaults.yaml`,
including activation, iteration count, S/N and per-array flux cuts, and whether
to retain every iteration. Product settings in `82_science_products.yaml` are
limited to reduction outputs outside the fruit-loop lifecycle.

The defaults and products files intentionally reassert accepted values already
present in the internal policy. Editing them changes the effective policy by
normal TolTECA precedence. Leaving them untouched reproduces accepted science
`redu31` exactly, with policy SHA-256
`10095418b09100f15c90af173ee34ea7bfcf12260cec41d80f43f6f50473a347`.

## Validation

From the Citlali repository root:

```bash
$HOME/tolteca/bin/python tools/config/tolteca_mode_kit.py validate \
  --mode science \
  --mode-dir config/tolteca/v2/science \
  --manifest config/tolteca/v2/manifest.yaml
```

The config preflight also checks exact policy identity, classification
boundaries, size limits for the normal files, and generator reproducibility.
