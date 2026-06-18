# Citlali Handoff - Source-Aware Pointing Maps

## 2026-06-18

### Context

- Branch: `gw_dev`
- Goal: bring the recent source-aware beammap lessons into pointing reductions
  without imposing a Gaussian PSF model on focus or out-of-focus holography
  products.
- Main decision: keep one pointing pipeline with a configurable source strategy.
  Standard compact-source pointing can use Gaussian fits as source-location
  diagnostics. PSF-preserving OOF/holography should use empirical fruitloops
  templates and source support only.

### Files Changed

- `data/config.yaml`
- `include/citlali/core/engine/engine.h`
- `include/citlali/core/engine/pointing.h`
- `include/citlali/core/timestream/timestream.h`

### New Pointing Source Strategy

New optional config block:

```yaml
pointing:
  source_strategy:
    mode: standard
    # fit_gaussian: false
    # fruitloops_center_mode: map_center
    # header_max_radius_arcsec: 0.0
    # header_require_coverage: true
```

Modes:

- `standard`: compact-source pointing. Defaults to Gaussian fitting enabled and
  `fruitloops_center_mode=auto`.
- `psf_preserve`: focus and out-of-focus holography style maps. Defaults to
  Gaussian fitting disabled and `fruitloops_center_mode=map_center`.

Fruitloops center modes:

- `auto`: use previous `POINTING.x_t/y_t` when the header center is valid and
  passes guardrails; otherwise fall back to the previous-map peak.
- `header`: use only valid previous `POINTING` header centers.
- `peak`: use the previous-map positive peak.
- `map_center`: use the map center for every map.

### Header-Center Guardrails

Previous-iteration `POINTING` centers are now rejected before seeding fruitloops
source masks/support when they are:

- missing or invalid;
- non-finite;
- outside the map bounds;
- outside `header_max_radius_arcsec` when that guard is positive;
- off positive-weight map coverage when `header_require_coverage=true`.

For standard pointing, `header_max_radius_arcsec` defaults to the configured
source-fitting radius. Setting it to `0.0` disables the radius guard.

### Output And Diagnostics

Pointing products now write additional metadata:

- `POINTING.fit_enabled`
- `POINTING.fit_valid`
- `POINTING.source_strategy`
- `POINTING.source_center_mode`
- `CONFIG.POINTING.*` and `CONFIG.FRUITLOOPS.*` strategy/guard keys in the
  primary header.

Fruitloops map loading logs aggregate source-center counts and per-map
provenance:

- `header`
- `peak`
- `map_center`
- `header_rejected_<reason>`
- `none`

Per-map provenance is logged at info level for up to 16 maps, otherwise debug
level.

### Intended Unity Test Matrix

When Unity is back:

1. Run a standard compact-source pointing reduction with fruitloops enabled for
   at least two iterations.
   - Expect first measurement iteration to report `mode=auto`.
   - Expect valid previous Gaussian fits to provide `header` centers.
   - Inspect any `header_rejected_*` provenance.
   - Compare boresight offsets and map morphology against the current baseline.

2. Run a focus or OOF holography reduction with:

   ```yaml
   pointing:
     source_strategy:
       mode: psf_preserve
   ```

   - Expect `fit_gaussian=false`.
   - Expect `fruitloops_center_mode=map_center`.
   - Confirm no Gaussian-derived `POINTING` center drives fruitloops support.
   - Inspect PSF fidelity and residual source bias across iterations.

3. Stress one messy OOF case with explicit alternatives if needed:
   - `fruitloops_center_mode: peak` to show why it is risky.
   - `header_require_coverage: false` only if valid centers are being rejected
     because the coverage mask is too strict.
   - Adjust fruitloops mask/support radii separately if map-center support is
     too small for the OOF structure.

### Verification Done Locally

- `git diff --check`: passed.
- `cmake --build build --target citlali -j 4`: passed.
- `ctest --test-dir build --output-on-failure`: ran, but the local `build`
  tree has no registered tests.

### Open Questions

- The current PSF-preserving default protects source support by map center, but
  the best default mask/support radii for very defocused maps still need Unity
  validation.
- Downstream consumers should treat `POINTING.fit_enabled=0` and
  `POINTING.fit_valid=0` as "no measured pointing fit", not a measured zero
  offset.
- If standard pointing finds many `header_rejected_radius` cases, check whether
  the source-fitting radius is too tight for real lissajous pointing maps.
