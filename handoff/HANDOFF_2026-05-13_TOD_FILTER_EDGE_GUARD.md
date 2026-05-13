# TOD Filter Edge Guard

## Context

The 3C273 `redu12` beammap enabled the raw time chunk FIR lowpass and static notch filter. Compared with `redu11`, many new `a1100` failures showed bright structure at raster-map edges. A representative case was `a1100 uid 469`: the detector was good in `redu11`, but in `redu12` the left/right map-edge robust scatter rose to several times the map-center scatter.

The failure mode is consistent with filter transients at raster scan boundaries. The FIR convolution already loads an outer context for the symmetric kernel, but Citlali did not have a standard way to mark samples near each scan edge as invalid after filtering. The static notch filter also becomes active whenever `raw_time_chunk.filter.enabled` is true, and its zero-phase IIR pass has no explicit edge guard.

## Changes

- Added a shared `timestream.raw_time_chunk.filter.edge_guard` configuration block.
- Added `RTCProc::configure_filter_edge_guard` to compute both:
  - `context_samples`: samples loaded around each scan to provide filtering support.
  - `guard_samples`: samples flagged at each retained scan edge after filtering.
- Added `RTCProc::apply_filter_edge_guard`, which flags the leading/trailing guarded rows in `in.flags.data` after FIR/notch/IIR filtering and before downsampling or mapmaking.
- Replaced the old RTC assumption that the retained scan always starts at `filter.n_terms`; the retained offset now comes from `scan_indices(0) - scan_indices(2)`, so FIR, notch, and IIR context are interpreted consistently.
- Extended gap-flag dilation in `pointing`, `beammap`, and `lali` paths to use the configured filter context rather than only FIR `n_terms`.
- Wrote edge-guard configuration and per-output-scan guard diagnostics to TOD NetCDF products.
- Added FITS provenance keys for whether the guard is enabled and the realized context/guard sample counts.
- Added unit coverage for notch settling estimates.

## Config

The default config now exposes:

```yaml
timestream:
  raw_time_chunk:
    filter:
      edge_guard:
        enabled: false
        mode: flag
        combine: sum
        min_samples: 0
        extra_samples: 0
        max_samples: 128
        iir_settle_attenuation: 0.01
        apply_fir: true
        apply_notch: true
        apply_dynamic_notch: true
        apply_iir_highpass: true
        apply_downsample: true
```

The default is disabled to avoid changing historical reductions unless explicitly requested. For filtered beammap tests, enable it with `mode: flag`. The `max_samples` cap is intentionally present because narrow IIR notches can have very long formal settling times; the guard should be operationally conservative without consuming whole raster scans.

## Suggested Next Test

Run a `redu13`-style beammap with the same lowpass/notch settings as `redu12`, plus:

```yaml
timestream:
  raw_time_chunk:
    filter:
      edge_guard:
        enabled: yes
        mode: flag
        combine: sum
        max_samples: 128
```

Then compare against `redu11` and `redu12`, focusing on `a1100` edge artifacts, the `a1100` good-detector yield, and known outliers such as uid 469.
