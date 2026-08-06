# SCI-ALIGN-001 split-direction Beammap diagnostic

This package documents an owner-authorized, diagnostic-only Citlali extension
for making ordinary Beammap products from one selected scan direction. It does
not alter timestamps, raw inputs, calibration, RTC/PTC processing, detector
eligibility, detector weights, pointing interpolation, map geometry, or the
production default.

## Configuration contract

Exactly one `beammap.direction_mode` value is accepted per execution:

| Value | Mapmaking scan support | Product suffix |
| --- | --- | --- |
| `standard` | Existing Citlali behavior; no direction classification | none |
| `left` | Accepted raster science legs with negative fast-axis rate | `_left` |
| `right` | Accepted raster science legs with positive fast-axis rate | `_right` |
| `all` | Union of the classified left and right science legs | `_all` |

The key is optional and defaults to `standard`. The four files under
`config_overlays/` are small overlays, not standalone Citlali configurations.
Merge exactly one into the normal Beammap request.

## Direction authority and placement

For nonstandard modes, Citlali uses its current observation's authoritative
raster science windows. These windows are the accepted `Hold == 0`, inside-map
legs already used by the normal pipeline; turnarounds and outside-map samples
do not enter them. For each window, the telescope trajectory is projected on
the fast axis defined by `Header.Map.ScanAngle`. A finite least-squares rate
and endpoint displacement must be nonzero and agree in sign. Telescope time
must be strictly increasing. Any ambiguous leg fails the execution rather than
being guessed.

The selector is evaluated after the common calibration, RTC, PTC, filtering,
cleaning, and weight state has been built. Whole processed PTC scan products
are admitted or skipped only at the Beammap accumulation boundary, uniformly
for every detector. This is the smallest late-stage implementation compatible
with the existing one-PTC-per-scan architecture.

The `standard` mode bypasses classification completely. The other modes emit
a deterministic registry in the observation's `raw/` directory:

```
beammap_direction_scan_registry_left.csv
beammap_direction_scan_registry_right.csv
beammap_direction_scan_registry_all.csv
```

Each row records the science window, time support, projected displacement,
signed rate, direction, and selection decision.

## Products

Map, noise-map, filtered-map, Beammap APT, and fit-QC basenames carry the mode
suffix for nonstandard modes. Directory structure and file formats are the
ordinary Citlali Beammap structure. FITS primary headers and APT metadata also
record `beammap.direction_mode`. Because the Unity campaign uses an isolated
output root per mode, all ancillary products and logs remain unambiguous even
when their legacy basename has no mode suffix.

`standard` and `all` are intentionally distinct. `standard` is the untouched
default path. `all` is a diagnostic assertion that every current raster
science window can be classified and that the union of left and right legs is
used.

## Scope boundary

This change creates the rich map/APT inputs for the later comparison. It does
not implement centroid comparison, expectation manifests, null maps,
bootstraps, difference maps, overlays, timing conversion, or mitigation. No
result from these products alone authorizes a timestamp correction or a claim
of physical clock failure.

See `UNITY_RUNBOOK.md` for the owner-run 150819-first campaign and
`RETURN_BUNDLE_SPEC.md` for the requested return evidence.
