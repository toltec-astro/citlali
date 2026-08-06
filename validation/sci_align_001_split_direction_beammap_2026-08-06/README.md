# SCI-ALIGN-001 split-direction Beammap diagnostic

This package documents an owner-authorized, diagnostic-only Citlali extension
that can emit standard, left-going, and right-going detector Beammap products
from one reduction. It does not alter timestamps, raw inputs, calibration,
RTC/PTC processing, detector eligibility, detector weights, pointing
interpolation, map geometry, or the production default.

## Configuration contract

Exactly one `beammap.direction_mode` value is accepted:

| Value | Mapmaking scan support | Product suffix |
| --- | --- | --- |
| `standard` | Existing Citlali behavior; no direction classification | none |
| `left` | Raster science legs with negative fast-axis rate | `_left` |
| `right` | Raster science legs with positive fast-axis rate | `_right` |
| `all` | Standard plus left and right products in one reduction | none, `_left`, `_right` |

The key defaults to `standard`. The four files under `config_overlays/` are
small overlays, not standalone configurations. `all` requires
`mapmaking.grouping: detector`, because it promises detector maps and APTs for
all three products.

## Direction authority and one-pass placement

Citlali derives direction from the current observation rather than importing a
prior SCI-ALIGN registry. It uses the authoritative raster science windows
already identified by the telescope pipeline. Turnarounds and outside-map
samples are absent from those windows. For each window, the telescope
trajectory is projected onto the fast axis defined by
`Header.Map.ScanAngle`. A finite least-squares rate and endpoint displacement
must be nonzero and agree in sign, and telescope time must be strictly
increasing. Ambiguous legs fail the reduction.

Calibration, RTC, PTC, filtering, cleaning, detector weights, and scan
eligibility run once. After PTC processing, `all` owns three detector-map
buffers. Every processed scan fills the standard buffer and exactly one of the
left or right buffers. Thus the expensive timestream pipeline is not repeated;
the incremental work is a second map accumulation per scan, normalization of
three buffers, and two additional final Beammap fit/QC/output passes. Peak map
storage is approximately three observation map buffers, not three complete
reductions.

The standard fit state is saved before each directional fit. Citlali uses the
existing Beammap fitter, flagging, reference subtraction, derotation,
calibration, APT, fit-QC, and FITS writers for each side, then restores the
standard state through an exception-safe transaction. A dedicated test mutates
the complete retained product-state surface, injects a failure, and verifies
exact restoration.

`standard` bypasses direction classification. `left` and `right` write one
matching registry. `all` writes one complete registry:

```
beammap_direction_scan_registry_left.csv
beammap_direction_scan_registry_right.csv
beammap_direction_scan_registry_all.csv
```

## Products

All products retain the ordinary Citlali reduction-directory structure.
Standard map, noise-map, filtered-map, APT, and fit-QC basenames are unchanged.
Directional siblings in the same `raw/` or `filtered/` directory carry
`_left` or `_right`. FITS primary headers and APT metadata record the realized
`standard`, `left`, or `right` identity.

The Citlali change creates map/APT inputs for later comparison; it does not
alter timestamps or implement a mitigation. A companion read-only diagnostic,
`tools/diagnostics/render_sci_align_001_split_direction_beammaps.py`, consumes
one completed `all` reduction and creates the first visual review product. It:

- selects up to 100 detectors from a requested array using only standard-APT
  quality and S/N (or a supplied, pre-existing UID list), never a measured
  left/right displacement;
- renders the standard, left, and right maps on their common absolute WCS,
  without recentering, with fitted centroids and the positive scan direction;
- renders common-coordinate contours, a left-minus-right map, and an
  along-scan profile for each detector;
- reports right-minus-left parallel and perpendicular fitted-centroid
  separations and a fit-derived timing-equivalent diagnostic; and
- writes a multipage PDF, ECSV selection and metric tables, a hash-bound input
  manifest, and output checksums.

The default is one detector per page. The only other supported layout is two
detectors per page; the program rejects larger values. The fit-derived timing
uncertainty uses the diagonal left/right centroid-fit errors only and is not a
claim that map pixels or detectors are independent. These diagnostic products
do not authorize a timestamp correction or a claim of physical clock failure.

See `UNITY_RUNBOOK.md` for the owner-run 150819-first campaign and
`RETURN_BUNDLE_SPEC.md` for return evidence.
