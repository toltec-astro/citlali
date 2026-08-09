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
the complete retained product-state surface, injects a failure, mutates nested
YAML metadata, and verifies exact restoration. Product-state calibration
copies deep-clone `apt_meta`; ordinary `YAML::Node` copy semantics otherwise
alias the metadata across the standard, left, and right products.

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

The first Unity `all` run for ObsNum 150819 (job 62656042, commit 9730f0e2)
completed its map writes but exposed the YAML aliasing defect: the final
standard detector-table write retained `right` metadata and overwrote the
right APT, leaving no unsuffixed standard APT or standard fit-QC table. That
output is retained as failed contract evidence and is not valid visualization
input. The follow-up deep-clones metadata for every captured/restored product
and fails closed if the standard detector-table output does not hold restored
`standard` metadata.

The Citlali change creates map/APT inputs for later comparison; it does not
alter timestamps or implement a mitigation. A companion read-only diagnostic,
`tools/diagnostics/render_sci_align_001_split_direction_beammaps.py`, consumes
one completed `all` reduction and creates the first visual review product. It:

- selects up to 100 detectors from a requested array using only standard-APT
  quality and S/N (or a supplied, pre-existing UID list), never a measured
  left/right displacement;
- renders the standard, left, and right maps on their common raw AltAz WCS,
  without recentering, with the matching `x_t_raw`/`y_t_raw` fitted centroids
  and the positive scan direction;
- renders common-coordinate contours, a left-minus-right map, and an
  along-scan profile for each detector;
- reports right-minus-left parallel and perpendicular raw-map-frame
  fitted-centroid separations and a fit-derived timing-equivalent diagnostic;
- writes a multipage PDF, ECSV selection and metric tables, a hash-bound input
  manifest, and output checksums.

The default is one detector per page. The only other supported layout is two
detectors per page; the program rejects larger values. The fit-derived timing
uncertainty uses the diagonal left/right centroid-fit errors only and is not a
claim that map pixels or detectors are independent. These diagnostic products
do not authorize a timestamp correction or a claim of physical clock failure.

The corrected owner-run ObsNum 150819 `all` reduction subsequently completed
in 3h45m and passed the standard/left/right retained-product checks. Its
100-detector a1100 review shows a coherent along-scan left/right displacement.
Because 3C273 has a mostly cross-scan jet, the visualizer's non-mirrored
profile structure is not by itself a clean filter test.

`tools/diagnostics/analyze_sci_align_001_split_direction_transfer.py` provides
that bounded follow-up without another Citlali reduction. It consumes the
same independently fixed UID table and requires the matching retained
`kernel_det_*_I` plane for every selected detector and every product. The
synthetic kernel is generated inside RTC after calibration/extinction and is
then carried through the subsequent RTC filters, PTC cleaning, and mapmaking.
Therefore it directly tests that downstream transfer path, while explicitly
remaining unable to test raw detector-data/timestamp association or operations
that precede kernel generation.

The diagnostic creates equal-detector median signal and kernel stacks in the
raw AltAz map frame. Signal translations are fit in three masks defined only
from the standard stack: nuclear core, significant mostly vertical support
outside the core, and their union. Kernel translations use the nuclear mask.
It also retains every detector's signal and kernel right-minus-left result.
The descriptive decision tolerance is one-half of the diagnostic stack pixel,
with a 0.25-arcsec floor:

- a resolved, morphology-stable signal displacement with a centered kernel
  strongly disfavors a downstream filtering/mapmaking artifact within the
  kernel's scope;
- a resolved signal displacement and a co-moving kernel favors such a
  downstream transfer artifact;
- unresolved, morphology-sensitive, or mixed results are inconclusive.

This is a resolution-bounded diagnostic classification, not an
independent-pixel likelihood or a formal confidence interval. The tool writes
the detector table, stack-registration table, exact decision JSON, stack
arrays, a two-page PDF, a hash-bound manifest, and checksums. It fails before
creating its output directory if any required retained kernel is absent or if
the signal/kernel shapes or WCS identities differ.

The owner-run mapmaker-dependence control cloned the accepted retry2
configuration, changing only `mapmaking.method` to `naive` and the output
root. It completed in 1h43m with standard, left, and right maps/APTs. Its
visual review retains the coherent along-scan displacement while exposing the
expected sparse nearest-pixel support of naive mapmaking.

A follow-up single-pass run disabled fruit loops, selected `standard`, retained
the full processed timestream, and used the same naive detector mapmaking. It
completed its processing iteration in about 11 minutes, then failed required
product finalization because Beammap tried to update the PTC TOD
`FRUITLOOPS_ITER` value before the deferred general TOD header had created the
variable. The repair creates this mutable field with the initial PTC file
schema. The existing scalar NetCDF writer is idempotent, so the later general
header updates the same field rather than duplicating it. A focused lifecycle
test covers schema creation, Beammap's iteration-time update, and the final
auxiliary-metadata pass. The failed partial PTC is not accepted evidence. The
fresh-root Unity retry completed successfully and retained 153,360 samples,
199 scans, 5,110 detectors, the required signal/flag/weight fields, detector
and telescope pointing on the common timebase, and `FRUITLOOPS_ITER = 0`.
The later selected-scan join established that this file's scan metadata is not
valid for variable-length PTC chunks: every row retained the first scan's
606-sample length. The arrays remain defect evidence, but a fresh replay after
the narrow append-bound repair is required before using them as scan-bound
authority. The distinct defect record is
`../../handoff/SCI_ALIGN_001_PTC_SCAN_METADATA_DEFECT_2026-08-08.md`.

This is a confirmed engineering product-lifecycle defect, kept separate from
the left/right timing interpretation. Its exact Unity trigger, root cause,
repair alternatives, regression evidence, and future SCI-BEAM-001 routing are
recorded in
`../../handoff/SCI_ALIGN_001_PTC_ITERATION_METADATA_DEFECT_2026-08-07.md`.

`tools/diagnostics/analyze_sci_align_001_ptc_sampling.py` consumes that full
PTC and the completed standard/left/right naive maps for one detector. It
self-classifies scan direction from the retained telescope trajectory and
replays Citlali's naive nearest-pixel assignment. Ordered, non-overlapping
raster science windows need not be contiguous; turnaround and inter-scan
samples remain explicitly unclassified. The completed UID 199 audit confirms
a continuous 8.192-ms pointing cadence and detector-minus-telescope step
residuals below 0.00011 arcsec. It also shows the expected smooth horizontal
scan tracks. Its pixel-support comparison is not exact, however: the retained
full PTC and directional maps are separate replays, so signal flags, weights,
and final fitted detector pointing are not the map run's accumulation state.
The low cross-run Jaccard values therefore cannot classify individual white
or colored pixels. They do not overturn the trajectory-continuity result.

`tools/diagnostics/analyze_sci_align_001_selected_sampling_join.py` closes
that retained-product gap for the scans saved by the map reduction's
detector-specific TOD. It joins same-run final-iteration signal/flags and
same-run per-scan detector weights to the full-PTC pointing using the explicit
one-based original scan identity. The map run's zero-based direction registry
is converted explicitly and checked against direction independently derived
from the full PTC. The selected TOD retains each complete processed scan
chunk, so the sample slice is the repaired full PTC's exact persisted append
extent. Duplicate uniform/dense retained
slots must be bytewise identical before they are deduplicated. For detector map grouping the tool
reconstructs the actual accumulation coordinates from telescope pointing plus
retained pointing offsets; it deliberately does not use the later
final-APT detector pointing. A selected hit missing from the corresponding
map is exact disagreement. A map pixel without a selected hit is untested,
because any non-retained scan may support it. Outputs are a two-page PDF,
joined-scan and selected-support ECSV tables, compressed hit-count arrays, a
hash-bound manifest, and checksums.

The completed unthresholded full-PTC reconstruction for ObsNum 150819 a1100
UID 199 provides a separate support-threshold control. Its standard hit count,
weight, and weighted-signal accumulations partition exactly into the left and
right products. The right-minus-left parallel displacement is -2.6985 arcsec
(-28.703 ms equivalent) before the final support threshold, compared with
-2.5380 arcsec (-26.996 ms) in the retained Citlali APT. Thus final support
thresholding or retained-pixel selection is strongly disfavored as the primary
cause for this detector, while the exact upstream timing origin and any
universal correction remain unresolved. The checksum-bound review, explicit
limitations, and proposed smallest local follow-up are recorded in
`UNTHRESHOLDED_FULL_PTC_EVIDENCE_2026-08-08.md`.

The approved local-only follow-up compares all stable networks in the three
successful enhanced maps sharing frozen T0-vector group
`roach-t0:44cf69da97d473965ef6`. The map-pair residuals after the fixed
minus-one-slot prediction have unique, transitive modal states on the
4.096-ms half-cadence lattice: 148670 = 0, 150819 = -3, and 151126 = -2.
Native phase changes are only tens of microseconds, every delivered
PPS/PpsTime transition association remains same-row, and anomaly-free network
controls retain the common timing bands. This disfavors native phase, the
measured slot residual alone, and delivered PPS increment anomalies as the
primary explanation within this frozen group. It does not establish the
upstream timestamp event, an FPGA association, a universal state, or a timing
correction. Exact identities, limitations, and the bounded next owner decision
are recorded in `SAME_T0_CADENCE_LATTICE_EVIDENCE_2026-08-08.md`.

See `UNITY_RUNBOOK.md` for the owner-run 150819-first campaign and
`RETURN_BUNDLE_SPEC.md` for return evidence.
