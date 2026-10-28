# Configurable POINT through the project interface

The `citlali-point-v1` pilot connects TolTECA's prepared, merged YAML to the
existing native RTC → CAL → PTC → MAP components. It processes one configured
POINT observation on a1100, a1400 and a2000. The existing observation/APT and KMP
paths remain project data. TolTECA continues to prepare telescope data, select
calibration items, merge numbered files and launch the selected executable.

This is a local development candidate. It is not the first collaborator release:
pointing-derived flux calibration, SCIENCE/coadds, corrected-frame feedback and
existing downstream image analysis still need their own demonstrated connections.

## Configuration

Place the contents of `data/configured_point_v1.yaml` inside the existing
Citlali step's `config.low_level` block in a new numbered override. The explicit
`migration: replace-legacy-point-controls-v1` selects new processing semantics.
Old templates and saved reductions are preserved; old algorithm switches do not
silently claim to control the new methods. The received YAML, replacement
explanations and separately generated effective YAML are retained with each run.

| Setting | Meaning and current domain |
| --- | --- |
| Ordinary `inputs`, APT calibration items, `kids.solver.fitreportdir` | Actual observation, exact matched APT and associated KMP reports; paths resolve from the launch directory |
| `runtime.n_threads` | Concurrent native network workers, integer1–12; numerical libraries run one thread per worker |
| `runtime.verbose` | False by default; optional reporting/covariance detail only |
| `successor.cleaning.method` | `observed-als`; the end-to-end MAP consumer does not yet admit all PTC-only alternatives |
| `successor.cleaning.ranks` | A positive integer for each array; these reach native PTC unchanged |
| `successor.rtc.profile` | `existing-array-lowpass-v1`: existing measured coefficients/factors1,1,2 and original cadence/speed checks |
| `successor.rtc.disturbance_exclusion` | `disabled` by default, or explicitly selected `repeated-level-shifts-unassessable-remainder-v1`; observation-local paired exclusion under the owner-approved rule |
| `successor.map` | `containing-pixel`, `delta-source-AltAz`, positive pixel size in arcseconds, coverage fraction0–1 |
| `successor.feedback.enabled` | False; true fails clearly because native feedback on corrected AltAz is not validated |
| `successor.products.fit_diagnostics` | Qualified POINT-EXPECTED fits, default true; scientific rejection/unavailability remains explicit |
| `successor.products.retain_intermediates` | False by default; true keeps large saved-stage binary payloads for later diagnostics/replay |

The v1 processing partition retains the existing forced10-second/context32
owner. Its legacy-shaped fields are an internal one-way adapter, not additional
user controls. The old `standard_pca.n_eig_to_cut` carrier holds the new ALS
ranks, with an explicit separate method. New map pixels belong under
`successor.map.pixel_size_arcsec`; the older TolTECA `image_frame_params` and
legacy `mapmaking` fields are retired by this profile. Polarimetry, beammap and
OOF are outside this POINT profile. Unknown new-profile controls are rejected.

## Execution and preservation

Grant triggers all reductions from the shell. Codex prepares the exact command
and configuration, then waits for Grant's completion notice. Implementation,
review and successful engineering checks do not authorize an automatic run.


`package_configured_point.py` freezes a clean source checkout, matching native
binary and required Python/data files into a new runtime directory. Its launcher
pins the runtime manifest and verifies the bound files. TolProj can snapshot
that launcher using its existing executable mechanism; the copied launcher
continues to resolve the exact frozen runtime. It does not depend on its copied
filename or on a mutable checkout. The Python interpreter/environment remains an
explicit local dependency, so this is not a portable Unity release package.

The launcher supports TolTECA's `--version`, `--dump_config`, `-l LEVEL config`
and native canonical-APT protocols. Select it through the existing TolProj
executable setting; no sibling package is modified. A normal call claims a new
`reduNN` directory. Inputs and native outputs are built beneath that new run.
No previous reduction is reused or removed. Maps become discoverable only after
all requested arrays and products have passed publication checks. Required
processing or output errors return a nonzero process status.

Quiet mode avoids process sampling, exhaustive storage traces, BASE feedback
preparation and Neighborhood construction. Saved native transports still exist
while their consumers need them. On successful publication, only this run's
known binary transport extensions are removed when retention is disabled;
receipts and input identities remain and explicitly record that replay payloads
were not retained. This is not a performance or sensitivity improvement claim.

## Products and compatibility

With disturbance exclusion selected, every reduction root contains required
`detector-drops.json`, including on upstream failure. It reports total/per-array
newly dropped, prior empty, evaluated and retained populations, and native
detector identities/reasons. Missing evaluation is unavailable, never zero.
The console summary is printed with verbosity off. Final map-contributor counts
are explicitly not measured by this RTC diagnostic. Native network receipts
bind the initial decision and final application; map publication still requires
all requested arrays. The rule and one-application boundary are recorded in
`COMPROMISED_DETECTOR_POLICY_2026-10-28.md`. The current natural-event path has
no admitted donor repairs; an unexpected donor-bearing exclusion fails safely
rather than reusing contaminated donor estimates.

| Product | Meaning | Current consumer status |
| --- | --- | --- |
| One FITS per array under `reduNN/<obs>/raw` | Native signal, NUMERATOR, NORMALIZATION, HITS, NORM_SUPPORT, SCI_SUPPORT; original bytes/frame/units retained | Discovered by TolTECA; not equivalent to its legacy image-analysis FITS layout |
| Qualified fit JSON per array, when requested | POINT-EXPECTED result, including rejected/unavailable values and conditional uncertainty qualifications | Inspection only; no conventional `ppt_*.ecsv` is issued |
| `RUN.json`, received/effective config, compact input bindings and native receipts | Actual executable, settings, inputs, stage outcomes and retention | Local traceability and diagnosis |
| Optional C0 details and saved native payloads | Requested before a run through verbosity/retention controls | Inspection/replay; not required to interpret ordinary completion |

Native map normalization is not inverse variance. Nominal-beam calibration is
not an empirically measured response. Fit sigma_A is conditional on the initial
covariance model; it is not silently exported as the historical calibration
consumer's amplitude error. The accepted centroid is not a validated pointing
correction. Legacy TolTECA image analysis expects different HDU positions and
units and must not consume these files as interchangeable legacy images.

The next integration milestone must establish a scientifically truthful POINT
calibration table, follow its calibration into the exact SCIENCE APT, and
produce all-array SCIENCE maps plus a genuine multi-observation celestial coadd.
The previously planned reference/new-SCIENCE/new-POINT-calibration comparisons
remain the route to judging improvement. No thresholds are invented here.
