# Independent local raw/readiness evidence review — 2026-09-26

Scope: read-only local evidence. No Unity connection, download, reduction, science evaluation, data mutation or deletion. This is a bounded supplemental review, not an independent full acquisition census. The parent's all-netCDF-header census remains the authority for the complete current grouping table.

## Strongest association evidence

`2025-C1-COM-01/docs/ngc4449_pointing_analysis/04_tune_health/README.md` and `observation_tune_apt_lineage.csv` explicitly establish eight POINT observations: **152389, 152391, 152393, 152418, 152420, 152430, 152432, 152434**. For each, networks **0–5, 7–9, 11–12** are active. The saved evidence says 6 and 10 are absent throughout the acquisition sequence, not local network dropouts. Each has its own same-observation subobs0/scan1 tune followed by subobs0/scan2 POINT; association is from Header.Toltec ObsNum/ScanNum/RoachIndex and named reduction APT, not numerical proximity.

Fresh path checks of all88 lineage rows found every recorded raw tune, processed tune, fit report, pointing raw, APT and audit path present: **528 successful path checks**. These are availability checks, not fresh payload hashes or sample-quality tests. The reduction config is `2025-C1-COM-01/pointings-refactor/72_pointing_observation.yaml`; APTs are `2025-C1-COM-01/apts/hero/apt_<obs>_matched.ecsv`, with associated hero-transfer audits. All eight telescope/HWPR/configured inputs are also represented locally (see config dependency inventory).

Important distinction: those tune files themselves refer to earlier target-sweep identities. The saved lineage report identifies33 network target references at152388,152414,152426 missing from **that project's data/**. This does not invalidate the own-observation tune files. A whole-root grouping may resolve152388 in `2025-ENG-pointing-152388-152389/data`; do not publish the project's local absence as whole-tree absence without checking headers.

## Independent configured-input inventory

`/private/tmp/local-readiness-configs.json` records27 configured observation slots from10 existing generated Citlali YAML files, with exact recorded remote paths, interface identities and all local basename candidates/bytes. `/private/tmp/review_local_readiness.py` generated it from the already saved stat inventory and compact YAML only. No full payload read.

26 slots have all configured data/APT filenames somewhere locally: POINT152389; OOF152385/6/7; beammap148670 with POINT support148669/148671; POINT152388; the eight POINTs above; SCIENCE152390/392/419/431/433; beammap152307; POINT123424. The27th is a useful negative control: a two-observation historical config includes129081, whose12 detector raw files are not local even though telescope/APT products remain. This is an incomplete input package, not a reduction-ready raw observation.

The expected network set here is **the saved reduction's explicit input inventory**, not a newly obtained observatory acquisition manifest. Say "complete against the saved reduction input inventory" rather than making an unrestricted archive-completeness claim. Local basename matching is a discovery link; the parent's header inventory must resolve duplicates/conflicts and exact observation/network identities.

Generated config evidence:

- `citlali-validation/v1/point/pointings/reduced/citlali_o152389_0_2_c1.yaml`
- `citlali-validation/v1/oof/wilson/1146+399/reduced/citlali_o152385_0_1_c3.yaml`
- `citlali-validation/v1/beammaps/3c273/reduced/citlali_o148670_0_2_c1.yaml`
- `citlali-validation/v1/beammaps/pointings/reduced/citlali_o148669_0_2_c2.yaml`
- `citlali-validation/v1/science/reduction/NGC4449/reduced/citlali_o152390_0_2_c2.yaml`
- `2025-ENG-pointing-152388-152389/pointings/reduced/citlali_o152388_0_2_c2.yaml`
- `2025-C1-COM-01/pointings-refactor/reduced/citlali_o152389_0_2_c8.yaml`
- `2025-C1-COM-01/NGC4449/reduced/citlali_o152390_0_2_c5.yaml`
- `beammaps/3c273/152307_redu50/citlali_o152307_0_2_c1.yaml`
- `beammaps/pointings/reduced/redu01/citlali_o123424_0_2_c2.yaml`

## Tune dependencies and readiness cautions

Three narrow fresh raw-header checks independently found:

- OOF152385/network0/sub0/scan1 → TargSweep152384/sub0/scan1. The local v1/oof data directory contains raw/processed tune152384 + text fit reports; use the all-network header census to establish each of152385–7 rather than infer sharing from proximity.
- SCIENCE152390/network0/sub0/scan2 → TargSweep152390/sub0/scan1.
- POINT123424/network0/sub0/scan2 → TargSweep123424/sub0/scan1. It has12 raw detector files (~0.359GB) and12 processed tune files (25,235,656bytes) in `fruit-development/point-123424/input/fitreports`. Text fit reports are present elsewhere, `tone-match-lab/c2025t_tune_text/beammaps/data` (12 files,1,220,172bytes). Do not label the text-fit reports globally missing merely because they are absent from the `fitreports` directory. **Raw tune acquisition remains unverified** until the all-header inventory confirms it.

Do not use `project.yaml` status bits alone as availability truth. v1/point says raw_copied=false/tunes_done=false despite present files and established reductions. The OOF reduce-tunes log first lists own152385/6/7 tunes missing, then marks tunes_done=true; the raw header correctly associates152384.

Other observed detector-stream packages, needing header/config reconciliation before completeness labels: POINT152398/403/408/413/425 all11 local files in2025-C1-COM-01/data. The tone-match/VNA labs contain many targeted subsets; targeted tune or VNA subsets should not be described as complete sky-observation packages just because each opens.

## Archive provenance

The generated configs preserve exact historically used Unity/work paths; for the v1 validation suite these begin `/work/toltec/citlali-validation/v1/<project>/data`, and APTs `/work/toltec/citlali-validation/v1/<project>/apts`. Other configs retain `/work/toltec/commissioning2025-test/...` or successor project paths. These are **locally preserved historical locations, not live verified remote availability**. No remote command was executed.

TolProj's portable suite at `/Users/gwilson/GitHub/tolproj/tolproj/templates/validation_suites/citlali_refactor_validation_v1/suite.yaml` identifies the intended four-mode validation package but is not itself a complete raw acquisition manifest. The local VNA inventory summaries preserve historical `/nese/toltec/{02_data_lmt,03_data_lmt,data_lmt}/toltec/ics` locations; those are a different tune/VNA evidence collection, not fresh remote confirmation of the sky packages.

A future owner-run narrow retrieval can use `rsync --dry-run --protect-args unity_toltec:<exact saved parent/path> <chosen local destination>` after the owner verifies current source location. Such a command is a proposal assembled from saved evidence, not a previously executed retrieval recipe. No bulk transfer authorized here.

## Possible small resident corpus

Prioritize measured complete-input packages, not the earlier proposed list. A compact possible tier is POINT152389 plus the three OOF152385–7 sharing tune152384, followed by **one** NGC4449 science observation152390 with its bracketing POINT152389/152391 if the owner wants long scans/extended-source coverage. The raw+all tune/support directories currently measure roughly0.53GB for the POINT and1.25GB for OOF; science152390 detector streams alone6.76GB (total including its tune/telescope and two POINT packages is about8GB). Derive final exact combined totals from the header census and count shared dependencies once. Do not double-count POINT152389 between the compact tier and science support.

Beammap148670 adds a much larger~16.24GB detector stream plus its tune/support pointings. It is a useful optional four-mode coverage tier, not a necessary daily resident set. POINT123424 is scientifically different historical network coverage and already useful, but a raw-tune gap should be explicitly classified if confirmed. Nothing here selects or launches a new observation experiment; no raw package was moved/deleted.

## Review limitations / required final integration

I have not received the finished NETCDF_HEADERS census yet. Final readiness labels must combine that census with the sources above, including all-network tune associations, time/structure checks, acquisition-support streams and duplicate/conflict accounting. It is acceptable to leave true archival completeness unverified where local saved reduction inventories are the only expectation evidence. Report that limitation plainly; do not substitute a fixed13-network rule.

## Final header-census integration

The closed census contains21,482 NetCDF header records. I independently aggregated every file containing acquisition `Data.Toltec.Is/Qs`, `Header.Toltec.ObsType==0` and TolTEC identity metadata: **27 distinct (obsnum,subobsnum,scannum) sky-stream groupings**. Results are `/private/tmp/local-readiness-header-review.json`, generated by `/private/tmp/check_readiness_headers.py`. The groupings are123424;148669–148671;152307;152385–152393;152398;152403;152408;152413;152418–152420;152425;152430–152434. Allsubobs0; OOF152385–7 scan1, othersscan2.

-26 groupings have11 networks(0–5,7–9,11–12);123424 has12(0–9,11–12). Network counts come from actual identities, not fixed expectations. Existing configured-input evidence overlaps22 of27 distinct observed groups; the additional five2025-C1-COM-01 POINTs and123424 limitations remain separately stated.
- Every network within each grouping records the same associated TargSweep identity. OOF152385–7 all point to152384/sub0/scan1; every other sky grouping points to its own obsnum/sub0/scan1.26 groupings have all exactly associated raw-tune network files somewhere locally. **123424 lacks all12 associated raw tune files**, though its12 processed sweeps and12 text fit reports are local.
- **123424 also lacks original telescope acquisition**, with only `tel_toltec_2024-11-27_123424_00_0002_recomputed.nc` copies present. It must not be called a complete raw package. Its established reduction inputs can be available while its original acquisition package is incomplete.
- All other26 sky groupings have original telescope acquisition with matching identity. Every inspected network's raw receive-time endpoints lies inside an original telescope-time endpoint range. Network endpoint spreads are<0.015s. These are structural endpoint checks only; no internal sample-drop or scientific-quality assertion is made. `/private/tmp/local-readiness-time-review.json` preserves these results.
- Every group has internally uniform recorded `Header.Toltec.SelectedMask`. Low13 bit sets coincide with observed network IDs (7103/15295 for11-net groups,7167 for123424's12). This is corroboration only: I found no local authoritative decoding specification for bit13 and do not elevate it to a definitive total acquisition inventory. Saved explicit reduction input sets and the eight-POINT lineage remain stronger expectation evidence.

The strongest currently usable complete input packages are therefore the configured26 slots/22 distinctobs except123424 (21 distinctobservedskyobs); the eight validated POINTs,3OOF,5science,3Januarybeammap/support andFebruarybeammap/POINT152388 provide substantial choice. Parent should use its exact observation table to enumerate without relying on these summary arithmetic counts. Newly discovered152398/403/408/413/425 have raw detector+original telescope+associated raw tunes, but APT/calibration readiness and independent expected inventory are not established by this review. No artifact completeness claim is extended from stored derived telescope products alone.
