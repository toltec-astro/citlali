# Local-data inventory and RTC/PTC purge — 2026-09-26

Completed the owner's bounded purge across `/Users/gwilson/work_toltec/local_data`.
Removed **221,249 identified inactive Citlali RTC/PTC files, 356.820 GB logical**;
measured available-space gain during deletion was **353.301 GB**. There were no
failed or deferred deletions. Raw data, tunes, calibration inputs, maps, scientific
records and all other original non-target files retain their recorded identities.
CUT3 and the second-observation comparison remain parked.

## Disk accounting

GB below means decimal 10^9 bytes; allocated file blocks and `df` are distinct
accounting measures and do not establish exclusive physical ownership on APFS.

| Measure | Before | After |
|---|---:|---:|
| Regular files | 550,386 | 330,451 |
| Logical file bytes | 970,753,985,023 | 619,128,938,835 |
| Allocated file-block bytes | 971,802,185,728 | 619,891,007,488 |
| Directories | 40,233 | 40,224 |
| Symlinks (not followed) | 20 | 20 |

The net logical reduction is **351.625 GB**. Removing 356.820 GB added 5.194 GB of
verified non-target contents retained from mixed NetCDF containers, plus 0.000668 GB
of additive purge indexes. No discarded signals were copied into those companions.
Nine resulting empty directories were removed. No hard-linked file groups,
unreadable locations or mount/traversal exceptions were found.

Available space immediately before deletion was 290,458,624,000 bytes and immediately
after was **643,759,407,104 bytes (599.548 GiB)**. This deletion-interval gain excludes
space used earlier for companion preservation and durable records. From the initial
inventory's available-space reading to immediately after deletion the net gain was
347,495,469,056 bytes; unrelated filesystem activity can also affect `df`.

| Largest original top-level contributors | Before GB | After GB |
|---|---:|---:|
| citlali-validation | 305.366 | 133.308 |
| 2026-ENG-hero-multiyear-pointings-v1 | 201.940 | 74.328 |
| 2025-C1-COM-01 | 122.171 | 120.140 |
| 2026-refactor | 112.588 | 84.296 |
| beammaps | 45.860 | 41.971 |
| tone-match-lab | 39.686 | 39.686 |
| fruit-development | 34.952 | 20.324 |
| 2025-C1-COM-21 | 22.734 | 22.734 |
| 2025-C1-COM-06 | 18.211 | 18.211 |
| sci_cal_001_tau025_engineering_extension_002_root | 17.194 | 17.194 |

All top-level totals are in `SUMMARY_BEFORE.json` and `SUMMARY_AFTER.json`. The
remaining historical tree is still 619.129 GB; the future 200 GB proposal below does
not claim to bound it or authorize further deletion.

## Products and preservation

The disjoint initial product classification is in `PRODUCT_CLASSES.json`; the
NetCDF subset is refined in `NETCDF_PRODUCT_CLASSES.json`. Major retained classes
are FITS maps (212.739 GB), original sky detector acquisitions (104.119 GB), original
tune/VNA acquisitions (14.510 GB), processed tune/KIDS NetCDF products (2.046 GB),
APT/matching/calibration tables (12.686 GB), and telescope/HWPR/housekeeping including
processed variants (10.296 GB). Figures total 0.596 GB. Text/configuration/measurement
records total 19.570 GB; only selected compact recovery records were additionally
preserved in Git. These class figures describe the original files, not added companions.

**95.401 GB of RTCdiag/PTCdiag scan statistics remain**: they are scan-by-detector
summaries, not saved sample streams. Another 58.639 GB of other NetCDF diagnostic/input
products remains. Independent basis/mean matrices, PSD/FFT products, pre-RTC KIDS
solver audit inputs and map/kernel planes were excluded deliberately. Original
other/unclassified holdings total 83.332 GB and remain protected. They include
24.203 GB of pickle files, with the largest in tone-match tune/VNA contexts;
6.115 GB of `.dat` files, with the largest in atmospheric grids; and 8.733 GB of
`.txt` files spread across calibration/model/build records. Extension/project context
alone does not prove every payload's meaning. `MISC_RETAINED_EXTENSION_CENSUS.json`
records this bounded inspection; no unpickling or blanket extension purge occurred.
All 55 reviewed source/report archives and reviewed NPZ families remain.

The exact deletion manifest `TARGETS.json.gz` covers conventional RTC/PTC outputs,
native applied streams, calibrated RTC sample caches, FRUIT occurrence streams,
frozen parents, compressed final-PTC spools and solely associated sample axes/masks.
Eligibility was bound to producer schemas/receipts and contents, including 42
compressed spools (10.129 GB); filenames alone were insufficient.

All **1,311 mixed NetCDF files** were separated before deletion. Their 5.194 GB of
retained variables, dimensions and attributes live in labeled `*_purge_retained.nc`
companions. Reopening and raw-value hashes verified every retained variable;
`MIXED_CONTAINER_PRESERVATION.json.gz` records original schemas and exact checks.
No mixed-container exceptions remain. Independent safety review approved the exact
candidate; a fresh active-use check found no live targets. `EXECUTION_RECEIPT.json`
and `DELETION_LOG.jsonl` record the actual removal.

Post-verification found all targets absent, all **329,137 original non-target file
identities unchanged**, all 20 original symlinks unchanged and all companions freshly
checksum-verified. This includes all 20,171 original non-target NetCDFs. Verification
uses size, inode, times and other original stat fields rather than a fresh whole-tree
payload hash. Six focused purge tests and eight existing archive-reader tests pass;
there were no reductions or numerical scientific validation runs.

## Local packages available for future work

The inventory discovers **188 original detector acquisition groups**, including
raw tunes/VNA, plus 157 processed and 775 support-only identities. There are 27 sky
acquisitions. The following **21** have complete local raw packages against explicit
saved reduction input inventories, exact raw/processed tune associations, original
telescope streams and identified matching APT/configuration dependencies:

| Role | Observations |
|---|---|
| POINT | 148669, 148671, 152388, 152389, 152391, 152393, 152418, 152420, 152430, 152432, 152434 |
| BEAM | 148670, 152307 |
| OOF | 152385, 152386, 152387 |
| SCIENCE | 152390, 152392, 152419, 152431, 152433 |

Five nearly ready packages—**152398, 152403, 152408, 152413, 152425**—have raw,
tune and telescope streams but lack established expected-inventory/calibration
readiness. **123424** has 12 raw detector streams, processed tunes and a recomputed
telescope product, but lacks all 12 associated raw tunes and the original telescope
stream. Historical **129081** has support/configuration records but all 12 configured
raw detector files are absent. It is not counted as a ready raw acquisition.

Exact identities, local paths, date/source/mode, time endpoints, per-network tune
associations, expected networks, duplicates, byte totals, matching APT hashes and
historical Unity paths are in `OBSERVATIONS.json.gz`; `OBSERVATIONS.md` provides the
sky table. The OOF triplet uses tune 152384/0/1 by per-network TargSweep metadata.
APT copies with identical basenames were resolved against saved project-relative
paths; variants were not treated as interchangeable.

Completeness is against saved inventories, **not a live Unity inventory or a sample
quality assessment**. All 1,513 raw I/Q files have readable, nonempty matching
structure; this does not rule out internal data gaps. Network inventories vary:
26 sky groups have 11 networks and 123424 has 12. No universal 13-network rule was
imposed. The 26 original telescope streams cover detector endpoints. Unity paths
are historically recorded and unverified live; no Unity access or transfers occurred.

## Durable recovery and intentionally unavailable restart states

Compact recovery records were committed **before deletion** at
`6a0b75236c8880e0ae588f9b21c8766ebaa4eb9c`, reachable on
`codex/map-successor-first-forward-001` in permanent `/Users/gwilson/GitHub/citlali/.git`.
The working checkout is `/private/tmp/citlali-point-additive-20260923`; durability does
not rely on that temporary checkout. `RECOVERY_RECORD_INDEX.json.gz` binds 87,940
original compact records to 71,969 unique contents in four archive parts totaling
184,019,125 bytes, all read back and verified. No bulk stream replacement was archived.

Useful results—maps, measurements, scientific assessments, definitions and existing
recipes—remain. Reusing deleted RTC/PTC routes now requires explicit upstream
reprocessing with the matching raw/tune/APT/configuration/runtime inputs. Retired
experiments have deliberately lost instant restart; exact replay of every abandoned
trajectory is not promised. See `RECOVERY_AND_DISPOSITION.md` for gaps and recipes.

Original manifests and scientific conclusions remain unchanged. The root's
`LOCAL_DATA_PURGE_2026-09-26.json` and exact path list, and the POINT compaction
`CURRENT_ARCHIVE_INDEX.json`, supersede old availability claims. Old full-payload
seals no longer validate the current layout. Current readers identify intentional
absence, including previously deduplicated canonical copies, and never regenerate.

## Proposed working root and budget — owner decision pending

Propose `/Users/gwilson/work_toltec/local_data/citlali-work` as the future single
working root. It has **not** been created and no input sets were moved or selected.
Possible resident corpora, with shared paths counted once:

| Possible corpus | Measured inputs GB |
|---|---:|
| POINT 152389 + OOF 152385–152387 | 1.830 |
| Above + SCIENCE 152390 and POINT 152391 | 9.410 |
| Above + BEAM 148670 and POINT 148669/148671 | 26.963 |

The compact POINT+OOF set is the smallest useful starting option; adding the other
two options broadens scientific modes. Exact input-path unions are preserved in
`CORPUS_AND_BUDGET_PROPOSAL.json`. This is a storage proposal, not a new experiment.

A **provisional 200 GB** working-root envelope allocates 35 GB inputs, 25 GB retained
results, 20 GB optional explicitly justified caches, 40 GB active outputs and 80 GB
peak scratch. Input allowance follows the measured corpus sizes; output/scratch
allowances are planning headroom, **not measured universal peaks or an approved quota**.
Large beammap jobs and parallel campaigns need a peak-space check before launch.
No permanent RTC/PTC restart cache is assumed in this proposal.

All current historical projects remain outside that proposed root. Known producer
paths in `/private/tmp` checkouts and saved Unity recipes also lie outside it; this
purge did not sweep them. The next owner decision is the root, actual hard budget
and resident packages. No merge, push, new science, download or reorganization occurred.
