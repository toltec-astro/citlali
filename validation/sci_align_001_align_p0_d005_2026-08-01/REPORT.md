# SCI-ALIGN-001 ALIGN-P0-D005 preregistration decision package

Date: 2026-08-01

Verdict: **OWNER_RETURN_REQUIRED — PHASE ONE UNAUTHORIZED**

## Frozen identity and scope

- Repair branch: `codex/repair-sci-align-001` at frozen phase-zero commit `53c7154a3633dfe19dc036cfb5a6250f729a897d`.
- Exact governing application parent: `9aae0e669384c5c0c0dda93debc194d6b8dac787`.
- Read-only coordination authority: `6785152c2a2d4113c9ba89073de00cb454aa70c4`; clean at generation.
- Corrected owner-decision identity: `4f905f4f353e91847a303f4f3959654f3f03c302`; the rejected expanded transcription `4f905f4f39461c8f9a86b0bf589880362d0a49f7` is not used.
- Frozen phase-zero `SHA256SUMS`: `074aff9deddd062d13a055589714f5d1b52ee18753052286119a184d2dbc08a2`. The package was verified and not rewritten.
- This package contains evidence/protocol only. No Citlali execution, application edit, Unity access, TolProj/suite mutation, sibling-repository edit, phase-one fixture, re-audit, merge, rebase, push, or production expansion occurred.

## Cohort and evidence classification

Pointing 152389 and Beammap 148670 are mandatory core fixtures. Beammap-supporting Pointings 148669/148671 and science-supporting Pointings 152389/152391/152393 provide legitimate Pointing-context repeatability; OOF 152385/152386/152387 provides multi-observation lifecycle/timing but not cross-focus PSF repeatability; science 152390/152392 is a long-mode timing/scan/product sentinel, not an ALIGN centroid/source-estimator gate. The suite selects but has not realized science-support observations 152418, 152420, 152430, 152432, and 152434. Their complete owner-local 240-file input corpus is available read-only and bound by manifest `1fb6bb026eb6a7e5e3c8398eb9fcd00470abf1810b37f1ee6873b8aac195f272`; only its native detector headers are used in D005, not as realized product evidence. Beammap 152307 is not selected because it is heterogeneous and outside the suite; the D002 combined-Beammap study therefore remains evidence-pending and is an explicit owner question.

The suite products identify historical Citlali builds `v4.0.0-3535-g7ca0be50, v4.0.0-3564-ge97de3fd, v4.0.0-3575-gcfae989c, v4.0.0-3585-gd339053c`. None is an exact whole-application `9aae0e669384c5c0c0dda93debc194d6b8dac787` execution. The Beammap 148670 product's `cfae989c` alignment/scan source files are byte-identical to the governing versions, so its 198-scan artifact is direct governing-code-path evidence but not an exact whole-build run. Frozen phase-zero continuity selects `pointings_v22`; suite/project metadata does not independently choose a versioned pointing directory. Accepted external Point/Beammap snapshots and suite products are separately classified in `baseline_product_manifest.csv`.

## Native timing and slot conclusions

All 187 distinct detector-file paths/references expose `FpgaFreq=256000000 Hz`, `AccumLen=2097152`, and `SampleFreq=122.0703125 Hz`: genuine native **1x** evidence only. They comprise 132 realized-config references plus 55 owner-local/unrealized support references; canonicalization yields 176 observation/interface identities across 16 observations and 8,673,204 native rows. The 11 duplicated observation/interface groups are asserted byte- and header-equivalent before canonicalization. Downstream RTC rates were not counted. Native 0.5x, 2x, and 4x evidence is absent and remains evidence-pending; 1x will not be resampled or scaled to manufacture those strata.

| Mandatory fixture | Native rows | Current supported | Union edge rows | Added grid positions | Max abs residual (ms) | Min half-cell margin (us) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| beammap 148670 | 4,220,705 | 4,220,689 | 16 | 3 | 3.918647766 | 177.352234 |
| point 152389 | 84,689 | 84,667 | 22 | 3 | 4.061937332 | 34.062668 |

Every ordinary supported row retains its slot under D002 round-half-up, with no exact half ties or collisions. The controlling limit is the strict native half cell, `abs(residual) < 4.096 ms` at 1x; measured margins do not replace that contract boundary.

All six realized low-level configs request exactly `0 s` on all 14 declared detector/HWPR offset entries. D005 applies each present detector's requested zero exactly once before slotting. The config value is bound, but the governing runtime does not persist independent realized-application evidence; the frozen phase-zero offset trace remains authoritative for that limitation. The five owner-local/unrealized supports have no config offset claim and contribute rate headers only.

All 132 realized detector references join the config-supplied `toltecN` identity exactly to the raw `Header.Toltec.RoachIndex` identity before offset lookup and slot analysis; swapped or duplicate supplied identities fail generation.

## Telescope envelope

All selected union-grid targets have finite, strictly increasing adjacent `TelTime` support. The exact fixed-cohort maximum used bracket is `0.021130561828613281 s`, set as an inclusive D005 validation envelope. This is not promoted to a general production cadence/gap limit; no padding was guessed. Invalid endpoints, nonpositive time, gaps, ambiguity, cross-gap support, and extrapolation fail closed.

## Scientific compatibility limits

- Point, OOF, and Beammap retain the repository's active zero-tolerance complete-product policies for unaffected records (volatile profile sidecar excluded by the existing policy). No nonzero source-crossing/centroid/PSF limit for records changed only by approved OD5 boundary repairs can be derived from one Beammap; that owner choice remains open.
- The source-crossing artifact retains distance and scan-window identities but no direct closest-approach timestamp/sample identity; absolute crossing time is unavailable and is not inferred.
- Science retains only the existing successor policy: map RMS `<=1e-8`, PTC-weight RMS `<=1e-9`, detector-median absolute/fractional `<=5e-5`/`1e-3`, and other diagnostic RMS `<=1e-7`. It is not used to tune CAL, AST, mapmaking, or source estimation.
- Pointing repeatability and fit-uncertainty summaries are descriptive. A single full Beammap and deliberately defocused OOF sequence cannot justify looser scientific thresholds.
- Single historical runtimes are not repeatability trials. The package proposes the already-established five-pair controlled design and 5% median ceiling, with owner return for five same-sign slowdowns. A setup-specific numerical margin and a storage byte ceiling remain unavailable.

## ALIGN-D004-HOLD-VALIDATION-001

Beammap 148670 has 383,699 governing common-grid rows and raw words `[0, 2, 8, 10, 64, 66, 72, 74]`. The independent outside-map-box condition is true on 232,776 rows. Released/current whole-word linear-to-nonzero, left/right raw-word nonzero, and left/right bit-`0x08` hypotheses differ materially before that condition; `0x02` and `0x40` both overlap `0x08` and occur alone.

Every tested Hold-true row is already outside the map box. Consequently all tested final states, all 198 legacy output windows, and all 241 maximal-false-run candidate windows are hypothesis-invariant; 167 distinct source-crossing-recorded legacy windows reproduce with zero mismatch. The governing implementation drops the first false sample after a true-to-false composite-final-state boundary in all 241 candidate runs, deletes 43 short/partial candidates, renumbers candidate `q=34..231` to outputs `1..198`, and applies extra edge context trimming.

These 241 zero runs are not authoritative physical scans. Only if owner Q1 authorizes the legacy whole-word-linear-any compatibility view combined with the separately applied governing outside-map-box condition to control raster segmentation do they become the preregistered candidate identities for applying OD5's half-open, first-sample, context, and retained-status repairs. Otherwise authoritative physical raster segmentation remains unavailable. The fixture proves compatibility of the current final state and reconstructs the exact 198 legacy outputs plus the conditional candidate identities, but it cannot identify a controlling predicate or transition side. The D004 owner-return condition is met. D005 selects neither and does not silently authorize a successor scan target.

The candidate registry is expressed in detector-reference lattice slots `k`, with separate current-grid and D002 union-grid array indices. The current grid spans `k=0..383698`; the union spans `k=-1..383700`. All three added union positions are final-state true, and the 241 candidate half-open lattice intervals and digest are unchanged by support expansion.

## Owner decision brief

Recommendation: record the D005 preregistration with its measured local evidence, explicit evidence gaps, and required owner return; do not treat it as phase-one authorization.

1. **D005-Q1-HOLD** — Will the owner explicitly authorize the named legacy whole-word-linear-any compatibility view combined with the separately applied governing outside-map-box condition to control raster segmentation without a producer-turn claim, leave stronger turnaround state unavailable, and apply only the OD5 scan-boundary repairs; or keep physical raster segmentation unavailable pending a separate scientific amendment with discriminating authority/evidence?
2. **D005-Q2-RATES** — May phase one be restricted to native 1x while 0.5x/2x/4x observational evidence remains pending, or are all four native observational strata prerequisites?
3. **D005-Q3-TELESCOPE** — Is 0.021130561828613281 s approved only as this fixed-cohort validation envelope, or as an admitted runtime limit? No general producer cadence/gap bound is proved.
4. **D005-Q4-BASELINE** — Which direct comparison authority controls phase one: a future exact governing-SHA 9aae run, or an explicit amendment naming historical accepted/suite snapshots?
5. **D005-Q5-RUNTIME** — Does the owner approve the five-pair runtime rule (existing 5% median ceiling plus owner return for five same-sign slowdowns) and a new setup-stage measurement, while I/O/storage retain only the frozen structural compactness rule until paired evidence exists; or what exact rule replaces it?
6. **D005-Q6-SCIENCE-SUPPORT** — Must the realized suite be completed with 152418/152420/152430/152432/152434, whose complete owner-local raw sets are now digest-bound, or is their raw-header-only 1x evidence sufficient while product/repeatability evidence stays pending?
7. **D005-Q7-COMBINED-BEAMMAP** — D002 assigns a combined-Beammap study to D005, but 148670 is the only selected suite Beammap. Will the owner admit and bind supplemental heterogeneous/out-of-suite Beammap 152307, name another already-local accepted Beammap reduction, or explicitly leave the combined-observation residual/source-crossing/centroid/PSF study evidence-pending?
8. **D005-Q8-CHANGED-SCIENCE** — For records changed solely by approved OD5 window repairs, must the current exact Point/Beammap product policy remain the gate (so any numeric change stops), or will the owner approve a separately specified uncertainty/repeatability-derived non-degradation rule? D005 evidence cannot freeze a nonzero value.

Until these choices are recorded, phase one remains unauthorized.
