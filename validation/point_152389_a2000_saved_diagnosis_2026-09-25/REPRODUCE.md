# Reproduction and availability

## Exact records

Owned worktree `/private/tmp/citlali-point-additive-20260923`, branch `codex/map-successor-first-forward-001`; initial clean base `26a3047c654386e38adac860b91491136888bc64`. No change to the separate main checkout, sibling repositories, native application, defaults or upstream scientific state.

Arithmetic source: `f5e02f9c40a4372e91210cc082dff948f3047130`. Initial implementation `ed0fd90a84260362da52e108367942f0343e1114` was prepared but never analyzed; review repairs preceded the f5 execution. Corrected figure labels and final tabulation: `257f6dbc1` (full identity in provenance/SOURCE.json). Prospective input/source hashes precede each analysis or figure execution. INTERPRETIVE_TABLES.json binds its two derivative JSON inputs.

Parent bundle: `/Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-guarded-feedback-comparison-20260925`. Parent root manifest SHA256: `5f6bf979f4e4ee2f4e2d249512c9107e73ed7f8fab8c58ccc257db27c65265a9`. PROSPECTIVE.json binds 3,424 relevant files totaling 937,841,216 bytes through sealed parent entries or authenticated ancestor references. Of these, 3,077 are opened by the analysis reader; remaining bound files provide preparation/authority or companion evidence. All bound bytes are checked before and afterward. The 125 GB parent was neither copied nor rehashed as a whole.

The cached POINT-EXPECTED-v1 Gaussian-plus-constant measurements and their fit/guard flags are copied, not refitted. Frozen template projection, aperture centers/radii, 4 arcsec² pixel area, original benchmark and archived signed/seed definitions come from the parent's bound caches and frozen geometry. Seed raster and projected data-injection reference remain separate operands. No library fit default is invoked. J2000 east/north offsets and brightness mJy/nominal beam are preserved. Fixed circular aperture radii are 9.4341720477, 14.1512580715 and 18.8683440953 arcsec about the frozen center; no missing-area fill. Shape norm domain: 1,257 finite seed pixels; full-map original benchmark: 10,601 supported pixels; even-image stamp: 1,681 pixels. All original response domains are complete.

## Commands

These are the commands executed locally (fresh output required for prepare/analyze). Exact numerical replay uses a clean checkout at f5; subsequent cached figure/tabulation publication uses a clean checkout at 257. Do not change the current checkout to reproduce; use a separate clean worktree if needed. The paths below record the original execution worktree, not an instruction to overwrite this sealed output. Use a fresh output path for any replay.

```bash
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-saved-mpl
export XDG_CACHE_HOME=/private/tmp/point-saved-cache
/Users/gwilson/tolteca/bin/python -c 'import matplotlib.pyplot'
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-point-additive-20260923/tools/timestream_successor/diagnose_point_saved_a2000.py prepare --output <fresh-output>
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-point-additive-20260923/tools/timestream_successor/diagnose_point_saved_a2000.py analyze --output <fresh-output>
# Publication at 257, using only already serialized arithmetic:
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-point-additive-20260923/tools/timestream_successor/diagnose_point_saved_a2000.py figures --output <fresh-output>
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-point-additive-20260923/tools/timestream_successor/tabulate_point_saved_a2000.py --output <fresh-output>
```

The font-cache initialization is necessary on a fresh plotting environment: the first attempt completed arithmetic but its strict subprocess guard blocked Matplotlib's `fc-list --help`. That aborted attempt is retained at the sibling `-aborted-font-discovery` directory, with execution_completed=false and error/log. After initializing fonts separately, the same f5 arithmetic route succeeded. Independent review confirmed equality of arrays and pixel records and numerical JSON excluding runtime. No tolerance was relaxed, failed parent product repaired, or inference fallback used. The earlier prepared-only record is retained at `-preparation-ed0`. Their compact prospective/error evidence is included in provenance; large duplicated derivative arrays need not be copied into this final bundle.

`EXECUTION_ROUTE.json` and `FIGURE_EXECUTION_ROUTE.json` record actual repository calls and completion status. The arithmetic/figure route blocks other scientific execution imports, non-Git subprocesses and parent writes. The tabulator reads only two derivative JSON files and formats their cached values/mask sets. Source inspection and independent review establish zero new fits, candidate/policy evaluations, covariance fits, processing, projection, mapmaking, injections or feedback. Those zeros are a reachability/execution conclusion, not invented invocation counters.

## Checks and measured runtime

Ten focused tests pass, independently reproduced. They exercise signed references/even controls, seed-domain versus full-map semantics, exact candidate/constant partition, unavailable versus zero, own-policy/stage matching, actual imposed m0 turnover, own trajectory drift and static/no-cache-fallback route restrictions. Existing upstream native/baseline/config gates are reused; no new native build or full science/config campaign is claimed for this reporting-only addition.

Independent recalculation: seed coefficients/partitions/own terminal drifts reproduce; all 99 signed responses differ by at most 4.44e−16; aperture bookkeeping by at most 3.98e−11. Saved score audit discrepancy is zero; cleaned/restored and applied-candidate accounting errors are below 1.57e−11 (inherited tolerance 1e−8). All five figures were visually inspected by writer and independent reviewer; the fitted-peak-change label was corrected without altering arithmetic.

| Phase | Measured seconds | Scope |
|---|---:|---|
| Initial ed0 preparation | 68.6614 | Prepared only; no analysis |
| Repaired f5 preparation | 11.6265 | Safe C YAML parser, input hashing |
| First aborted attempt arithmetic | 115.7867 | Completed arithmetic; plotting import then blocked |
| Separate plotting initialization | 15.2473 | Font cache only |
| Successful arithmetic | 107.9424 | Saved reads, checks, calculations; single-thread numeric libraries |
| Successful initial render/tables | 19.9551 | Five figures and numerical report |
| Successful pre/post input verification | 1.7503 / 0.8091 | Eight I/O hash workers, no nested numeric parallelism |
| Corrected cached figure publication | 20.2550 | Reads serialized arithmetic; no repeat calculations of science products |
| Figure pre/post verification | 0.7829 / 0.7964 | Same bound bytes |
| Final compact tabulation | 0.1288 | Existing derivative JSON only |

Successful process peak RSS: 1,336,049,664 bytes (macOS process high-water, libraries/cached arrays included; no analysis child workers). Timings name their scopes: serialization outside the analysis/render timers and process initialization were not separately timed. They must not be summed or advertised as total wall time; no end-to-end total was recorded. Independent review has separate runtimes. RUNTIME.json gives exact values, support and consumed-path lists.

## Availability

| Requested diagnosis | Completed evidence / boundary |
|---|---|
| All 15 histories, M0/T1–T10 and m0–m10 | 165 cached state/measurement records and masks; m0 seed exception explicit; m10 unapplied |
| Seed parallel/orthogonal/full norms and checkpoints | 33 paired stages; shape/candidate/full domains separate; original normalization |
| Stage membership and selected brightness | Complete masks/identities, normal baseline/supplement transitions and actual candidate turnover |
| Candidate mask/brightness/constant partition | 30 post-M0 exact partitions, signed projections, norms and three apertures |
| Pixel evidence | 100 implicated pixels × 165 states; short union-of-two-complete-sets ledger with 11 pixels |
| Neighbor contributions / normalization causes | Individual weights and raw numerator/normalizer not serialized; unavailable, not reconstructed |
| Score audit and full-footprint validity | Saved normalized EVIDENCE/sqrt(VAR) only on available tests; true availability certifies combined predicate; separate reason for false not recoverable |
| Same-map versus trajectory selection | Nine authenticated REF T10 cross-candidates, still unapplied |
| Centering, residual/restored/total and state | 120 paired post-M0 non-REF comparisons and 1,440 matched segment summaries; no basis application or direct matched-basis distance |
| Signed compact/shoulder response and even control | 99 original-domain responses, 33 even maps; ideal even zero; saved ideal fits only at T10 |
| Source-scale measurements and own late drift | All-stage cached peak/centroid/width/ellipse/constant/template/apertures; own map/candidate deltas and complete T10 actual/ideal tables |
| Causal mechanism, asymptotic behavior, significance | Not established by saved history; no new intervention/inference authorized here |
| Source-location uncertainty | Retains a2000 source-transfer qualification; frozen-C0 errors conditional |

Stop after assessment/review/seal/local commit. No method adoption, thresholds, weights, production change, applied m10, cycle11, Unity, merge or push.
