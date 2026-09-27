# Fixed 11 Hz comparison evidence and reproduction

Science candidate: `b3d997456a8b9762d9fc323e37827f96b4ec7b37`, tree
`ff40c74129a45a9d4d5e00874dbf09255ecf98b2`, based on completed RTC closeout
`ab43474a7c4988fcc7d72023e6695c034f56c22a`. Later commits add only extraction,
reporting, preservation and closeout. Branch `codex/rtc-fixed11-20260927`;
managed checkout `/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor`.
The user checkout, old branches, frozen profile/tag and prior compact evidence are preserved.

The adopted execution order is OWNER_DIRECTIVE.md. SELECTION.md, APPLICABILITY.json and
DESIGN.json define the exact one-time applicability and historical recipe.
APPLICABILITY hash `8c41c3946e87dc22a2baaaa6411122c14af47d32611f17a167ba644365962150`
was independently reviewed and fixed before all C executions. PREFLIGHT-REVIEW.json
records the fresh-context verdict and independent full-population reproduction.

All six timing runs and three C response branches ran on that same clean source
and binary. Nine trajectories, 45 outer feedback applications, two additional
RTC-only captures (11 original ingresses including those captures); no replacement.
Both captures were needed for raw coordinate/calibration/support evidence absent
from retained standard products. ATTEMPT_CHECK.json counts 22 actual network RTC Apply start/complete markers
across those 11 ingresses, with zero scientific revisions. Receipt attempt1001 is
a plan identity, not an execution count; parent counts remain separate.
B1 passed the prior B comparator and all binary/discrete checks. Each arm's second
and third runs passed strict own-arm comparisons. B source response is inherited
from prior A responses through completed A↔B equivalence and the new B1 gate.
No B injection trajectories were repeated.

## Saved-only report

From the managed checkout, run:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/fixed11-mpl XDG_CACHE_HOME=/private/tmp/fixed11-cache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python tools/timestream_successor/report_point_fixed11.py --root validation/rtc_fixed11_2026-09-27/compact --old validation/rtc_cost_response_2026-09-27/compact --output /private/tmp/fixed11-report-reproduced
```

This consumes compact JSON/NPZ only, including prior compact inherited controls.
It does not require original data, native scratch, a build, or new reductions.
`verify_saved_report.py` runs this with OS read denials on the campaign tree and
31 protected inputs, including a positive denial control, and requires byte-exact
METRICS.md, effects.json, report-facts.json and six PNGs. ASSESSMENT.md is the
human-authored interpretation of those reproducible measurements.

## Extraction and retained evidence

`extract_population.py` used the unchanged spectral/continuum recipe and current
response construction. Own-arm full support and B∩C matched-window outputs are
separate. Source exclusion is fixed to B M0 in all four population extractions.
`extract_comparison.py` checks untreated native identity, groups all/full/treated
and per-processing-segment statistics, and measures saved full maps/background.
`extract_detail.py` retains illustrative prior-event/native-end traces and the
combined finite filter response. `extract_timing.py` preserves every repeat,
worker-clock bounds, CPU, sampled RSS, warnings and counters. These require saved
native arrays and are provenance scripts, not post-cleanup commands.

- `compact/report/*-matched`: common physical windows, PSDs, full and treated coherence.
- `compact/report/B`, `C`: full support, per-detector boundary/event diagnostics;
  C also holds all signed source responses, own-reference ideal accounting and rasters.
- `compact/report/matched-population-summary.json`: power, continuum excess,
  prominence, prevalence/persistence and upper tails with full denominators and segments.
- `compact/report/native-comparison.json`: raw x/r and untreated byte identity;
  full/treated/untreated support loss before spectral intersection.
- `compact/report/final-map-comparison.*`: full support, common background,
  physical source vectors, own endpoint scales, existing directional correlations.
- `compact/B-REF-rep1/NATIVE_REGRESSION.json`: prior native/equivalence/response bindings.
- `COMPACT_MANIFEST.json`, `HASH_ONLY.json.gz`, `NATIVE_REMOVALS.json.gz`:
  compact preservation, redundant metadata/product hashes and exact new native scratch plan.

## Validation and limits

134 focused native tests, 42 initial Python tests, 207 baseline tests and full
required-data config preflight passed before execution. Final exact-source native
membership/gap/paired-coordinate fixtures passed again. Reporting/cleanup focused
fixtures are separately recorded. Nine saved endpoint operators and sigmas were
reproduced exactly using non-BLAS contraction; retained local RuntimeWarnings do
not stand in for validation. A reporting-only local-index lookup was corrected
before detail extraction; its initial failure made no science attempt. The first
filter-response plotting contraction triggered the known local BLAS warning; the
retained version uses direct finite sums. No scientific replacement was used.

Subset coherence is a read-only slice of the same complete-window complex Fourier
coefficients, not a changed estimator. No new threshold, classifier, scientific
noise estimator, map correction, source definition, or production default.
Local AppleClang/Homebrew/M4 Pro evidence only; no Unity execution or extrapolation.

## Guarded retention

After compact evidence is committed, saved-only reporting passes and independent
final review passes at an exact clean commit, `compact_record.py clean` verifies
the committed compact manifest, originals and prior compact evidence, verifies
every enumerated inactive new native payload, and removes only those files.
A reviewer does not repeat exhaustive scratch hashing. Metadata/FITS and logs
left in the campaign root are recorded honestly. Cleanup is not a blanket directory
removal. No push, merge, integration or automatic follow-on is authorized.
