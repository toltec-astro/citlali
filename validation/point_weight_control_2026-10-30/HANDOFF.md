# 152418 equal-versus-noise-weight comparison

This is an owner-run paired experiment, not calibration qualification. No reduction
is launched by preparation. Temporal research remains paused.

## What is controlled

One TolTECA-prepared 152418 observation runs through RAW/KIDs/RTC/CAL once and
learns optical health once. Both arms read those exact same publications. Arrays
a1100/a1400/a2000, APT and telescope inputs, disturbance exclusion, optical cut
0.10, original source masks, ranks 3/4/6, 2 arcsec pixels, coverage cut 0.15,
ALS settings and ten FRUIT applications are fixed.

The noise arm uses the current reference-variance cleaning weights and
`1/max(reference variance, remaining variance)` map coefficients. The equal arm
qualifies the same noise estimates but uses equal coefficients in cleaning and
mapmaking. Unsupported noise never becomes a unit-weight fallback. Equal
coefficients have the explicitly arbitrary scale 1 beam²/mJy², which cancels
from the map signal; its separate product family makes no noise/precision claim.

This measures the **combined effect of detector weighting in cleaning and
mapmaking**. It does not separate those two contributions. M0 maps give the
pre-feedback contrast. Both subsequent trajectories use the same FRUIT recipe,
but models and learned values may differ as downstream consequences of weighting.
This is not a frozen-source-model comparison. The scientific source fitting,
source guards and uncertainty limitations remain unchanged.

Compact per-segment hashes compare ordered detector/slot domains, input and
source-free masks, residual support, cleaned support and actual admitted MAP
sample support across all corresponding cycles. The same map grid is required.
Changing effective pixel coverage due to coefficient sums is a weighting effect;
the coverage-cut value is held fixed. If either arm stops early, fails cleaning,
or differs in sample support, the full pair is explicitly **not** reported as a
matched completed comparison. Do not retune membership or thresholds to force a
pass; inspect the saved diagnostics. No observational claim is made before Grant
runs it and those checks pass.

## Prepared layout and owner procedure

Local project root:
`/Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027`.

Fresh trial: `pointings-weight-control-152418-20261030`.
Evidence: `evidence/weight-control-152418-20261030`.
Preparation uses the existing weighted 152418 numbered YAML files as its source,
existing preparation environment, TolProj executable snapshots and guarded
node-local launch. It does not reuse an old map as either comparison arm.

```bash
cd /Users/gwilson/work_toltec/local_data/2026-refactor/projects/COM01-local-trial-20261027/pointings-weight-control-152418-20261030
bash 02_redu.sh
bash 02_redu.sh --run
```

The first command checks only; the second launches both arms sequentially.
Strict shell options stay inside the child script. Each invocation reserves a
new `reduNN`; existing reductions and reference products are never removed.
The frozen READY record verifies configuration, runtime and support bindings
before launch. No automatic retries or background monitoring are installed.

For the first successful invocation:

- Noise-weighted final maps: `reduced/redu00/152418/raw`.
- Equal-weight final maps: `reduced/redu00/comparison/equal/152418/raw`.
- Controlled-pair result: `reduced/redu00/weight-comparison.json`, requiring
  `status: matched_support_complete`.
- Initial/intermediate noise maps: `reduced/redu00/.point-work/weighted-00/products`
  through `weighted-10/products`.
- Initial/intermediate equal maps:
  `reduced/redu00/.point-work/equal-control/weighted-00/products` through
  `weighted-10/products`.
- Common health/drop diagnostics stay in the main reduction directory.

Quiet mode remains enabled. Shared CAL binary transports are released only after
both arms and required outputs finish successfully. Maps, fit diagnostics, source
models, receipts, compact support checks and frozen health are retained. A failed
comparison preserves its generated evidence and does not perform that cleanup.

## Verification and interpretation

Native focused tests cover independent unweighted-SVD agreement with unequal
detector noise, unchanged default weighting, equal-control noise unavailability,
exact FRUIT restoration and family/evidence mismatch rejection. Python tests cover
opt-in configuration, one shared upstream/health domain, arm-specific feedback,
honest control units, all support-key mismatches and complete/failed pair outcomes.
Full local CTest, baseline-tool and configuration gates are required. The build
is the existing supplemental Mac AppleClang/Homebrew realization, with the retained
KIDs dependency qualification; it is not Unity Spack evidence.

After the owner run, compare M0 and M10 peak, centroid, widths, radial/background
structure and aperture behavior on all arrays, with exact support checks first.
Do not interpret peak change alone as flux loss, detector failure or M1 damage.
Both arms share one observation's telescope state. A result can establish a
weighting-dependent difference under that state, not the absolute correctness of
either map. Existing POINT products remain ineligible for calibration.

Exact source/runtime/preparation identities and independent review are recorded
in the completion record accompanying this handoff.
