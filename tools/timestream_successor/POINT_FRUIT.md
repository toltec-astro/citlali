# Bounded POINT 152389 first FRUIT cycle

`run_point_fruit.py` binds the reviewed MAP, threshold-3 candidate and first-pass
pixel-scale products, invokes the native successor `point-fruit` terminal for
all three arrays, checks fresh-PTC zero replay, then executes one nonzero cycle.
It measures T1 and saves a replacement m1 without applying it. It requires a
clean committed tree, matching executable revision and fresh output directory.

The owner approved both additional choices on 2026-09-23: no feedback outside
the 40″ AST-centered analysis disk with original data/eligibility preserved;
and direct containing-pixel sampling on the exact saved post-RTC/CAL grid,
subtract before PTC and restore the identical operand afterward. These choices
are recorded in `validation/point_152389_fruit_first_cycle_2026-09-23/owner-binding.json`.
This empirical convention does not claim complete upstream response, zero sky
outside the disk, or recovery of missing absolute modes. Unsupported interior
model samples remain unavailable. The original constant background is excluded.

Native ownership: FruitPointModel owns this model evaluation and support rule;
FruitResidual owns exact immutable y-minus-operand; FruitLearned owns the
residual-bound invocation of unchanged native PTC numerical code. Its interface
cannot accept a saved first-pass basis. AST supplies detector coordinates on
saved RTC slots. PTC residual publication and uniform coefficients, VAL named
use and ForwardMapOperator resolve the new support. T1, R1 and A1 use that one
operator; total is a FRUIT product, not mislabeled as a PTC cleaned signal.

Each segment preserves the original input/slot/detector/mask/solver references,
exact operand, model support classes, fresh means/basis/cleaned values/causes,
convergence, eligibility changes and projector comparison. Output grid extent
and coverage threshold are resolved under the unchanged first-forward policy.
No old coverage is fabricated. Component maps each retain the six MAP planes;
the original six-plane products remain untouched. No model-only output occurs.

Selection for m1 uses its own descriptive centroid and background and the frozen
first-pass pixel-noise field intersected with new support. A failed, constrained
or incomplete-footprint fit withholds m1. No smoothing, union, fitted-source
replacement, rescaling, damping or new uncertainty campaign is performed.

Reproduction arguments: `--binary`, `--maps` (parent products), `--map-request`,
`--measurement`, `--admission` (reviewed companion-r2), `--additive` (reviewed
companion-r2), `--owner`, `--output`, `--workers` (1–3 array processes).
The existing measured successor process launcher caps each numerical library
to one thread. Original network/segment recipe and rank10 remain unchanged;
any renewed convergence change remains visible, without a rescue intervention.

A failed zero control blocks the nonzero cycle. A completed invocation stops
because the authorized one cycle is complete, not because of convergence.
All source measurements use the unchanged saved descriptive fit. Frozen-template
amplitudes have mJy/nominal-beam units; aperture sums times pixel area have
mJy/nominal-beam arcsec² units and are not integrated flux. Restored-model and
residual components are not independent source measurements.

## Bounded continuation through ten total cycles

The 2026-09-23 continuation directive supersedes only the single-cycle limit.
`run_point_fruit_ten.py --binary PATH --first FIRST_CYCLE_BUNDLE --owner OWNER_JSON
--output FRESH_DIRECTORY --workers 3` verifies original cycle/corrected manifests,
original MAP/admission/noise products and m1. Native continuation requests bind
cycle index, guarded predecessor state and exact original parent. Only indices
2–10 are accepted for nonzero execution; zero controls use index0 through the
same continuation path. No first nonzero cycle is repeated.

Each array stops independently at an invalid result, preserving trigger products.
Native output, exact operand/fresh learned state, total/residual/restored maps,
replacement candidate, fixed-definition diagnostics, ready-state lineage and
manifest are retained per cycle. Original inputs remain shared references.
Candidates replace rather than accumulate. m10 remains unapplied. Run directories
must be fresh; implicit retries/resumption/overwrite are not supported.

`report_point_fruit_ten.py --run RUN_DIRECTORY --output FRESH_REPORT_DIRECTORY`
verifies saved manifests and produces trajectory plots, fixed-scale map panels,
profiles, candidate supports and JSON/Markdown tables without scientific reruns.
Differences compare k−1 and k−2 on full science support and the fixed 40″ disk;
RMS normalization uses the original fitted amplitude, not a noise estimate.
Background-subtracted differences are separately labeled. Retained candidate
brightness changes use the intersection of selected supports.
