# Reproduction and read-only reporting

Science ran once at e3cbf1eeadda256e4d9f400a834f9f7d462c1ae2. The original input
manifest and local build procedure remain in data/development/bright_point_base_v1
and doc/BRIGHT_POINT_PIPELINE.md. Use a new explicitly authorized output for any
reproduction; existing attempts cannot be resumed or overwritten. No Unity command
or transfer is needed for these locally verified inputs.

The experiment's committed code/profile differs scientifically only in ptc.rank.
Exact resolved source/input/operand bindings are in compact/PREFLIGHT.json,
compact/RESOLVED-COMMON-INPUTS.json, each rank's freeze/prospective.json and
compact/EXECUTION.json. The matched native executable SHA/version and thread
settings are in the execution record and each native RUN.json.

Actual invocation, after the reviewed preflight and matching build:

```sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MKL_NUM_THREADS=1 BLIS_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR=/private/tmp/ptc-rank-mpl XDG_CACHE_HOME=/private/tmp/ptc-rank-cache
POINT_REPO=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
RANK_OUTPUT=/Users/gwilson/work_toltec/local_data/citlali-validation/successor-point-152430-a2000-ptc-rank-20260926
$HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" rank-run \
  --output "$RANK_OUTPUT" --binary "$POINT_REPO/build/bin/citlali" \
  --review "$RANK_OUTPUT/PREFLIGHT-REVIEW.json"
```

The command above records the completed run, not permission to rerun it. To inspect
or regenerate reporting from the retained compact record, with no original/native
arrays:

```sh
POINT_REPO=/Users/gwilson/.codex/worktrees/ptc-rank-response/citlali-refactor
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/ptc-rank-mpl XDG_CACHE_HOME=/private/tmp/ptc-rank-cache \
  $HOME/tolteca/bin/python "$POINT_REPO/tools/timestream_successor/point_pipeline.py" rank-report \
  --output "$POINT_REPO/validation/ptc_rank_response_2026-09-26/compact"
```

No fitter, covariance learner or native processing is invoked by reporting. Spatial
figures evaluate the already saved fit parameters only. Historical absolute paths
in bindings are provenance; the report resolves product paths under its explicit
root and accepts the verified lossless .fits.gz representations.

Retained audit (no raw input access):

```sh
$HOME/tolteca/bin/python "$POINT_REPO/validation/ptc_rank_response_2026-09-26/audit_saved.py" \
  --root "$POINT_REPO/validation/ptc_rank_response_2026-09-26/compact"
```

The focused gate evidence records 42 Python tests, 34 CTests, 207 baseline-tool
tests and full config preflight. Eight reporting-focused tests passed again after
adding final saved-product diagnostics; 44 focused tests passed after the final
CLI failure-propagation repair (mocked failures only, no science rerun). These are local AppleClang/Homebrew results,
not representative Unity/Spack qualification. The original reference profile and
freeze tag remain untouched. No push or merge was performed.
