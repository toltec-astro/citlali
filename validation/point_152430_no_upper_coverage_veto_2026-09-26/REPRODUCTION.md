# Reproduction and retained-product use

Execution source: `ac2bee1febaab82b808f7f751374b797de033c6a`, branch `codex/map-successor-first-forward-001`, permanent common Git `/Users/gwilson/GitHub/citlali/.git`. Preflight review binds this source, the native executable SHA256, and prospective manifest. Later commits add only current report/cleanup tooling and compact results. No push or merge.

## Read retained results without native processing

The `compact/` directory mirrors the replacement output root. `report/MEASUREMENTS.md` contains the 15-cell matrix, selected M0/T1/T3/T5/T10 histories and measurement definitions; JSON files retain all M0–T10 rows. Maps, candidates, admission masks, signed controls/residuals, injection rasters and noise models are retained. History bindings retain their original producer absolute paths and hashes. Retained FITS maps use lossless `.fits.gz` compression; all decompressed bytes were verified. The saved-product report command rebases those paths onto the supplied compact root and reads that explicit compressed representation; it fails on a missing retained map and never falls back to the original local_data tree.

From the checked-out source containing this report tool:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point152430-report-mpl XDG_CACHE_HOME=/private/tmp/point152430-report-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python tools/timestream_successor/report_independent_point.py --output validation/point_152430_no_upper_coverage_veto_2026-09-26/compact
```

This reads saved measurements and maps only: no fitting, covariance learning, admission reevaluation or native processing. Reporting runtime is separate from science runtime. SOURCE/INPUT hashes are not silently rewritten to new locations.

Large current native YAML metadata is losslessly compressed as `original.yaml.gz`; CLEANUP.json lists those representations and original digests. To read it use Python `gzip.open(path, 'rt')`. Binary numeric RTC/PTC products are intentionally deleted. Their old receipt hashes are provenance, not a claim that the checkpoint remains available. Do not invoke a saved per-cycle request as a restart.

## Fresh scientific reproduction (not executed at closeout)

Use the execution source commit and the preserved local environment. The existing cached build is a supplemental local realization, not the accepted Spack V2 campaign. Build environment metadata and CMake cache are retained. No environment migration is implied.

1. Check out the exact execution source in an appropriate clean workspace. Build `citlali_cli` using the existing compatible environment and verify `--version` matches the checkout. Record the resulting binary identity.
2. Verify INPUTS.json: same detector acquisition 152430/0/2, networks 11/12; tune 152430/0/1; associated telescope stream, canonical APT, calibrations and selected configuration. No transfers were needed in this run. Inputs outside the campaign root remain unchanged.
3. Prepare a fresh dedicated output root with `prepare_independent_point.py --output <fresh-root>`. This reconstructs input/config bindings from preserved originals; do not reuse compact maps/noise as processing parents.
4. Bind an independent preflight review JSON to that exact clean source, prospective manifest and binary. The original review file is valid only for its exact identities; changed paths/configuration identities need a new binding, not an edited old seal.
5. With all numerical thread variables fixed to one, run these stages in order:

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point152430-reproduce-mpl XDG_CACHE_HOME=/private/tmp/point152430-reproduce-cache OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python tools/timestream_successor/run_independent_point.py upstream --output <fresh-root> --binary <reviewed-binary> --review <new-exact-review.json>
```

Repeat the same command with `base`, then `shapes`, `references`, and `responses` in place of `upstream`, honoring each recorded stop/dependency. BASE REF must complete before shape construction; shapes freeze before other policy endpoints. Never rerun a failed phase or logical native attempt in the same root. Hard caps remain 15 trajectories, 150 feedback attempts, four initial physical inputs, nine ideal T10 controls and three REF endpoint covariance analyses. T10 consumes m9; m10 is saved and unapplied.

The exact executed command, executable identity, thread environment, native wall time and sampled RSS are in each RUN.json. Successful shapes, native accounting/control audits and per-cycle fitted measurements are in compact records. The current cleanup tool is restricted to the named original campaign root and requires terminal phase records; adapt a future cleanup scope only under that future work order.

The independent 52 focused tests, native 31-test gate and full config preflight are regression evidence; they do not establish observational false-admission calibration, upstream optical response or production acceptance. Endpoint propagation is conditional on the background model, including the unchanged finite high-coverage continuation. Practical effect/sigma ratios are not significance tests of paired differences.
