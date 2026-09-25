# Reproduce POINT-EXPECTED-v1

This records the completed bounded experiment, not a request to repeat it now.
Use the existing /Users/gwilson/tolteca environment. Campaign source is
0441f293e3daa5bab853efd43e278cb445fdec3c; reporting source is recorded in
report/REPORT.json. A clean checkout with Git metadata is required. Archives
preserve exact contents; they do not substitute for source identity. The
49 consumed inputs are hash-bound in freeze/prospective.json, including
the sealed parent indices. Earlier products and the first preparation attempt
remain unchanged.

Run from a clean checkout of the campaign revision and use a **fresh** output
path; the commands reject existing stage destinations. All source/seed/mask/
mean/covariance/threshold bindings must be retained. The independent review
must bind the realized fresh checks and exact source before calibration.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-expected-mpl XDG_CACHE_HOME=/private/tmp/point-expected-cache
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_expected.py prepare --output /ABSOLUTE/FRESH/EXPECTED
```

Review the 18 diagnosed cases, 15 bright controls and six noiseless scenes
before proceeding. A review JSON must contain disposition `pass`, the exact
`source_revision`, and `prospective_sha256`, with the review scope/evidence.
Copy that receipt to the fresh bundle; never reuse a review with a different
prospective hash. Then run in order:

```sh
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_expected.py calibration --output /ABSOLUTE/FRESH/EXPECTED --review /ABSOLUTE/FRESH/EXPECTED/PREFLIGHT_REVIEW.json
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_expected.py validation --output /ABSOLUTE/FRESH/EXPECTED --review /ABSOLUTE/FRESH/EXPECTED/PREFLIGHT_REVIEW.json
/Users/gwilson/tolteca/bin/python tools/timestream_successor/run_point_expected.py screen --output /ABSOLUTE/FRESH/EXPECTED
```

Each array uses 1024 fields per phase and two paired scenes. Stress reuses
the first256 validation fields at1.25 amplitude. Twelve independent workers,
one requested numerical thread, one fit shared by three policies. Thresholds
freeze before validation; fields were previously examined, not untouched.
No scientific retry, additional field, feedback cycle or native call occurs.
The screen reuses the checked bright-source measurements (zero new fits).

At the report revision, with a clean checkout and absent report directory:

```sh
/Users/gwilson/tolteca/bin/python tools/timestream_successor/report_point_expected.py --output /ABSOLUTE/FRESH/EXPECTED
```

Reporting reads saved rows/FITS only. Review source, realized numerical
results and all five figures before closure. The closure tool at the final
recorded revision copies task logs from /private/tmp, checks the consumed
inputs/stage manifests, preserves source archives and regression-binary
identities, and seals once. Inventory expects the original task's sibling
first-preparation path and successful logs; it is an evidence workflow, not a
portable science dependency. A rerun should document its own attempt paths
and check logs instead of reusing unrelated evidence.

```sh
/Users/gwilson/tolteca/bin/python tools/timestream_successor/close_point_expected.py inventory --output /ABSOLUTE/FRESH/EXPECTED
/Users/gwilson/tolteca/bin/python tools/timestream_successor/close_point_expected.py seal --output /ABSOLUTE/FRESH/EXPECTED --review /ABSOLUTE/FINAL_REVIEW.md
```

Focused tests: `python -m pytest tools/timestream_successor/test_point_expected.py tools/timestream_successor/test_point_corr_admission.py -q`
using the prescribed local interpreter. Final focused17, successor300 with111
subtests, baseline207 with137 subtests, and required configuration checks pass.
The existing frozen native build supplies38 available native/CLI tests;
unrelated `_NOT_BUILT` tests could not run, so the broad initial CTest attempt
is not described as a full pass. Logs are preserved under provenance/logs.
No native source changed or executable was rebuilt for this work order.
Native executable source remains ee54efb6d32759003d4249f3a4ad90dbeb9c10d9.

The latest code implements the named diagnostic route only. Historical fitter
defaults are unchanged; no production or normative SCI-POINT contract changes.
