# Reproduce the bounded diagnosis

Use the existing Python environment and a clean checkout of the final reviewed
source recorded in COMPLETION.json. The diagnostic function executed at
970a86dbcb1ad8a86b6025c91cfcecbf0f15b986; later changes only add its report.
Use a new output directory. The following records reproduction, not a request
to repeat the completed experiment now.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/point-blank-mpl XDG_CACHE_HOME=/private/tmp/point-blank-cache
/Users/gwilson/tolteca/bin/python tools/timestream_successor/diagnose_point_blank_fits.py --parent /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924 --output /ABSOLUTE/FRESH/DIAGNOSIS
/Users/gwilson/tolteca/bin/python tools/timestream_successor/diagnose_point_blank_fits.py --report-only --parent /Users/gwilson/work_toltec/local_data/citlali-validation/development-runs/successor-point-152389-correlation-admission-20260924 --output /ABSOLUTE/FRESH/DIAGNOSIS
/Users/gwilson/tolteca/bin/python -m pytest tools/timestream_successor/test_point_corr_admission.py -q
```

The source and parent identities are verified before use. The command records
the 18-case selection before fitting, reuses prescribed existing noise seeds,
and performs exactly three budget-only probes. Report-only reads recorded
results without generating fields or running fits. Its report directory must
not exist. The inspected original 40-arcsecond domains each contain 1,257
finite supported pixels. No native binary is called; no upstream work or
admission products are published. Source archives preserve contents but need
Git metadata for source-identity checks.

Closure verifies the 13 consumed parents again against the original manifest,
preserves the initial diagnostic manifest and render, records logs and source
archives, and publishes a final bundle manifest. The sealed preceding
correlation experiment is never modified. Optimizer failures remain unknown
for calibration; the isolated probes do not replace any outcome.
