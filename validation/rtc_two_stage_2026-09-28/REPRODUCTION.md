# Saved evidence reproduction

Run from this directory using `/Users/gwilson/tolteca/bin/python`.
The following commands render saved summaries and arrays only. They do not
launch a reduction, refit a source or generate a new uncertainty ensemble.

```sh
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/two-stage-mpl XDG_CACHE_HOME=/private/tmp/two-stage-cache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python two_stage_report.py --compact compact --output /private/tmp/two-stage-saved-report
MPLBACKEND=Agg MPLCONFIGDIR=/private/tmp/two-stage-mpl XDG_CACHE_HOME=/private/tmp/two-stage-cache OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 /Users/gwilson/tolteca/bin/python two_stage_comparison_report.py --compact compact --output /private/tmp/two-stage-saved-report
/Users/gwilson/tolteca/bin/python test_saved_report.py
/Users/gwilson/tolteca/bin/python verify_saved_report.py
```

`verify_saved_report.py` uses macOS Seatbelt to deny reads of every local science
campaign tree and all protected original dependencies, with a positive denial
control. It requires all eight PNGs and two Markdown tables to reproduce byte
for byte against `generated/`. The isolation receipt records the exact result.
This verifies report regeneration, not replay without native science inputs.

`COMPACT_MANIFEST.json` binds each retained file to its source hash, encoding,
packed hash and bytes. FITS and large YAML are losslessly gzip-compressed; native
arrays are only enumerated in the separate unapproved scratch proposal. Packed
FITS can be read by Astropy or decompressed into a separate temporary directory.
Compact evidence retains original receipts, per-run source and binary identities,
all M0–T5 products, measurement/noise packages, complete B/Q population tables,
fixed windows, original x/r excerpts, native support accounting, full B replay
comparators and historical response bindings. Absolute paths in original receipts
are provenance, not silently rebased replay paths.

`run_campaign.py` and `run_quarantine.py` are the historical guarded launchers.
They are **not reproduction commands**: the authorized science allowance is
exhausted. Population/response extraction scripts document calculations performed
on the retained native arrays; no further observational run is needed or authorized.
`two_stage_compact.py` is the one-time preservation script and intentionally
refuses an existing destination. It never deletes. No cleanup command is provided.

Scientific identities: infrastructure `ff92aa69d73aee9441e3269adf4ff4b9892ae509`;
B `9d28b2c6e134375b6f377ae59acc4e9f967ac3d4`; all Q
`a7351335f791c995e39d7cda4546412e3f4a5638`. Final documentation commits do not
relabel those executions. See `compact/EXECUTION_ACCOUNTING.json` for exact times,
resources, binary hashes and unchanged previous-failure provenance.
