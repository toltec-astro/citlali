# SCI-FRUIT Q02 weight-precision diagnostic

The approved r0.5 diagnostic is complete. Start with the
[scientific report](SCIENTIFIC_REPORT.md), then the
[full result tables](report/FULL_RESULT_TABLES.md). Its findings guide method
preparation; they do not qualify weights or certify 20% precision.

The [approval record](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/WEIGHT_PRECISION_APPROVAL_2026-09-09.md)
binds the unchanged [protocol](../../doc/scientific_contracts/packages/SCI-FRUIT/v0.1/method_preparation/ordinary_map/method_definition/q02_review/r0.5/WEIGHT_PRECISION_PROTOCOL.md).
The dated directory is the predeclared run root. The numerical run occurred on
2026-09-09; report closeout occurred on 2026-09-10.

## Reproducibility and files

- `run_weight_precision.py` is the isolated implementation. It admits only the
  two exact files in `settings.json`, checks both hashes before signal access,
  verifies the arithmetic, records source/settings identities and preserves a
  new attempt directory. It has no Citlali execution or mapping path.
- `test_weight_precision.py` contains the eleven deterministic checks. It opens
  no observational signal and generates no random trials.
- `attempt_01/RUN_START.json` binds the actual pre-signal source/settings hashes,
  approval and environment; `DETERMINISTIC_VERIFICATION.json` records checks.
- `attempt_01/SUMMARY.json` and the per-observation/per-K JSON summaries contain
  all requested distributions and availability/influence counts.
- `attempt_01/*_groups.npz` retains rectangular window-by-detector arrays. `e>0`
  defines required groups. `n,mu,v,w,gamma,score_ok,p_raw,p_normalized` identify
  the measured statistics and availability. Undefined floats are NaN; JSON
  uses null. Nonrequired slots never enter normalization or summary populations.
  The last precision/bin-count axis follows the six taus in `settings.json`.
- `*_training_bins.npz` retains original-time bin IDs and counts, keyed by window
  and tau index. They conserve each group's exact training count.
- `*_chunk_diagnostics.npz` retains original-chunk and earlier/later training
  statistics, evaluation/central counts, and exclusion census. Sequential
  training exclusions are flags, coordinates, signal, source guard; evaluation
  has the first three only. Raw signal counters are training masked/nonfinite,
  then evaluation masked/nonfinite, including already flagged slots. They are
  not additional exclusions.
- `*_lag_diagnostics.npz` has detector-by-lag-by-category counts and correlations.
  Category order is within-chunk, cross-chunk. The last correlation axis is
  standardized centered signal, then its square minus one. No-pair/zero-norm
  correlations are unavailable.
- `verify_products.py` checks saved results and source identities and rehashes
  only the two admitted discovery inputs. It never opens 129081.
- `render_report.py` reads saved products only to render the full tables and
  the normalized-precision figure. It performs no new raw-input analysis.
- `attempt_01/PRODUCT_MANIFEST.json` binds the 33 run products, with a SHA-256
  sidecar. The top-level `RESULT_MANIFEST.json` binds the implementation,
  approval/review sources, report and completed attempt together.

The local verification command is:

```sh
/Users/gwilson/tolteca/bin/python /private/tmp/citlali-sci-fruit-method-scope-20260908/validation/fruit_q02_weight_precision_2026-09-09/verify_products.py attempt_01
```

No second scientific run is required. Authorized repairs must preserve this
attempt and use a distinct attempt directory. The tool's fixed input/method
checks do not authorize changing scientific settings or adding observations.
