# EL-F8 analysis attempt R0.3

## Status

Analysis r0.3 completed every registered numerical calculation and execution
check, then failed while serializing the result JSON because
`q_identity_closure.closure_pass` was a NumPy boolean rather than a built-in
Python boolean.

Four tables were written before the serialization failure and are retained as
non-authoritative partial evidence under `analysis-r0.3`:

| Object | Bytes | SHA-256 |
|---|---:|---|
| `COMPONENT_METRICS_R0.3.csv` | 10058 | `922e74507d7b0795542eaa61bc2a814697701a54da713e073a3403f0d0980aea` |
| `CROSS_TERMS_R0.3.csv` | 5258 | `d0971408e99dff471b035ab198bdc57950408576b12e4c45395e1827101b841d` |
| `TRIGGER_PIXELS_R0.3.csv` | 637 | `9444425fa2de289b44373d80725d153829bb15642fc97c66b63bb435f1114d25` |
| `PRIMARY_EXECUTION_R0.3.csv` | 299 | `9606aa5dda68b659655ebd6d89a9f100143457aa258583e8589b3c50ec6c2be0` |

No JSON result, report, plot, component FITS file, or provenance record was
written.  No scientific interpretation is adopted from the partial tables.

The bounded repair explicitly converts the roundoff bound to a Python float
and the closure comparison to a Python boolean before serialization.  A
focused JSON round-trip test freezes this requirement.  No input, replay,
checkpoint, map, numerical calculation, metric, or claim limit changes.  A
new analysis revision must use a fresh output directory.
