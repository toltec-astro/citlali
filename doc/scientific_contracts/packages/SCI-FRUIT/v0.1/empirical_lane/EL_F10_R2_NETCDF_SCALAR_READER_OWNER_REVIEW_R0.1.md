# SCI-FRUIT EL-F10-R2 — NetCDF scalar-reader repair owner review r0.1

Decision identity:
`SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1`

Status: **owner decision required; no retry is authorized**

## What happened

EL-F10-R1 passed its repaired checkpoint-compatibility gate. The frozen
analyzer then failed while reading the receipt's schema name because NetCDF
returned an ordinary Python string and the helper called `.item()` on it.

Only `schema_identity` was accessed. The analyzer did not read total or target
`N`, `C`, or `Q`, did not open the target ledger, and wrote no result products.
This is a Python type-handling defect, not a failed scientific gate.

## Exact proposed repair

Change only the scalar conversion helper from its two-way bytes/`.item()`
logic to:

```python
if isinstance(value, bytes):
    return value.decode()
if hasattr(value, "item"):
    return value.item()
return value
```

Add focused tests showing that the helper returns the correct value for a
native `str`, `bytes`, and a NumPy numeric scalar. Run the full baseline and
FRUIT-loop Python test suite, Ruff, byte compilation, and the repository
whitespace check. Freeze the repaired analyzer and test hashes only after
those checks pass.

Then create an R0.3 analysis registration that rebinds the exact unchanged
receipt, ledger, diagnostic maps, checkpoint, historical comparison products,
R1 compatibility record, repaired analyzer, and repaired tests before one
analysis retry.

## Gates that may not change

The R1 checkpoint normalization and required difference set remain exact.
Every map-neutrality comparison, total-accumulator closure rule, 305/34/271
sample ledger, binary64 error formula and safety factor, support rule, region,
trigger pixel, descriptive output, and claim limit remains unchanged from the
approved EL-F10 and R1 packets.

Any hash failure, reader failure, neutrality failure, closure failure,
sample-ledger mismatch, out-of-bound reconstruction, or unexplained support
change stops the retry. No tolerance or scientific interpretation may be
chosen after seeing the values.

## Boundaries

No Citlali replay is needed or authorized. R2 does not authorize a Citlali
code or configuration change, replacement of retained files, a detector
judgment, a safeguard or penalty decision, a recurrence change, production
use, qualification, Gate D, Stage B, or Unity activity.

## Owner choices

### Choice A — Approve the parser-only repair and analysis retry (recommended)

Approve
`SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1` against the exact
`EL_F10_R2_BUNDLE_MANIFEST_R0.1.md`.

### Choice B — Keep the R1 stop

Leave the accounting values unread and retain the reader exception as the end
of EL-F10.

### Choice C — Request a different repair

Revise the parser, verification, or retry scope before the accounting values
are accessed.

The exact affirmative statement for Choice A is:

> I approve `SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1` against the exact `EL_F10_R2_BUNDLE_MANIFEST_R0.1.md`.
