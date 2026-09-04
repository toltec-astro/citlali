# SCI-FRUIT EL-F10-R2 execution result r0.3

Test ID: `SCI-FRUIT-EL-F10-R2-NETCDF-SCALAR-READER-REPAIR-R0.1`

Status: **stopped at the unchanged exact-closure gate; routine reader defect identified**

The approved scalar repair passed its focused and full Python verification.
The R0.3 registration then validated all 24 bound files, and the repaired
analyzer passed the map-neutrality and checkpoint-compatibility gates.

The exact total-accumulator closure gate stopped the analysis because the four
comparisons against FITS were false. No result products were written, and no
Citlali replay was performed.

## Read-only diagnosis

The diagnostic receipt stores the internal Eigen map orientation. Citlali's
FITS writer explicitly reverses the column axis before serializing image HDUs
(`include/citlali/core/utils/fits_io.h`, `add_typed_hdu`). The analyzer compared
the receipt arrays directly with the FITS arrays without applying that output
orientation.

The diagnosis established exact, zero-difference equality after reversing only
the receipt column axis for all relevant checks:

- total-accumulator signal versus FITS `signal_I`;
- finalized and captured formal coefficient versus FITS `weight_formal_I`;
- captured empirical coefficient versus FITS `weight_I`; and
- captured normalization support versus FITS formal support.

The finalized formal coefficient already matched the captured formal
coefficient exactly in the untransformed receipt orientation. Shape, data type,
row order, numerical values, and thresholds otherwise agreed. This is therefore
a local analysis-reader orientation defect, not a failed accounting closure.

## Disposition

Under
`SCIENTIFIC_OWNER_ROUTINE_DEFECT_REPAIR_DIRECTION_2026-09-04.md`, the analyzer
will align each two-dimensional receipt plane with the documented FITS output
orientation, add a focused regression test, rebind the unchanged inputs and
repaired analyzer before access, and continue the already authorized analysis.
No scientific gate, bound, region, trigger, input, Citlali algorithm,
configuration, reduction, or replay count changes.
