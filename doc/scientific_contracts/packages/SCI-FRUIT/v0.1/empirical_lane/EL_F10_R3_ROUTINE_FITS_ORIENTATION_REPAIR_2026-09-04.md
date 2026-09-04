# SCI-FRUIT EL-F10-R3 — delegated FITS-orientation reader repair

Repair identity:
`SCI-FRUIT-EL-F10-R3-FITS-ORIENTATION-READER-REPAIR-R0.1`

Date: `2026-09-04`

Authority:
`SCIENTIFIC_OWNER_ROUTINE_DEFECT_REPAIR_DIRECTION_2026-09-04.md`

The R0.3 analyzer passed the approved NetCDF scalar repair and then stopped at
the unchanged exact-closure gate. Read-only diagnosis showed that all failed
comparisons become bitwise exact after reversing only the receipt's column
axis. This is the transformation explicitly performed by Citlali's FITS writer
when converting an internal Eigen matrix to a FITS image.

This record delegates only the following routine repair:

1. read every two-dimensional receipt plane in its stored internal matrix
   orientation;
2. reverse its column axis once to match the science FITS output convention;
3. add a focused orientation test;
4. rerun the same Python verification;
5. freeze a new output-bound registration against the unchanged retained
   products; and
6. continue the local accounting analysis.

This repair does not alter a scientific equation, gate, numerical bound,
support rule, region, trigger, input, output-map value, Citlali algorithm,
configuration, reduction, or replay count. Any scientific gate failure after
the orientation repair remains a significant decision point and stops the
analysis.
