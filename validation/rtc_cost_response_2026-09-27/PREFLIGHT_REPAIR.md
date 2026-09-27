Independent review of f25ca650 required two bounded repairs before science:
combined native assigned-time/AST containing-pixel/signed CAL injection control,
and run-owned reserve/finally-accounting for the extra RTC-only diagnostic.
Both implemented. The new control traverses a pixel edge on consecutive native
samples, checks detector offset rotation, outside-raster zero, signed response,
calibration-only reconstruction, invalid NaN preservation and unchanged r/validity.
The first synthetic fixture incorrectly declared its NaN finite; ingress rejected
it, then the test declaration was corrected. No science attempt was consumed.
All76 focused native cases and31 Python cases pass. The previous config and207
baseline gates remain applicable; no production algorithm changed in this repair.
Diagnostic byte-equivalence now also includes native schedule i64 files.
