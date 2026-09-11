# Representation decision before empirical freeze

The initial 4-arcsec lattice / 5-arcsec FWHM failed the predeclared 5% raw
representation gate: the independent diffraction core had 8.23–9.21% image
error. The preserved initial implementation is `rbf_initial_representation.py`;
its complete measurements are in `REPRESENTATION_R0.1.json`.

The one permitted preparation adjustment changes only resolution to a 3-arcsec
lattice / 4-arcsec FWHM. Penalty strength, admission and all empirical gates stay
fixed. `REPRESENTATION_R0.2.json` passes: unregularized compact errors are
0.64–0.85%, diffraction 3.46–3.55%, and coma 1.03–1.58%, across arrays and phases.
Regularized errors are reported separately; diffraction reaches 4.95% and its
effective-area bias reaches 9.79%. Coma effective-area bias is 0.92–1.12% after
regularization; its phase sensitivity is below 0.2 percentage point. None of
these concentration values selected the geometry or penalty.

There are 3,093 basis centers per array. Basis setup is approximately 0.4 seconds
per array in this preparation run; screened conditioning bounds are below 4,830.
All pure-plane fits return zero source. Four focused tests pass with warnings
promoted to errors: normal operator/gradient, full overlap integrals and scale
invariance, nuisance-plane rejection, and multilobed admission independent of
Gaussian fitting or concentration. The test's explicit matrix algebra uses the
predecessor's finite-checked BLAS wrapper for the known macOS status-flag issue.

Both full representation directories are retained under `/private/tmp/` with
names `sci-fruit-point-rbf-representation-20260911-r0.1` and `...-r0.2`.
No empirical RBF trajectory was inspected before this choice. The single
empirical revision allowance remains unused.
