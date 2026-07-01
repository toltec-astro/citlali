# Wiener Filter Follow-Ups

Date: 2026-07-01

This note tracks the remaining P2 Wiener-filter work after the denominator
runtime controls and OpenMP accumulation changes.

## P2 Items To Address Next

1. Reduce FFT wrapper allocation and copy overhead.

   The denominator loop still calls `engine_utils::fft2` three times per
   denominator term. `fft2` copies the input matrix into FFTW buffers, executes
   the plan, allocates a new `Eigen::MatrixXcd`, copies FFTW output back into
   Eigen storage, and returns by value. Add an `fft2_into(...)` style API, or
   otherwise reuse caller-owned output buffers, before attempting deeper
   denominator math changes.

2. Validate and potentially taper the kernel-template radial tail.

   `make_kernel_template` currently fills radii beyond the spline range with
   the final radial-average value. If that value is nonzero or noisy, the
   template can become effectively full-map support, which increases
   denominator cost and may affect normalization. Inspect representative kernel
   templates first; then consider zeroing, clipping, or tapering beyond the
   last reliable radial bin.

3. Add focused denominator tests and benchmarks.

   Add a small synthetic map test that checks `max_denom_iters`,
   `denom_rel_tol`, and OMP/non-OMP consistency under a fixed cap. Add a
   denominator microbenchmark on a modest map size so runtime changes are
   measurable without a full TolTEC reduction.
