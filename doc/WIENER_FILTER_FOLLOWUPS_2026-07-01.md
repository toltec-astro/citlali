# Wiener Filter Follow-Ups

Date: 2026-07-01

This note tracks the P2 Wiener-filter work after the denominator runtime
controls and OpenMP accumulation changes.

## Status

1. Reduce FFT wrapper allocation and copy overhead.

   Addressed by adding `engine_utils::fft2_into(...)` and switching the Wiener
   numerator, convolution, template FFT cache, and denominator paths to reuse
   caller-owned output matrices. This removes repeated `Eigen::MatrixXcd`
   allocations from the hottest FFT call sites while preserving the existing
   FFTW input/output buffer copy model.

2. Validate and potentially taper the kernel-template radial tail.

   Addressed with `wiener_filter.kernel_template_tail_mode`. The default
   `constant` preserves legacy behavior. `zero` truncates beyond the last
   valid radial bin, and `cosine` tapers that tail to zero at the map edge.
   The builder now logs the tail value, relative tail amplitude, and fraction
   of pixels affected.

3. Add focused denominator tests and benchmarks.

   Focused synthetic tests were added for `fft2_into`, `max_denom_iters`,
   `denom_rel_tol`, and zero-tail kernel templates. A runnable denominator
   microbenchmark is still worth adding once the local test dependency setup is
   fixed so benchmark targets can be built normally.
