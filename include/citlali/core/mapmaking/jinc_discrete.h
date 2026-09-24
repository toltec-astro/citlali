#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace citlali::mapmaking {
// Numerical primitive for SCI-JINC v0.1/r0.3. Explicit inputs only: this type
// selects no TolTEC parameters, coefficient family, or numerical certificate.
// Coordinates are same-occurrence continuous FITS pixels (1-based centers).
// M is the row-major affine tangent-plane pixel matrix, radians per pixel.
struct JincDiscreteSpec {
    std::size_t rows, columns, phases;
    std::array<double,4> pixel_matrix_rad;
    double scale_rad, a, b, c, rmax;
};
struct JincDiscreteSample {
    double x, y, value, coefficient, frequency_hz;
    bool admitted;
};
struct JincDiscreteAccumulation {
    std::vector<double> numerator, normalization, quadratic, coefficient_squared_time;
    std::vector<double> absolute_normalization;
    std::vector<std::uint64_t> members;
    // Algebraic diagnostics only, never a certified JINC science-validity mask.
    std::vector<double> algebraic_value;
    std::vector<std::uint8_t> algebraic_support;
};
struct JincCertifiedAccumulation {
    JincDiscreteAccumulation sums;
    // N,C,Q,T absolute bounds and full signal bound; diagnostic, not base roles.
    std::array<std::vector<double>,5> error;
    std::vector<double> tolerance_lower;
    // 1 range/FP path,2 denominator,4 ancillary,8 signal,16 no members.
    std::vector<std::uint8_t> causes;
};
// Compact phase cache, no sample-to-pixel matrix. One immutable realization can
// be shared across independent accumulations. Worker-local planes are combined
// in fixed contiguous-block order; results promise tolerance, not bit invariance.
class JincDiscreteKernel {
public:
    explicit JincDiscreteKernel(JincDiscreteSpec);
    JincDiscreteAccumulation accumulate(std::span<const JincDiscreteSample>,
                                       std::size_t workers) const;
    // Caller must bind proved cache-error bounds to this exact realization.
    JincCertifiedAccumulation accumulate_certified(std::span<const JincDiscreteSample>,
        std::span<const double> coefficient_errors, std::size_t workers, double frozen_peak) const;
    double coefficient(std::size_t qx, std::size_t qy, int dx, int dy) const;
    std::size_t half_width() const noexcept {return half_;}
    const JincDiscreteSpec &spec() const noexcept {return spec_;}
    static double analytic(double normalized_radius, double a, double b,
                           double c, double rmax);
private:
    JincDiscreteSpec spec_;
    std::size_t half_, side_;
    std::vector<double> cache_;
};
} // namespace citlali::mapmaking
