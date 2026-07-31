#pragma once

#include <cmath>

namespace mapmaking {

inline constexpr double convolve_numerical_support_fraction_floor = 1e-6;

inline bool convolve_stochastic_input_weight(double weight) {
    return weight > 0.0 && std::isfinite(weight);
}

inline double convolve_numerical_support_floor(double kernel_square_sum) {
    return convolve_numerical_support_fraction_floor * kernel_square_sum;
}

inline bool convolve_has_numerical_variance_support(
    double kernel_square_overlap, double kernel_square_sum) {
    return std::isfinite(kernel_square_overlap) &&
           std::isfinite(kernel_square_sum) &&
           kernel_square_sum > 0.0 &&
           kernel_square_overlap >
               convolve_numerical_support_floor(kernel_square_sum);
}

}  // namespace mapmaking
