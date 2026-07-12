#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>

namespace timestream::transient {

inline std::ptrdiff_t notch_settle_samples_for_width(
    double sample_rate_hz, double width_hz, double attenuation) {
    if (sample_rate_hz <= 0.0 || width_hz <= 0.0) {
        return 0;
    }
    if (!(attenuation > 0.0 && attenuation < 1.0)) {
        attenuation = 0.01;
    }

    constexpr double pi = 3.14159265358979323846;
    const double bandwidth = 2.0 * pi * width_hz / sample_rate_hz;
    const double beta = std::tan(bandwidth / 2.0);
    if (!std::isfinite(beta) || beta <= 0.0) {
        return 0;
    }
    const double gain = 1.0 / (1.0 + beta);
    const double radius_squared = 2.0 * gain - 1.0;
    if (!(radius_squared > 0.0)) {
        return 1;
    }
    const double radius = std::sqrt(radius_squared);
    if (!(radius > 0.0 && radius < 1.0)) {
        return 0;
    }
    const double samples = std::log(attenuation) / std::log(radius);
    if (!std::isfinite(samples) || samples <= 0.0) {
        return 0;
    }
    return static_cast<std::ptrdiff_t>(std::ceil(samples));
}

inline std::ptrdiff_t iir_highpass_settle_samples(
    double sample_rate_hz, double frequency_hz, int order) {
    if (sample_rate_hz <= 0.0 || frequency_hz <= 0.0 || order <= 0) {
        return 0;
    }

    constexpr double pi = 3.14159265358979323846;
    const double tau_sec = 1.0 / (2.0 * pi * frequency_hz);
    const double samples =
        5.0 * tau_sec * sample_rate_hz *
        static_cast<double>(std::max(1, order));
    return static_cast<std::ptrdiff_t>(
        std::ceil(std::max(0.0, samples)));
}

}  // namespace timestream::transient
