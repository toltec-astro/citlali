#pragma once

#include <cmath>
#include <string>

namespace timestream {

template <class TransmissionMap>
std::string select_extinction_model(
    double tau_225_zenith, const TransmissionMap &transmission_zenith) {
    constexpr double pi = 3.14159265358979323846;
    constexpr double reference_elevation_rad = 80.0 * pi / 180.0;
    const double cosine_zenith =
        std::cos(pi / 2.0 - reference_elevation_rad);
    const double secant_zenith = 1.0 / cosine_zenith;
    const double airmass =
        secant_zenith *
        (1.0 - 0.0012 * (std::pow(secant_zenith, 2) - 1.0));

    std::string selected{"am_q0"};
    for (const auto &[name, transmission] : transmission_zenith) {
        const double model_tau = -std::log(transmission) / airmass;
        if (model_tau <= tau_225_zenith) {
            selected = name;
        }
    }
    return selected;
}

}  // namespace timestream
