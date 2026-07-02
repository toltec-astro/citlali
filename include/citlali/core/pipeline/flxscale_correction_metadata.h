#pragma once

#include <cmath>

namespace citlali::pipeline {

template <class RawObs>
auto flxscale_correction_metadata(const RawObs &rawobs) {
    return rawobs.flxscale_correction();
}

template <class FlxscaleCorrection>
bool has_flxscale_correction(const FlxscaleCorrection *flxscale_corr) {
    return flxscale_corr != nullptr;
}

template <class FlxscaleCorrection>
double flxscale_correction_factor(const FlxscaleCorrection &flxscale_corr) {
    return flxscale_corr.value();
}

inline bool is_valid_flxscale_correction_factor(double factor) {
    return std::isfinite(factor) && factor > 0.0;
}

}  // namespace citlali::pipeline
