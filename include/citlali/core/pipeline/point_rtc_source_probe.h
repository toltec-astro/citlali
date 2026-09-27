#pragma once
#include <citlali/core/pipeline/fruit_cycle.h>
#include <cmath>
#include <stdexcept>

namespace citlali::pipeline {
// Explicit finite-response probe, not optical response or an inverse filter.
// Brightness is mJy/nominal beam at the native assigned occurrence; multiplier
// is CAL's selected flxscale times its atmosphere correction in those units/x.
// Input x is dimensionless delta_f/f. Caller supplies immutable original validity.
// Missing mapping never means zero brightness. Invalid original cells stay exact.
struct PointRtcProbeValue { double x, raw_increment, brightness; };
inline PointRtcProbeValue point_rtc_source_probe(double original_x, bool valid,
    const FruitOperand &sky, double cal_multiplier) {
    if(!valid)return {original_x,0,0};
    if(!std::isfinite(original_x) || sky.support==FruitModelSupport::unavailable ||
       !std::isfinite(sky.value) || !std::isfinite(cal_multiplier) || cal_multiplier==0)
        throw std::invalid_argument("pre-RTC source requires finite native calibration/projection");
    const double increment=sky.value/cal_multiplier;
    const double injected=original_x+increment;
    if(!std::isfinite(injected) || !std::isfinite(increment))
        throw std::invalid_argument("pre-RTC source overflow");
    return {injected,increment,sky.value};
}
} // namespace citlali::pipeline
