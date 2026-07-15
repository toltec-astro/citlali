#pragma once

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/pipeline/pointing_offset_state.h>

#include <algorithm>
#include <cctype>
#include <string>
#include <utility>

namespace citlali::pipeline {

inline std::string normalized_pointing_axis_name(std::string axis_name) {
    std::transform(axis_name.begin(), axis_name.end(), axis_name.begin(),
                   [](unsigned char c) {
                       return static_cast<char>(std::tolower(c));
                   });
    return axis_name;
}

inline PointingOffsetState make_pointing_offset_state(
    const citlali::config::AstrometryPointingOffsetsConfig &config) {
    PointingOffsetState state;
    state.arcsec[citlali::config::pointing_axis_az()] =
        Eigen::Map<const Eigen::VectorXd>(
            config.az_arcsec.data(), config.az_arcsec.size());
    state.arcsec[citlali::config::pointing_axis_alt()] =
        Eigen::Map<const Eigen::VectorXd>(
            config.alt_arcsec.data(), config.alt_arcsec.size());
    state.modified_julian_date = Eigen::Map<const Eigen::ArrayXd>(
        config.modified_julian_date.data(),
        config.modified_julian_date.size());
    return state;
}

inline void install_astrometry_config(
    citlali::config::AstrometryConfig observation,
    citlali::config::AstrometryConfig &target,
    PointingOffsetState &pointing_offsets) {
    auto next_pointing_offsets =
        make_pointing_offset_state(observation.pointing_offsets);
    target = std::move(observation);
    pointing_offsets = std::move(next_pointing_offsets);
}

}  // namespace citlali::pipeline
