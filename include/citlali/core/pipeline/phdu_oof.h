#pragma once

#include <cmath>
#include <limits>
#include <string>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class MapBuffer, class Logger>
double phdu_oof_rms(const MapBuffer &mb, Eigen::Index map_index,
                    const std::string &redu_type,
                    const std::string &array_name,
                    const std::string &filepath, const Logger &logger) {
    double rms = 0.0;

    if (redu_type != "beammap" && std::isfinite(mb->median_err(map_index)) &&
        mb->median_err(map_index) > std::numeric_limits<double>::epsilon()) {
        rms = std::pow(mb->median_err(map_index), 0.5);
    }
    else if (redu_type != "beammap" &&
             std::isfinite(mb->median_err(map_index)) &&
             mb->median_err(map_index) < 0.0) {
        logger->warn("negative median_err for PHDU {} in {}; using OOF_RMS=0",
                     array_name, filepath);
    }

    return rms;
}

}  // namespace citlali::pipeline
