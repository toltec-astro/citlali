#pragma once

#include <cmath>
#include <string>

namespace citlali::pipeline {

template <class VectorMap, class Logger>
double telescope_header_scalar(const VectorMap &tel_header,
                               const std::string &key, double fallback,
                               const Logger &logger) {
    const auto it = tel_header.find(key);
    if (it == tel_header.end() || it->second.size() < 1) {
        logger->warn("tel_header '{}' missing/empty; using fallback {}", key,
                     fallback);
        return fallback;
    }
    const double value = it->second(0);
    if (!std::isfinite(value)) {
        logger->warn("tel_header '{}' non-finite ({}); using fallback {}", key,
                     value, fallback);
        return fallback;
    }
    return value;
}

template <class VectorMap, class Logger>
double telescope_data_mean(const VectorMap &tel_data, const std::string &key,
                           double fallback, const Logger &logger) {
    const auto it = tel_data.find(key);
    if (it == tel_data.end() || it->second.size() < 1) {
        logger->warn("tel_data '{}' missing/empty; using fallback {}", key,
                     fallback);
        return fallback;
    }
    const double value = it->second.mean();
    if (!std::isfinite(value)) {
        logger->warn("tel_data '{}' mean non-finite ({}); using fallback {}",
                     key, value, fallback);
        return fallback;
    }
    return value;
}

}  // namespace citlali::pipeline
