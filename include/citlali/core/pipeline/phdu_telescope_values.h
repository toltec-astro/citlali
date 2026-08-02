#pragma once

#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <CCfits/CCfits>
#include <fmt/core.h>

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

template <class VectorMap, class Logger>
double telescope_data_mean(
    const VectorMap &tel_data, const TimestreamAlignmentState &alignment,
    const std::string &key, double fallback, const Logger &logger) {
    const auto it = tel_data.find(key);
    if (it == tel_data.end() || it->second.size() < 1) {
        logger->warn("tel_data '{}' missing/empty; using fallback {}", key,
                     fallback);
        return fallback;
    }
    const double value = governing_compatibility_mean(
        it->second, alignment);
    if (!std::isfinite(value)) {
        logger->warn(
            "tel_data '{}' governing-compatibility mean non-finite ({}); using fallback {}",
            key, value, fallback);
        return fallback;
    }
    return value;
}

template <class FitsEntry, class Logger>
void add_phdu_double_key(FitsEntry &fits_entry, const std::string &array_name,
                         const Logger &logger, const std::string &key,
                         double value, const std::string &comment,
                         double fallback = 0.0) {
    if (!std::isfinite(value)) {
        logger->warn(
            "PHDU key '{}' non-finite ({}) for array {} in {}; using fallback {}",
            key, value, array_name, fits_entry.filepath, fallback);
        value = fallback;
    }
    try {
        fits_entry.pfits->pHDU().addKey(key, value, comment);
    } catch (const CCfits::FitsError &e) {
        throw std::runtime_error(
            fmt::format(
                "failed PHDU float key '{}' for array '{}' (file={} value={}): {}",
                key, array_name, fits_entry.filepath, value, e.message()));
    }
}

template <class Logger>
std::string apt_table_header_name(const std::string &apt_filepath,
                                  const Logger &logger) {
    if (apt_filepath.empty()) {
        logger->warn("APT filepath empty; using N/A");
        return "N/A";
    }

    std::vector<std::string> apt_filename;
    std::stringstream ss(apt_filepath);
    std::string item;
    char delim = '/';

    while (std::getline(ss, item, delim)) {
        apt_filename.push_back(item);
    }
    if (apt_filename.empty()) {
        logger->warn("APT filepath '{}' parsed empty; using N/A", apt_filepath);
        return "N/A";
    }
    return apt_filename.back();
}

}  // namespace citlali::pipeline
