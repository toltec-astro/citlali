#pragma once

#include <citlali/core/config/config_error.h>

#include <string_view>
#include <vector>

namespace citlali::config {

inline constexpr const char *pointing_axis_az() {
    return "az";
}

inline constexpr const char *pointing_axis_alt() {
    return "alt";
}

inline bool is_pointing_axis_az(std::string_view axis) {
    return axis == pointing_axis_az();
}

inline bool is_pointing_axis_alt(std::string_view axis) {
    return axis == pointing_axis_alt();
}

inline bool is_supported_pointing_axis(std::string_view axis) {
    return is_pointing_axis_az(axis) || is_pointing_axis_alt(axis);
}

struct AstrometryPointingOffsetsConfig {
    bool enabled = false;
    std::vector<double> az_arcsec;
    std::vector<double> alt_arcsec;
    std::vector<double> modified_julian_date;
};

struct AstrometryConfig {
    AstrometryPointingOffsetsConfig pointing_offsets;
};

inline void validate(const AstrometryPointingOffsetsConfig &config,
                     ValidationReport &report) {
    const ConfigPath path{"inputs", "cal_items", "astrometry",
                          "pointing_offsets"};
    if (!config.enabled) {
        return;
    }

    if (config.az_arcsec.empty()) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "az", "value_arcsec"},
                         "must not be empty");
    }
    if (config.alt_arcsec.empty()) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "alt", "value_arcsec"},
                         "must not be empty");
    }
    if (config.az_arcsec.size() != config.alt_arcsec.size()) {
        report.add_error(path, "az and alt value_arcsec lengths must match");
    }
    const auto n_offsets = config.az_arcsec.size();
    if (n_offsets != 1 && n_offsets != 2) {
        report.add_error(path,
                         "must contain one or two values per pointing axis");
    }
    if (!config.modified_julian_date.empty() &&
        config.modified_julian_date.size() != 2) {
        report.add_error({"inputs", "cal_items", "astrometry",
                          "pointing_offsets", "modified_julian_date"},
                         "must be empty or contain two values after legacy "
                         "normalization");
    }
}

inline void validate(const AstrometryConfig &config, ValidationReport &report) {
    validate(config.pointing_offsets, report);
}

}  // namespace citlali::config
