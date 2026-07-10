#pragma once

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

}  // namespace citlali::config
