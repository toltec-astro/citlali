#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <optional>
#include <string>
#include <string_view>

namespace citlali::config {

enum class MapGrouping {
    automatic,
    detector,
    network,
    array,
    frequency_group
};

enum class MapMethod {
    naive,
    jinc,
    maximum_likelihood
};

inline constexpr std::array<EnumName<MapGrouping>, 5> map_grouping_names{{
    {MapGrouping::automatic, "auto"},
    {MapGrouping::detector, "detector"},
    {MapGrouping::network, "nw"},
    {MapGrouping::array, "array"},
    {MapGrouping::frequency_group, "fg"},
}};

inline constexpr std::array<EnumName<MapMethod>, 3> map_method_names{{
    {MapMethod::naive, "naive"},
    {MapMethod::jinc, "jinc"},
    {MapMethod::maximum_likelihood, "maximum_likelihood"},
}};

inline std::optional<MapGrouping> parse_map_grouping(std::string_view value) {
    return parse_enum(value, map_grouping_names);
}

inline std::optional<MapMethod> parse_map_method(std::string_view value) {
    return parse_enum(value, map_method_names);
}

inline std::string_view to_string(MapGrouping value) {
    return enum_name(value, map_grouping_names);
}

inline std::string_view to_string(MapMethod value) {
    return enum_name(value, map_method_names);
}

inline bool is_map_grouping(std::string_view value, MapGrouping grouping) {
    return value == to_string(grouping);
}

inline bool is_map_grouping(MapGrouping value, MapGrouping grouping) {
    return value == grouping;
}

inline bool is_detector_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::detector);
}

inline bool is_detector_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::detector);
}

inline bool is_network_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::network);
}

inline bool is_array_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::array);
}

inline bool is_frequency_group_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::frequency_group);
}

struct MapmakingConfig {
    bool enabled = true;
    double crpix1 = 0.0;
    double crpix2 = 0.0;
    double crval1_j2000 = 0.0;
    double crval2_j2000 = 0.0;
    double tan_ra = 0.0;
    double tan_dec = 0.0;
    std::string unit = "mJy/beam";
    MapGrouping grouping = MapGrouping::automatic;
    MapMethod method = MapMethod::naive;
    std::string pixel_axes = "radec";
    double pixel_size_arcsec = 1.0;
    int x_size_pix = 0;
    int y_size_pix = 0;
    double coverage_cut = 0.0;
};

inline void set_mapmaking_enabled(MapmakingConfig &config, bool enabled) {
    config.enabled = enabled;
}

inline bool mapmaking_active(const MapmakingConfig &config) {
    return config.enabled;
}

inline void validate(const MapmakingConfig &config, ValidationReport &report) {
    check_greater_than(config.pixel_size_arcsec, 0.0,
                       {"mapmaking", "pixel_size_arcsec"}, report);
    check_minimum(config.x_size_pix, 0, {"mapmaking", "x_size_pix"}, report);
    check_minimum(config.y_size_pix, 0, {"mapmaking", "y_size_pix"}, report);
}

}  // namespace citlali::config
