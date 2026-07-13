#pragma once

#include <citlali/core/config/enum_parser.h>

#include <array>
#include <map>
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

enum class MapPixelAxes {
    radec,
    altaz,
    galactic
};

enum class SourceMapRegime {
    source_dominant,
    source_faint,
    blank_field,
    unknown
};

struct MapmakingJincFilterConfig {
    double r_max = 3.0;
    int subpixel_n = 1;
    std::map<std::string, std::array<double, 3>> shape_params{
        {"a1100", {1.1, 1.67, 2.0}},
        {"a1400", {1.1, 2.17, 2.0}},
        {"a2000", {1.1, 3.17, 2.0}},
    };
};

struct MapmakingMaximumLikelihoodConfig {
    int max_iterations = 50;
    double tolerance = 1e-20;
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

inline constexpr std::array<EnumName<MapPixelAxes>, 3> map_pixel_axes_names{{
    {MapPixelAxes::radec, "radec"},
    {MapPixelAxes::altaz, "altaz"},
    {MapPixelAxes::galactic, "galactic"},
}};

inline constexpr std::array<EnumName<SourceMapRegime>, 4>
    source_map_regime_names{{
        {SourceMapRegime::source_dominant, "source_dominant"},
        {SourceMapRegime::source_faint, "source_faint"},
        {SourceMapRegime::blank_field, "blank_field"},
        {SourceMapRegime::unknown, "unknown"},
    }};

inline std::optional<MapGrouping> parse_map_grouping(std::string_view value) {
    return parse_enum(value, map_grouping_names);
}

inline std::optional<MapMethod> parse_map_method(std::string_view value) {
    return parse_enum(value, map_method_names);
}

inline std::optional<MapPixelAxes> parse_map_pixel_axes(std::string_view value) {
    return parse_enum(value, map_pixel_axes_names);
}

inline std::optional<SourceMapRegime> parse_source_map_regime(
    std::string_view value) {
    return parse_enum(value, source_map_regime_names);
}

inline std::string_view to_string(MapGrouping value) {
    return enum_name(value, map_grouping_names);
}

inline std::string_view to_string(MapMethod value) {
    return enum_name(value, map_method_names);
}

inline std::string_view to_string(MapPixelAxes value) {
    return enum_name(value, map_pixel_axes_names);
}

inline std::string_view to_string(SourceMapRegime value) {
    return enum_name(value, source_map_regime_names);
}

inline bool is_map_grouping(std::string_view value, MapGrouping grouping) {
    return value == to_string(grouping);
}

inline bool is_map_grouping(MapGrouping value, MapGrouping grouping) {
    return value == grouping;
}

inline bool is_map_method(MapMethod value, MapMethod method) {
    return value == method;
}

inline bool is_map_pixel_axes(std::string_view value, MapPixelAxes axes) {
    return value == to_string(axes);
}

inline bool is_map_pixel_axes(MapPixelAxes value, MapPixelAxes axes) {
    return value == axes;
}

inline bool is_detector_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::detector);
}

inline bool is_automatic_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::automatic);
}

inline bool is_detector_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::detector);
}

inline bool is_network_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::network);
}

inline bool is_network_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::network);
}

inline bool is_array_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::array);
}

inline bool is_array_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::array);
}

inline bool is_frequency_group_map_grouping(std::string_view value) {
    return is_map_grouping(value, MapGrouping::frequency_group);
}

inline bool is_frequency_group_map_grouping(MapGrouping value) {
    return is_map_grouping(value, MapGrouping::frequency_group);
}

inline bool is_naive_map_method(MapMethod value) {
    return is_map_method(value, MapMethod::naive);
}

inline bool is_jinc_map_method(MapMethod value) {
    return is_map_method(value, MapMethod::jinc);
}

inline bool is_maximum_likelihood_map_method(MapMethod value) {
    return is_map_method(value, MapMethod::maximum_likelihood);
}

inline bool is_radec_map_pixel_axes(std::string_view value) {
    return is_map_pixel_axes(value, MapPixelAxes::radec);
}

inline bool is_radec_map_pixel_axes(MapPixelAxes value) {
    return is_map_pixel_axes(value, MapPixelAxes::radec);
}

inline bool is_altaz_map_pixel_axes(std::string_view value) {
    return is_map_pixel_axes(value, MapPixelAxes::altaz);
}

inline bool is_altaz_map_pixel_axes(MapPixelAxes value) {
    return is_map_pixel_axes(value, MapPixelAxes::altaz);
}

inline bool is_galactic_map_pixel_axes(std::string_view value) {
    return is_map_pixel_axes(value, MapPixelAxes::galactic);
}

inline bool is_galactic_map_pixel_axes(MapPixelAxes value) {
    return is_map_pixel_axes(value, MapPixelAxes::galactic);
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
    MapPixelAxes pixel_axes_frame = MapPixelAxes::radec;
    SourceMapRegime source_map_regime = SourceMapRegime::unknown;
    double pixel_size_arcsec = 1.0;
    int x_size_pix = 0;
    int y_size_pix = 0;
    double coverage_cut = 0.0;
    MapmakingJincFilterConfig jinc_filter;
    MapmakingMaximumLikelihoodConfig maximum_likelihood;
};

inline void set_mapmaking_enabled(MapmakingConfig &config, bool enabled) {
    config.enabled = enabled;
}

inline bool mapmaking_active(const MapmakingConfig &config) {
    return config.enabled;
}

}  // namespace citlali::config
