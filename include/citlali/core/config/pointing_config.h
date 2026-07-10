#pragma once

#include <citlali/core/config/enum_parser.h>

#include <array>
#include <optional>
#include <string_view>

namespace citlali::config {

enum class PointingSourceStrategy {
    standard,
    psf_preserve
};

enum class FruitLoopsCenterMode {
    automatic,
    header,
    peak,
    map_center
};

inline constexpr std::array<EnumName<PointingSourceStrategy>, 2>
    pointing_source_strategy_names{{
        {PointingSourceStrategy::standard, "standard"},
        {PointingSourceStrategy::psf_preserve, "psf_preserve"},
    }};

inline constexpr std::array<EnumName<FruitLoopsCenterMode>, 4>
    fruit_loops_center_mode_names{{
        {FruitLoopsCenterMode::automatic, "auto"},
        {FruitLoopsCenterMode::header, "header"},
        {FruitLoopsCenterMode::peak, "peak"},
        {FruitLoopsCenterMode::map_center, "map_center"},
    }};

inline std::optional<PointingSourceStrategy> parse_pointing_source_strategy(
    std::string_view value) {
    return parse_enum(value, pointing_source_strategy_names);
}

inline std::optional<FruitLoopsCenterMode> parse_fruit_loops_center_mode(
    std::string_view value) {
    return parse_enum(value, fruit_loops_center_mode_names);
}

inline std::string_view to_string(PointingSourceStrategy value) {
    return enum_name(value, pointing_source_strategy_names);
}

inline std::string_view to_string(FruitLoopsCenterMode value) {
    return enum_name(value, fruit_loops_center_mode_names);
}

inline bool is_pointing_source_strategy(std::string_view value,
                                        PointingSourceStrategy strategy) {
    return value == to_string(strategy);
}

inline bool is_pointing_source_strategy(PointingSourceStrategy value,
                                        PointingSourceStrategy strategy) {
    return value == strategy;
}

inline bool is_standard_pointing_source_strategy(std::string_view value) {
    return is_pointing_source_strategy(value, PointingSourceStrategy::standard);
}

inline bool is_standard_pointing_source_strategy(
    PointingSourceStrategy value) {
    return is_pointing_source_strategy(value, PointingSourceStrategy::standard);
}

inline bool is_psf_preserve_pointing_source_strategy(std::string_view value) {
    return is_pointing_source_strategy(
        value, PointingSourceStrategy::psf_preserve);
}

inline bool is_psf_preserve_pointing_source_strategy(
    PointingSourceStrategy value) {
    return is_pointing_source_strategy(
        value, PointingSourceStrategy::psf_preserve);
}

inline bool is_fruit_loops_center_mode(
    std::string_view value, FruitLoopsCenterMode mode) {
    return value == to_string(mode);
}

inline bool is_fruit_loops_auto_center_mode(std::string_view value) {
    return is_fruit_loops_center_mode(value, FruitLoopsCenterMode::automatic);
}

inline bool is_fruit_loops_header_center_mode(std::string_view value) {
    return is_fruit_loops_center_mode(value, FruitLoopsCenterMode::header);
}

inline bool is_fruit_loops_peak_center_mode(std::string_view value) {
    return is_fruit_loops_center_mode(value, FruitLoopsCenterMode::peak);
}

inline bool is_fruit_loops_map_center_mode(std::string_view value) {
    return is_fruit_loops_center_mode(value, FruitLoopsCenterMode::map_center);
}

struct PointingConfig {
    PointingSourceStrategy source_strategy = PointingSourceStrategy::standard;
    bool fit_gaussian = true;
    FruitLoopsCenterMode fruitloops_center_mode = FruitLoopsCenterMode::automatic;
    double header_max_radius_arcsec = 0.0;
    bool header_require_coverage = true;
};

}  // namespace citlali::config
