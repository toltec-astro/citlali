#pragma once

#include <citlali/core/config/config_error.h>
#include <citlali/core/config/enum_parser.h>

#include <array>
#include <map>
#include <optional>
#include <string>
#include <string_view>

namespace citlali::config {

enum class MapFilterType {
    wiener_filter,
    convolve,
    destripe
};

enum class MapFilterTemplateType {
    kernel,
    gaussian,
    airy,
    highpass
};

enum class MapFilterEdgeTaperMode {
    none,
    cosine
};

inline constexpr std::array<EnumName<MapFilterType>, 3> map_filter_type_names{{
    {MapFilterType::wiener_filter, "wiener_filter"},
    {MapFilterType::convolve, "convolve"},
    {MapFilterType::destripe, "destripe"},
}};

inline constexpr std::array<EnumName<MapFilterTemplateType>, 4>
    map_filter_template_type_names{{
        {MapFilterTemplateType::kernel, "kernel"},
        {MapFilterTemplateType::gaussian, "gaussian"},
        {MapFilterTemplateType::airy, "airy"},
        {MapFilterTemplateType::highpass, "highpass"},
    }};

inline constexpr std::array<EnumName<MapFilterEdgeTaperMode>, 2>
    map_filter_edge_taper_mode_names{{
        {MapFilterEdgeTaperMode::none, "none"},
        {MapFilterEdgeTaperMode::cosine, "cosine"},
    }};

inline std::optional<MapFilterType> parse_map_filter_type(std::string_view value) {
    return parse_enum(value, map_filter_type_names);
}

inline std::optional<MapFilterTemplateType> parse_map_filter_template_type(
    std::string_view value) {
    return parse_enum(value, map_filter_template_type_names);
}

inline std::optional<MapFilterEdgeTaperMode> parse_map_filter_edge_taper_mode(
    std::string_view value) {
    return parse_enum(value, map_filter_edge_taper_mode_names);
}

inline std::string_view to_string(MapFilterType value) {
    return enum_name(value, map_filter_type_names);
}

inline std::string_view to_string(MapFilterTemplateType value) {
    return enum_name(value, map_filter_template_type_names);
}

inline std::string_view to_string(MapFilterEdgeTaperMode value) {
    return enum_name(value, map_filter_edge_taper_mode_names);
}

inline bool is_map_filter_template_type(std::string_view value,
                                        MapFilterTemplateType type) {
    return value == to_string(type);
}

inline bool map_filter_template_uses_fwhm(MapFilterTemplateType value) {
    return value == MapFilterTemplateType::gaussian ||
           value == MapFilterTemplateType::airy;
}

inline bool map_filter_template_uses_fwhm(std::string_view value) {
    if (auto parsed = parse_map_filter_template_type(value)) {
        return map_filter_template_uses_fwhm(*parsed);
    }
    return false;
}

struct MapFilterEdgeGuardConfig {
    bool enabled = true;
    std::string weight_threshold_mode = "coverage_cut";
    std::string hits_threshold_mode = "core_median_fraction";
    double hits_core_fraction = 0.15;
    double guard_radius_fwhm = 1.0;
    std::string fill_mode = "core_median";
    MapFilterEdgeTaperMode taper_mode = MapFilterEdgeTaperMode::none;
    double taper_min_fraction = 0.25;
};

struct MapFilterConfig {
    bool enabled = false;
    MapFilterType type = MapFilterType::convolve;
    MapFilterTemplateType template_type = MapFilterTemplateType::kernel;
    bool lowpass_only = false;
    bool normalize_errors = false;
    MapFilterEdgeGuardConfig edge_guard;
    double denom_rel_tol = 1.e-4;
    double tail_frac_tol = 5.e-2;
    int max_loops = 500;
    int denom_check_iters = 0;
    int max_denom_iters = 0;
    std::map<std::string, double> template_fwhm_arcsec;
};

struct SourceFindingConfig {
    bool enabled = false;
    double source_sigma = 0.0;
    double source_window_arcsec = 0.0;
    std::string mode = "default";
};

struct SourceFittingConfig {
    bool active = false;
    double bounding_box_arcsec = 0.0;
    double fitting_radius_arcsec = 0.0;
    bool fit_rotation_angle = false;
    std::array<double, 2> amp_limit_factors{0.0, 0.0};
    std::array<double, 2> fwhm_limit_factors{0.0, 0.0};
};

struct PostProcessingConfig {
    bool map_filtering_enabled = false;
    MapFilterConfig map_filtering;
    int map_histogram_n_bins = 50;
    bool source_finding_enabled = false;
    SourceFindingConfig source_finding;
    SourceFittingConfig source_fitting;
};

inline void set_map_filtering_enabled(PostProcessingConfig &config,
                                      bool enabled) {
    config.map_filtering_enabled = enabled;
    config.map_filtering.enabled = enabled;
}

inline bool map_filtering_active(const PostProcessingConfig &config) {
    return config.map_filtering.enabled;
}

inline void set_source_finding_enabled(PostProcessingConfig &config,
                                       bool enabled) {
    config.source_finding_enabled = enabled;
    config.source_finding.enabled = enabled;
}

inline bool source_finding_active(const PostProcessingConfig &config) {
    return config.source_finding.enabled;
}

inline void set_source_fitting_active(PostProcessingConfig &config,
                                      bool active) {
    config.source_fitting.active = active;
}

inline bool source_fitting_active(const PostProcessingConfig &config) {
    return config.source_fitting.active;
}

inline void validate(const MapFilterEdgeGuardConfig &config,
                     ValidationReport &report) {
    check_minimum(config.hits_core_fraction, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "hits_core_fraction"},
                  report);
    check_minimum(config.guard_radius_fwhm, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "guard_radius_fwhm"},
                  report);
    check_minimum(config.taper_min_fraction, 0.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "taper_min_fraction"},
                  report);
    check_maximum(config.taper_min_fraction, 1.0,
                  {"post_processing", "map_filtering", "edge_guard",
                   "taper_min_fraction"},
                  report);
}

inline void validate(const MapFilterConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    validate(config.edge_guard, report);
    check_minimum(config.denom_rel_tol, 0.0,
                  {"wiener_filter", "denom_rel_tol"}, report);
    check_maximum(config.denom_rel_tol, 1.0,
                  {"wiener_filter", "denom_rel_tol"}, report);
    check_minimum(config.tail_frac_tol, 0.0,
                  {"wiener_filter", "tail_frac_tol"}, report);
    check_maximum(config.tail_frac_tol, 1.0,
                  {"wiener_filter", "tail_frac_tol"}, report);
    check_minimum(config.max_loops, 1, {"wiener_filter", "max_loops"}, report);
    check_minimum(config.denom_check_iters, 0,
                  {"wiener_filter", "denom_check_iters"}, report);
    check_minimum(config.max_denom_iters, 0,
                  {"wiener_filter", "max_denom_iters"}, report);
    for (const auto &[array_name, fwhm_arcsec] : config.template_fwhm_arcsec) {
        if (array_name.empty()) {
            report.add_error({"wiener_filter", "template_fwhm_arcsec"},
                             "array name must not be empty");
        }
        check_greater_than(fwhm_arcsec, 0.0,
                           {"wiener_filter", "template_fwhm_arcsec"},
                           report);
    }
}

inline void validate(const SourceFittingConfig &config, ValidationReport &report) {
    if (!config.active) {
        return;
    }
    check_minimum(config.bounding_box_arcsec, 0.0,
                  {"post_processing", "source_fitting", "bounding_box_arcsec"},
                  report);
}

inline void validate(const SourceFindingConfig &config, ValidationReport &report) {
    if (!config.enabled) {
        return;
    }
    check_minimum(config.source_sigma, 0.0,
                  {"post_processing", "source_finding", "source_sigma"},
                  report);
    check_minimum(config.source_window_arcsec, 0.0,
                  {"post_processing", "source_finding", "source_window_arcsec"},
                  report);
    if (config.mode.empty()) {
        report.add_error({"post_processing", "source_finding", "mode"},
                         "must not be empty");
    }
}

inline void validate(const PostProcessingConfig &config, ValidationReport &report) {
    validate(config.map_filtering, report);
    check_minimum(config.map_histogram_n_bins, 0,
                  {"post_processing", "map_histogram_n_bins"}, report);
    validate(config.source_finding, report);
    validate(config.source_fitting, report);
}

}  // namespace citlali::config
