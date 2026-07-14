#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/pipeline/config_parse_tracking.h>

#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace citlali::pipeline {

template <class Config, class Diagnostics>
void read_post_processing_limit_factor(
    Config &config, const char *name, std::size_t index, double &target,
    Diagnostics &diagnostics) {
    const auto sequence_key =
        std::tuple{"post_processing", "source_fitting", "gauss_model", name};
    const auto path = std::vector<std::string>{
        "post_processing", "source_fitting", "gauss_model", name,
        std::to_string(index)};
    try {
        if (!config.has(sequence_key) ||
            index >= config.get_node(sequence_key).size()) {
            diagnostics.missing_key_paths().push_back(path);
            return;
        }
        target = config.template get_typed<double>(
            std::tuple{"post_processing", "source_fitting", "gauss_model",
                       name, index});
    }
    catch (YAML::TypedBadConversion<double>) {
        diagnostics.invalid_key_paths().push_back(path);
    }
    catch (YAML::InvalidNode) {
        diagnostics.invalid_key_paths().push_back(path);
    }
}

template <class Config, class Key, class Enum, class Parser,
          class Diagnostics>
void read_post_processing_enum(
    Config &config, const Key &key, Enum &target, Parser parser,
    Diagnostics &diagnostics, std::vector<std::string> accepted_values) {
    std::string value{citlali::config::to_string(target)};
    read_config_value_if_clean(
        config, key, value,
        [&target, &parser, &key, &diagnostics](const auto &parsed_value) {
            if (auto parsed = parser(parsed_value)) {
                target = *parsed;
            } else {
                add_invalid_config_key(
                    key, diagnostics.invalid_key_paths());
            }
        },
        diagnostics, std::move(accepted_values));
}

template <class Config, class Key, class Enum, class Parser,
          class Diagnostics>
void read_optional_post_processing_enum(
    Config &config, const Key &key, Enum &target, Parser parser,
    Diagnostics &diagnostics, std::vector<std::string> accepted_values) {
    if (!config.template has_typed<std::string>(key)) {
        return;
    }
    read_post_processing_enum(
        config, key, target, parser, diagnostics,
        std::move(accepted_values));
}

template <class Config, class Diagnostics>
void read_map_filter_request_config(
    Config &config, citlali::config::MapFilterConfig &request,
    Diagnostics &diagnostics) {
    read_config_value(
        config, request.enabled, diagnostics,
        std::tuple{"post_processing", "map_filtering", "enabled"});
    read_post_processing_enum(
        config, std::tuple{"post_processing", "map_filtering", "type"},
        request.type, citlali::config::parse_map_filter_type, diagnostics,
        {"wiener_filter", "convolve", "destripe"});
    read_config_value(
        config, request.normalize_errors, diagnostics,
        std::tuple{
            "post_processing", "map_filtering", "normalize_errors"});

    auto &edge = request.edge_guard;
    read_config_value(
        config, edge.enabled, diagnostics,
        std::tuple{
            "post_processing", "map_filtering", "edge_guard", "enabled"});
    read_config_value(
        config, edge.weight_threshold_mode, diagnostics,
        std::tuple{"post_processing", "map_filtering", "edge_guard",
                   "weight_threshold_mode"},
        {"coverage_cut"});
    read_config_value(
        config, edge.hits_threshold_mode, diagnostics,
        std::tuple{"post_processing", "map_filtering", "edge_guard",
                   "hits_threshold_mode"},
        {"core_median_fraction"});
    read_config_value(
        config, edge.hits_core_fraction, diagnostics,
        std::tuple{"post_processing", "map_filtering", "edge_guard",
                   "hits_core_fraction"},
        {}, {0.0});
    read_config_value(
        config, edge.guard_radius_fwhm, diagnostics,
        std::tuple{"post_processing", "map_filtering", "edge_guard",
                   "guard_radius_fwhm"},
        {}, {0.0});
    read_config_value(
        config, edge.fill_mode, diagnostics,
        std::tuple{
            "post_processing", "map_filtering", "edge_guard", "fill_mode"},
        {"core_median"});
    read_post_processing_enum(
        config,
        std::tuple{
            "post_processing", "map_filtering", "edge_guard", "taper_mode"},
        edge.taper_mode,
        citlali::config::parse_map_filter_edge_taper_mode, diagnostics,
        {"none", "cosine"});
    read_config_value(
        config, edge.taper_min_fraction, diagnostics,
        std::tuple{"post_processing", "map_filtering", "edge_guard",
                   "taper_min_fraction"},
        {}, {0.0}, {1.0});

    read_post_processing_enum(
        config, std::tuple{"wiener_filter", "template_type"},
        request.template_type,
        citlali::config::parse_map_filter_template_type, diagnostics,
        {"kernel", "gaussian", "airy", "highpass"});
    read_optional_post_processing_enum(
        config,
        std::tuple{"wiener_filter", "kernel_template_tail_mode"},
        request.kernel_template_tail_mode,
        citlali::config::parse_map_filter_kernel_tail_mode, diagnostics,
        {"constant", "zero", "cosine"});
    read_config_value(
        config, request.lowpass_only, diagnostics,
        std::tuple{"wiener_filter", "lowpass_only"});
    read_config_value(
        config, request.denom_rel_tol, diagnostics,
        std::tuple{"wiener_filter", "denom_rel_tol"}, {}, {0.0}, {1.0});
    read_config_value(
        config, request.tail_frac_tol, diagnostics,
        std::tuple{"wiener_filter", "tail_frac_tol"}, {}, {0.0}, {1.0});
    read_config_value(
        config, request.max_loops, diagnostics,
        std::tuple{"wiener_filter", "max_loops"}, {}, {1});
    read_config_value(
        config, request.denom_check_iters, diagnostics,
        std::tuple{"wiener_filter", "denom_check_iters"}, {}, {0});
    read_config_value(
        config, request.max_denom_iters, diagnostics,
        std::tuple{"wiener_filter", "max_denom_iters"}, {}, {0});

    request.template_fwhm_arcsec.clear();
    for (const char *array_name : {"a1100", "a1400", "a2000"}) {
        auto &fwhm = request.template_fwhm_arcsec[array_name];
        read_config_value(
            config, fwhm, diagnostics,
            std::tuple{"wiener_filter", "template_fwhm_arcsec", array_name},
            {}, {0.0});
    }
}

template <class Config, class Diagnostics>
void read_source_fitting_request_config(
    Config &config, citlali::config::SourceFittingConfig &request,
    Diagnostics &diagnostics) {
    read_config_value(
        config, request.bounding_box_arcsec, diagnostics,
        std::tuple{
            "post_processing", "source_fitting", "bounding_box_arcsec"},
        {}, {0.0});
    read_config_value(
        config, request.fitting_radius_arcsec, diagnostics,
        std::tuple{
            "post_processing", "source_fitting", "fitting_radius_arcsec"});
    read_post_processing_enum(
        config, std::tuple{"post_processing", "source_fitting", "model"},
        request.model, citlali::config::parse_source_fit_model, diagnostics,
        {"gaussian"});
    read_config_value(
        config, request.fit_rotation_angle, diagnostics,
        std::tuple{"post_processing", "source_fitting", "gauss_model",
                   "fit_rotation_angle"});
    for (std::size_t index = 0; index < 2; ++index) {
        read_post_processing_limit_factor(
            config, "amp_limit_factors", index,
            request.amp_limit_factors[index], diagnostics);
        read_post_processing_limit_factor(
            config, "fwhm_limit_factors", index,
            request.fwhm_limit_factors[index], diagnostics);
    }
}

template <class Config, class Diagnostics>
void read_source_finding_request_config(
    Config &config, citlali::config::SourceFindingConfig &request,
    Diagnostics &diagnostics) {
    read_config_value(
        config, request.enabled, diagnostics,
        std::tuple{"post_processing", "source_finding", "enabled"});
    read_config_value(
        config, request.source_sigma, diagnostics,
        std::tuple{"post_processing", "source_finding", "source_sigma"});
    read_config_value(
        config, request.source_window_arcsec, diagnostics,
        std::tuple{
            "post_processing", "source_finding", "source_window_arcsec"});
    read_config_value(
        config, request.mode, diagnostics,
        std::tuple{"post_processing", "source_finding", "mode"},
        {"default", "negative", "both"});
}

template <class Config, class Diagnostics>
void read_post_processing_request_config(
    Config &config, citlali::config::PostProcessingConfig &request,
    Diagnostics &diagnostics) {
    read_map_filter_request_config(
        config, request.map_filtering, diagnostics);
    request.map_filtering_enabled = request.map_filtering.enabled;
    read_config_value(
        config, request.map_histogram_n_bins, diagnostics,
        std::tuple{"post_processing", "map_histogram_n_bins"}, {}, {0});
    read_source_finding_request_config(
        config, request.source_finding, diagnostics);
    request.source_finding_enabled = request.source_finding.enabled;
    read_source_fitting_request_config(
        config, request.source_fitting, diagnostics);
}

}  // namespace citlali::pipeline
