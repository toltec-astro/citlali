#pragma once

#include <citlali/core/engine/detail/config_parse_tracking.h>
#include <citlali/core/pipeline/map_source_finding.h>

#include <Eigen/Core>

#include <cstddef>
#include <set>
#include <string>
#include <tuple>
#include <vector>

namespace citlali::engine_detail {

inline const std::vector<std::string> &interface_sync_offset_keys() {
    static const std::vector<std::string> keys = {
        "toltec0",  "toltec1", "toltec2", "toltec3", "toltec4",
        "toltec5",  "toltec6", "toltec7", "toltec8", "toltec9",
        "toltec10", "toltec11", "toltec12", "hwpr"};
    return keys;
}

template <class Config, class OffsetMap, class Logger>
void read_interface_sync_offsets(
    Config &config, OffsetMap &interface_sync_offset, const Logger &logger) {
    const auto &interface_keys = interface_sync_offset_keys();
    for (const auto &key : interface_keys) {
        interface_sync_offset[key] = 0.0;
    }

    if (!config.has(std::tuple{"interface_sync_offset"})) {
        return;
    }

    auto interface_node = config.get_node(std::tuple{"interface_sync_offset"});
    std::set<std::string> configured_keys;
    for (Eigen::Index i = 0; i < interface_node.size(); ++i) {
        bool found_key = false;
        for (const auto &key : interface_keys) {
            if (config.has(std::tuple{"interface_sync_offset", i, key})) {
                auto offset = config.template get_typed<double>(
                    std::tuple{"interface_sync_offset", i, key});
                if (configured_keys.find(key) != configured_keys.end()) {
                    logger->warn(
                        "interface_sync_offset for {} specified multiple times; "
                        "using last value",
                        key);
                }
                interface_sync_offset[key] = offset;
                configured_keys.insert(key);
                found_key = true;
            }
        }
        if (!found_key) {
            logger->warn(
                "interface_sync_offset entry {} does not contain a recognized "
                "interface key; ignoring entry",
                i);
        }
    }

    for (const auto &key : interface_keys) {
        if (configured_keys.find(key) == configured_keys.end()) {
            logger->warn("interface_sync_offset missing {}; using 0.0 s", key);
        }
    }
}

template <class Config, class KeyList, class PostProcessingConfig>
void read_post_processing_activation_config(
    Config &config, bool &run_map_filter, bool &run_source_finder,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(config, run_map_filter, missing_keys, invalid_keys,
                           std::tuple{"post_processing", "map_filtering", "enabled"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.map_filtering_enabled = run_map_filter;
            typed_post_processing_config.map_filtering.enabled = run_map_filter;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(config, run_source_finder, missing_keys, invalid_keys,
                           std::tuple{"post_processing", "source_finding", "enabled"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_finding_enabled = run_source_finder;
            typed_post_processing_config.source_finding.enabled = run_source_finder;
        }
    }
}

template <class Config, class MapFitter, class PostProcessingConfig,
          class KeyList>
void read_source_fitting_config(
    Config &config, const std::string &reduction_type, bool run_map_filter,
    bool run_source_finder, MapFitter &map_fitter, double pixel_size_rad,
    double arcsec_to_rad, PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    if (!citlali::pipeline::source_fitting_config_needed(
            reduction_type, run_map_filter, run_source_finder)) {
        return;
    }

    typed_post_processing_config.source_fitting.active = true;

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.bounding_box_pix, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "bounding_box_arcsec"},
            {}, {0});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.bounding_box_arcsec =
                map_fitter.bounding_box_pix;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.fitting_region_pix, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "fitting_radius_arcsec"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.fitting_radius_arcsec =
                map_fitter.fitting_region_pix;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, map_fitter.fit_angle, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_fitting", "gauss_model",
                       "fit_rotation_angle"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_fitting.fit_rotation_angle =
                map_fitter.fit_angle;
        }
    }

    map_fitter.bounding_box_pix =
        citlali::pipeline::source_fitting_arcsec_to_pixels(
            map_fitter.bounding_box_pix, arcsec_to_rad, pixel_size_rad);
    map_fitter.fitting_region_pix =
        citlali::pipeline::source_fitting_arcsec_to_pixels(
            map_fitter.fitting_region_pix, arcsec_to_rad, pixel_size_rad);

    map_fitter.flux_limits.resize(2);
    map_fitter.fwhm_limits.resize(2);
    for (Eigen::Index i = 0; i < map_fitter.flux_limits.size(); ++i) {
        map_fitter.flux_limits(i) =
            config.template get_typed<double>(
                std::tuple{"post_processing", "source_fitting", "gauss_model",
                           "amp_limit_factors", i});
        typed_post_processing_config.source_fitting
            .amp_limit_factors[static_cast<std::size_t>(i)] =
            map_fitter.flux_limits(i);

        map_fitter.fwhm_limits(i) =
            config.template get_typed<double>(
                std::tuple{"post_processing", "source_fitting", "gauss_model",
                           "fwhm_limit_factors", i});
        typed_post_processing_config.source_fitting
            .fwhm_limit_factors[static_cast<std::size_t>(i)] =
            map_fitter.fwhm_limits(i);
    }

    citlali::pipeline::apply_positive_source_fit_limits(map_fitter);
}

template <class Config, class ObservationMapBuffer, class CoaddMapBuffer,
          class PostProcessingConfig, class KeyList>
void read_source_finding_config(
    Config &config, bool run_source_finder, ObservationMapBuffer &omb,
    CoaddMapBuffer &cmb, bool run_coadd, double arcsec_to_rad,
    PostProcessingConfig &typed_post_processing_config,
    KeyList &missing_keys, KeyList &invalid_keys) {
    if (!run_source_finder) {
        return;
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, omb.source_sigma, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_finding", "source_sigma"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_finding.source_sigma =
                omb.source_sigma;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, omb.source_window_rad, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_finding", "source_window_arcsec"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_finding.source_window_arcsec =
                omb.source_window_rad;
        }
    }

    {
        const auto missing_before = missing_keys.size();
        const auto invalid_before = invalid_keys.size();
        ::get_config_value(
            config, omb.source_finder_mode, missing_keys, invalid_keys,
            std::tuple{"post_processing", "source_finding", "mode"});
        if (config_parse_clean(
                missing_keys, invalid_keys, missing_before, invalid_before)) {
            typed_post_processing_config.source_finding.mode =
                omb.source_finder_mode;
        }
    }

    omb.source_window_rad =
        citlali::pipeline::source_window_arcsec_to_rad(
            omb.source_window_rad, arcsec_to_rad);

    citlali::pipeline::mirror_source_finding_config_to_coadd(
        omb, cmb, run_coadd);
}

}  // namespace citlali::engine_detail
