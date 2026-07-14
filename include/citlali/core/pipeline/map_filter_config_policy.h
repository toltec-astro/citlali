#pragma once

#include <citlali/core/config/noise_config.h>
#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>

#include <cstdlib>
#include <string>

namespace citlali::pipeline {

template <class WienerFilter>
void adapt_map_filter_config_one_way(
    const citlali::config::MapFilterConfig &config,
    double arcsec_to_rad, WienerFilter &wiener_filter) {
    wiener_filter.filter_type =
        std::string{citlali::config::to_string(config.type)};
    wiener_filter.template_type =
        std::string{citlali::config::to_string(config.template_type)};
    wiener_filter.kernel_template_tail_mode = std::string{
        citlali::config::to_string(config.kernel_template_tail_mode)};
    wiener_filter.run_lowpass = config.lowpass_only;
    wiener_filter.normalize_error = config.normalize_errors;
    wiener_filter.edge_guard_enabled = config.edge_guard.enabled;
    wiener_filter.edge_weight_threshold_mode =
        config.edge_guard.weight_threshold_mode;
    wiener_filter.edge_hits_threshold_mode =
        config.edge_guard.hits_threshold_mode;
    wiener_filter.edge_hits_core_fraction =
        config.edge_guard.hits_core_fraction;
    wiener_filter.edge_guard_radius_fwhm =
        config.edge_guard.guard_radius_fwhm;
    wiener_filter.edge_fill_mode = config.edge_guard.fill_mode;
    wiener_filter.edge_taper_mode =
        std::string{citlali::config::to_string(config.edge_guard.taper_mode)};
    wiener_filter.edge_taper_min_fraction =
        config.edge_guard.taper_min_fraction;
    wiener_filter.denom_rel_tol = config.denom_rel_tol;
    wiener_filter.tail_frac_tol = config.tail_frac_tol;
    wiener_filter.max_loops = config.max_loops;
    wiener_filter.denom_check_iters = config.denom_check_iters;
    wiener_filter.max_denom_iters = config.max_denom_iters;
    wiener_filter.template_fwhm_rad.clear();
    if (citlali::config::map_filter_template_uses_fwhm(
            config.template_type)) {
        for (const auto &[array_name, fwhm_arcsec] :
             config.template_fwhm_arcsec) {
            wiener_filter.template_fwhm_rad[array_name] =
                fwhm_arcsec * arcsec_to_rad;
        }
    }
}

template <class NoiseConfig, class WienerFilter,
          class RuntimeTimestreamProc, class MapFitter,
          class Logger>
void apply_map_filter_runtime_policy(
    const NoiseConfig &noise_config,
    const citlali::config::MapFilterConfig &map_filter_config,
    const RuntimeTimestreamProc &rtcproc, const MapFitter &map_fitter,
    const std::string &parallel_policy, WienerFilter &wiener_filter,
    const Logger &logger) {
    if (map_filter_config.template_type ==
        citlali::config::MapFilterTemplateType::kernel) {
        if (!rtcproc.run_kernel) {
            logger->error("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        wiener_filter.map_fitter = map_fitter;
    }

    if (!citlali::config::noise_maps_active(noise_config) &&
        (!map_filter_config.lowpass_only &&
         map_filter_config.type ==
             citlali::config::MapFilterType::wiener_filter)) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    wiener_filter.parallel_policy = parallel_policy;
}

}  // namespace citlali::pipeline
