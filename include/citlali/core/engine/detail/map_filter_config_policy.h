#pragma once

#include <citlali/core/config/post_processing_config.h>
#include <citlali/core/config/runtime_config.h>

#include <cstdlib>
#include <string>

namespace citlali::engine_detail {

template <class WienerFilter, class PostProcessingConfig>
void mirror_wiener_filter_config(
    const WienerFilter &wiener_filter, bool run_map_filter,
    double rad_to_arcsec, PostProcessingConfig &typed_post_processing_config) {
    typed_post_processing_config.map_filtering_enabled = run_map_filter;
    auto &typed_map_filter = typed_post_processing_config.map_filtering;
    typed_map_filter.enabled = run_map_filter;
    if (auto parsed =
            citlali::config::parse_map_filter_type(wiener_filter.filter_type)) {
        typed_map_filter.type = *parsed;
    }
    if (auto parsed = citlali::config::parse_map_filter_template_type(
            wiener_filter.template_type)) {
        typed_map_filter.template_type = *parsed;
    }
    typed_map_filter.lowpass_only = wiener_filter.run_lowpass;
    typed_map_filter.normalize_errors = wiener_filter.normalize_error;
    typed_map_filter.edge_guard.enabled = wiener_filter.edge_guard_enabled;
    typed_map_filter.edge_guard.weight_threshold_mode =
        wiener_filter.edge_weight_threshold_mode;
    typed_map_filter.edge_guard.hits_threshold_mode =
        wiener_filter.edge_hits_threshold_mode;
    typed_map_filter.edge_guard.hits_core_fraction =
        wiener_filter.edge_hits_core_fraction;
    typed_map_filter.edge_guard.guard_radius_fwhm =
        wiener_filter.edge_guard_radius_fwhm;
    typed_map_filter.edge_guard.fill_mode = wiener_filter.edge_fill_mode;
    if (auto parsed = citlali::config::parse_map_filter_edge_taper_mode(
            wiener_filter.edge_taper_mode)) {
        typed_map_filter.edge_guard.taper_mode = *parsed;
    }
    typed_map_filter.edge_guard.taper_min_fraction =
        wiener_filter.edge_taper_min_fraction;
    typed_map_filter.denom_rel_tol = wiener_filter.denom_rel_tol;
    typed_map_filter.tail_frac_tol = wiener_filter.tail_frac_tol;
    typed_map_filter.max_loops = wiener_filter.max_loops;
    typed_map_filter.denom_check_iters = wiener_filter.denom_check_iters;
    typed_map_filter.max_denom_iters = wiener_filter.max_denom_iters;
    typed_map_filter.template_fwhm_arcsec.clear();
    for (const auto &[array_name, fwhm_rad] :
         wiener_filter.template_fwhm_rad) {
        typed_map_filter.template_fwhm_arcsec[array_name] =
            fwhm_rad * rad_to_arcsec;
    }
}

template <class WienerFilter, class RuntimeTimestreamProc, class MapFitter,
          class Logger>
void apply_map_filter_runtime_policy(
    citlali::config::ReductionType reduction_type, bool run_noise,
    const RuntimeTimestreamProc &rtcproc, const MapFitter &map_fitter,
    const std::string &parallel_policy, WienerFilter &wiener_filter,
    bool &write_filtered_maps_partial, const Logger &logger) {
    write_filtered_maps_partial =
        reduction_type == citlali::config::ReductionType::science;

    if (wiener_filter.template_type == "kernel") {
        if (!rtcproc.run_kernel) {
            logger->error("wiener filter kernel template requires kernel");
            std::exit(EXIT_FAILURE);
        }
        wiener_filter.map_fitter = map_fitter;
    }

    if (!run_noise &&
        (!wiener_filter.run_lowpass &&
         wiener_filter.filter_type == "wiener_filter")) {
        logger->error("wiener filter requires noise maps");
        std::exit(EXIT_FAILURE);
    }

    wiener_filter.parallel_policy = parallel_policy;
}

}  // namespace citlali::engine_detail
