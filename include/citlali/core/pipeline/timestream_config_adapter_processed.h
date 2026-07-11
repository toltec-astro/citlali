#pragma once

#include <citlali/core/config/timestream_config.h>

#include <Eigen/Core>

#include <string>

namespace citlali::pipeline {

template <class PtcProc>
void apply_fruit_loops_config_to_processor(
    const citlali::config::TimestreamFruitLoopsConfig &config,
    PtcProc &ptcproc) {
    ptcproc.run_fruit_loops = config.enabled;
    ptcproc.fruit_loops_recompute_weights_after_addback =
        config.recompute_weights_after_addback;
    if (!config.enabled) {
        return;
    }

    ptcproc.save_all_iters = config.save_all_iters;
    ptcproc.fruit_loops_path = config.path;
    ptcproc.fruit_loops_type = config.type;
    ptcproc.fruit_mode =
        std::string{citlali::config::to_string(config.mode)};
    ptcproc.fruit_loops_sig2noise = config.sig2noise_limit;
    ptcproc.fruit_loops_flux.resize(
        static_cast<Eigen::Index>(config.array_flux_limit.size()));
    for (std::size_t i = 0; i < config.array_flux_limit.size(); ++i) {
        ptcproc.fruit_loops_flux(static_cast<Eigen::Index>(i)) =
            config.array_flux_limit[i];
    }
    ptcproc.fruit_loops_peak_fraction_limit = config.peak_fraction_limit;
    ptcproc.fruit_loops_local_snr_floor = config.local_snr_floor;
    ptcproc.fruit_loops_local_sigma_inner_radius_arcsec =
        config.local_sigma_inner_radius_arcsec;
    ptcproc.fruit_loops_local_sigma_outer_radius_arcsec =
        config.local_sigma_outer_radius_arcsec;
    ptcproc.fruit_loops_local_sigma_inner_fwhm =
        config.local_sigma_inner_fwhm;
    ptcproc.fruit_loops_local_sigma_outer_fwhm =
        config.local_sigma_outer_fwhm;
    ptcproc.fruit_loops_local_sigma_edge_guard_arcsec =
        config.local_sigma_edge_guard_arcsec;
    ptcproc.fruit_loops_local_sigma_min_pixels =
        config.local_sigma_min_pixels;
    ptcproc.fruit_loops_adaptive_support_radius_arcsec =
        config.adaptive_support_radius_arcsec;
    ptcproc.fruit_loops_adaptive_support_radius_fwhm =
        config.adaptive_support_radius_fwhm;
    ptcproc.fruit_loops_weight_feedback_enabled =
        config.weight_feedback.enabled;
    ptcproc.fruit_loops_weight_feedback_reference = std::string{
        citlali::config::to_string(config.weight_feedback.reference)};
    ptcproc.fruit_loops_weight_feedback_low_relative_weight =
        config.weight_feedback.low_relative_weight;
    ptcproc.fruit_loops_weight_feedback_high_relative_weight =
        config.weight_feedback.high_relative_weight;
    ptcproc.fruit_loops_center_keep_radius_arcsec =
        config.center_keep_radius_arcsec;
    ptcproc.fruit_loops_interp_mode_override = std::string{
        citlali::config::to_string(config.interp_mode_override)};
    ptcproc.fruit_loops_legacy_center = config.legacy_center;
    ptcproc.fruit_loops_iters = config.max_iters;
}

}  // namespace citlali::pipeline
