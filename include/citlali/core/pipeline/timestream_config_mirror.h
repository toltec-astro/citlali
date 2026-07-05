#pragma once

#include <citlali/core/config/timestream_config.h>

#include <cstddef>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class FruitLoopsConfig, class PtcProc>
void mirror_fruit_loops_config(FruitLoopsConfig &target,
                               const PtcProc &ptcproc) {
    target.enabled = ptcproc.run_fruit_loops;
    if (!ptcproc.run_fruit_loops) {
        return;
    }

    target.save_all_iters = ptcproc.save_all_iters;
    target.path = ptcproc.fruit_loops_path;
    target.type = ptcproc.fruit_loops_type;
    if (auto parsed =
            citlali::config::parse_fruit_loops_mode(ptcproc.fruit_mode)) {
        target.mode = *parsed;
    }
    target.sig2noise_limit = ptcproc.fruit_loops_sig2noise;
    target.array_flux_limit.clear();
    target.array_flux_limit.reserve(
        static_cast<std::size_t>(ptcproc.fruit_loops_flux.size()));
    for (Eigen::Index i = 0; i < ptcproc.fruit_loops_flux.size(); ++i) {
        target.array_flux_limit.push_back(ptcproc.fruit_loops_flux(i));
    }
    target.peak_fraction_limit = ptcproc.fruit_loops_peak_fraction_limit;
    target.local_snr_floor = ptcproc.fruit_loops_local_snr_floor;
    target.local_sigma_inner_radius_arcsec =
        ptcproc.fruit_loops_local_sigma_inner_radius_arcsec;
    target.local_sigma_outer_radius_arcsec =
        ptcproc.fruit_loops_local_sigma_outer_radius_arcsec;
    target.local_sigma_inner_fwhm = ptcproc.fruit_loops_local_sigma_inner_fwhm;
    target.local_sigma_outer_fwhm = ptcproc.fruit_loops_local_sigma_outer_fwhm;
    target.local_sigma_edge_guard_arcsec =
        ptcproc.fruit_loops_local_sigma_edge_guard_arcsec;
    target.local_sigma_min_pixels = ptcproc.fruit_loops_local_sigma_min_pixels;
    target.adaptive_support_radius_arcsec =
        ptcproc.fruit_loops_adaptive_support_radius_arcsec;
    target.adaptive_support_radius_fwhm =
        ptcproc.fruit_loops_adaptive_support_radius_fwhm;
    target.weight_feedback.enabled =
        ptcproc.fruit_loops_weight_feedback_enabled;
    if (auto parsed =
            citlali::config::parse_fruit_loops_weight_feedback_reference(
                ptcproc.fruit_loops_weight_feedback_reference)) {
        target.weight_feedback.reference = *parsed;
    }
    target.weight_feedback.low_relative_weight =
        ptcproc.fruit_loops_weight_feedback_low_relative_weight;
    target.weight_feedback.high_relative_weight =
        ptcproc.fruit_loops_weight_feedback_high_relative_weight;
    target.center_keep_radius_arcsec = ptcproc.fruit_loops_center_keep_radius_arcsec;
    if (auto parsed =
            citlali::config::parse_fruit_loops_interp_mode_override(
                ptcproc.fruit_loops_interp_mode_override)) {
        target.interp_mode_override = *parsed;
    }
    target.legacy_center = ptcproc.fruit_loops_legacy_center;
    target.recompute_weights_after_addback =
        ptcproc.fruit_loops_recompute_weights_after_addback;
    target.max_iters = ptcproc.fruit_loops_iters;
}

}  // namespace citlali::pipeline
