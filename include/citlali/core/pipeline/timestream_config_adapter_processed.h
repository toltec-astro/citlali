#pragma once

#include <citlali/core/config/timestream_config.h>

#include <Eigen/Core>

#include <cstdint>
#include <string>
#include <utility>

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

template <class PtcProc, class ArrayNameMap>
void apply_processed_clean_config_to_processor(
    const citlali::config::ProcessedTimeChunkCleanConfig &config,
    const ArrayNameMap &array_name_map, PtcProc &ptcproc) {
    ptcproc.run_clean = config.enabled;
    if (!config.enabled) {
        return;
    }

    auto &cleaner = ptcproc.cleaner;
    cleaner.grouping = config.grouping;
    ptcproc.mask_radius_arcsec = config.mask_radius_arcsec;
    cleaner.tau = config.tau;
    cleaner.standard_pca.enabled = config.standard_pca.enabled;
    cleaner.stddev_limit = config.standard_pca.stddev_limit;
    cleaner.n_calc = config.standard_pca.n_calc;
    cleaner.n_eig_to_cut.clear();
    for (const auto &[array_id, array_name] : array_name_map) {
        const auto it = config.standard_pca.n_eig_to_cut.find(array_name);
        if (it == config.standard_pca.n_eig_to_cut.end()) {
            continue;
        }
        Eigen::Matrix<Eigen::Index, Eigen::Dynamic, 1> values(
            static_cast<Eigen::Index>(it->second.size()));
        for (std::size_t i = 0; i < it->second.size(); ++i) {
            values(static_cast<Eigen::Index>(i)) = it->second[i];
        }
        cleaner.n_eig_to_cut[array_id] = std::move(values);
    }

    cleaner.corr_grouping.enabled = config.corr_grouping.enabled;
    cleaner.corr_grouping.metric = std::string{
        citlali::config::to_string(config.corr_grouping.metric)};
    cleaner.corr_grouping.corr_min = config.corr_grouping.corr_min;
    cleaner.corr_grouping.min_overlap = config.corr_grouping.min_overlap;
    cleaner.corr_grouping.min_good_frac = config.corr_grouping.min_good_frac;
    cleaner.corr_grouping.min_group_size = config.corr_grouping.min_group_size;
    cleaner.corr_grouping.max_samples = config.corr_grouping.max_samples;
    cleaner.corr_grouping.clean_residual = config.corr_grouping.clean_residual;

    cleaner.null_model.enabled = config.null_model.enabled;
    cleaner.null_model.n_surrogates = config.null_model.n_surrogates;
    cleaner.null_model.quantile = config.null_model.quantile;
    cleaner.null_model.min_good_frac = config.null_model.min_good_frac;
    cleaner.null_model.max_modes = config.null_model.max_modes;
    cleaner.null_model.max_samples = config.null_model.max_samples;
    cleaner.null_model.seed =
        static_cast<std::uint32_t>(config.null_model.seed);
    cleaner.null_model.grouping = config.null_model.grouping;

    cleaner.marchenko_pastur.enabled = config.marchenko_pastur.enabled;
    cleaner.marchenko_pastur.min_good_frac =
        config.marchenko_pastur.min_good_frac;
    cleaner.marchenko_pastur.max_modes = config.marchenko_pastur.max_modes;
    cleaner.marchenko_pastur.max_samples = config.marchenko_pastur.max_samples;
    cleaner.marchenko_pastur.band_low_Hz = config.marchenko_pastur.band_low_Hz;
    cleaner.marchenko_pastur.band_high_Hz =
        config.marchenko_pastur.band_high_Hz;
    cleaner.marchenko_pastur.clip_z = config.marchenko_pastur.clip_z;
    cleaner.marchenko_pastur.bulk_keep_frac =
        config.marchenko_pastur.bulk_keep_frac;
    cleaner.marchenko_pastur.q_grid_size = config.marchenko_pastur.q_grid_size;
    cleaner.marchenko_pastur.grouping = config.marchenko_pastur.grouping;

    cleaner.adaptive_selector.enabled = config.adaptive_selector.enabled;
    cleaner.adaptive_selector.min_good_frac =
        config.adaptive_selector.min_good_frac;
    cleaner.adaptive_selector.max_det = config.adaptive_selector.max_det;
    cleaner.adaptive_selector.max_samples = config.adaptive_selector.max_samples;
    cleaner.adaptive_selector.max_pairs = config.adaptive_selector.max_pairs;
    cleaner.adaptive_selector.seed =
        static_cast<std::uint32_t>(config.adaptive_selector.seed);
    cleaner.adaptive_selector.clip_z = config.adaptive_selector.clip_z;
    cleaner.adaptive_selector.low_weight = config.adaptive_selector.low_weight;
    cleaner.adaptive_selector.tail_weight = config.adaptive_selector.tail_weight;
    cleaner.adaptive_selector.topmode_weight =
        config.adaptive_selector.topmode_weight;
    cleaner.adaptive_selector.reg_weight = config.adaptive_selector.reg_weight;
    cleaner.adaptive_selector.low_band_Hz = config.adaptive_selector.low_band_Hz;
    cleaner.adaptive_selector.mid_band_Hz = config.adaptive_selector.mid_band_Hz;
    cleaner.adaptive_selector.candidate_offsets =
        config.adaptive_selector.candidate_offsets;
    cleaner.adaptive_selector.grouping = config.adaptive_selector.grouping;
    cleaner.adaptive_selector.log_candidates =
        config.adaptive_selector.log_candidates;
}

}  // namespace citlali::pipeline
