#pragma once

// Included by timestream_config_mirror.h inside namespace citlali::pipeline.

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

template <class CleanConfig, class PtcProc, class ArrayNameMap>
void mirror_processed_clean_config(CleanConfig &target, const PtcProc &ptcproc,
                                   const ArrayNameMap &array_name_map) {
    target.enabled = ptcproc.run_clean;
    if (!ptcproc.run_clean) {
        return;
    }

    if (auto parsed = citlali::config::parse_processed_cleaner_mode(
            ptcproc.cleaner.active_cleaner_label())) {
        target.active = *parsed;
    }
    target.grouping = ptcproc.cleaner.grouping;
    target.mask_radius_arcsec = ptcproc.mask_radius_arcsec;
    target.tau = ptcproc.cleaner.tau;
    target.standard_pca.enabled = ptcproc.cleaner.standard_pca.enabled;
    target.standard_pca.stddev_limit = ptcproc.cleaner.stddev_limit;
    target.standard_pca.n_calc = ptcproc.cleaner.n_calc;
    target.standard_pca.n_eig_to_cut.clear();
    for (const auto &[arr_index, arr_name] : array_name_map) {
        const auto it = ptcproc.cleaner.n_eig_to_cut.find(arr_index);
        if (it == ptcproc.cleaner.n_eig_to_cut.end()) {
            continue;
        }
        std::vector<int> n_eig_to_cut;
        n_eig_to_cut.reserve(static_cast<std::size_t>(it->second.size()));
        for (Eigen::Index i = 0; i < it->second.size(); ++i) {
            n_eig_to_cut.push_back(static_cast<int>(it->second(i)));
        }
        target.standard_pca.n_eig_to_cut[arr_name] =
            std::move(n_eig_to_cut);
    }

    auto &typed_corr_grouping = target.corr_grouping;
    typed_corr_grouping.enabled = ptcproc.cleaner.corr_grouping.enabled;
    if (auto parsed = citlali::config::parse_processed_corr_grouping_metric(
            ptcproc.cleaner.corr_grouping.metric)) {
        typed_corr_grouping.metric = *parsed;
    }
    typed_corr_grouping.corr_min = ptcproc.cleaner.corr_grouping.corr_min;
    typed_corr_grouping.min_overlap = ptcproc.cleaner.corr_grouping.min_overlap;
    typed_corr_grouping.min_good_frac =
        ptcproc.cleaner.corr_grouping.min_good_frac;
    typed_corr_grouping.min_group_size =
        ptcproc.cleaner.corr_grouping.min_group_size;
    typed_corr_grouping.max_samples = ptcproc.cleaner.corr_grouping.max_samples;
    typed_corr_grouping.clean_residual =
        ptcproc.cleaner.corr_grouping.clean_residual;

    auto &typed_null_model = target.null_model;
    typed_null_model.enabled = ptcproc.cleaner.null_model.enabled;
    typed_null_model.n_surrogates = ptcproc.cleaner.null_model.n_surrogates;
    typed_null_model.quantile = ptcproc.cleaner.null_model.quantile;
    typed_null_model.min_good_frac = ptcproc.cleaner.null_model.min_good_frac;
    typed_null_model.max_modes = ptcproc.cleaner.null_model.max_modes;
    typed_null_model.max_samples = ptcproc.cleaner.null_model.max_samples;
    typed_null_model.seed = static_cast<int>(ptcproc.cleaner.null_model.seed);
    typed_null_model.grouping = ptcproc.cleaner.null_model.grouping;

    auto &typed_mp = target.marchenko_pastur;
    typed_mp.enabled = ptcproc.cleaner.marchenko_pastur.enabled;
    typed_mp.min_good_frac =
        ptcproc.cleaner.marchenko_pastur.min_good_frac;
    typed_mp.max_modes = ptcproc.cleaner.marchenko_pastur.max_modes;
    typed_mp.max_samples = ptcproc.cleaner.marchenko_pastur.max_samples;
    typed_mp.band_low_Hz = ptcproc.cleaner.marchenko_pastur.band_low_Hz;
    typed_mp.band_high_Hz = ptcproc.cleaner.marchenko_pastur.band_high_Hz;
    typed_mp.clip_z = ptcproc.cleaner.marchenko_pastur.clip_z;
    typed_mp.bulk_keep_frac =
        ptcproc.cleaner.marchenko_pastur.bulk_keep_frac;
    typed_mp.q_grid_size = ptcproc.cleaner.marchenko_pastur.q_grid_size;
    typed_mp.grouping = ptcproc.cleaner.marchenko_pastur.grouping;

    auto &typed_adaptive = target.adaptive_selector;
    typed_adaptive.enabled = ptcproc.cleaner.adaptive_selector.enabled;
    typed_adaptive.min_good_frac =
        ptcproc.cleaner.adaptive_selector.min_good_frac;
    typed_adaptive.max_det = ptcproc.cleaner.adaptive_selector.max_det;
    typed_adaptive.max_samples =
        ptcproc.cleaner.adaptive_selector.max_samples;
    typed_adaptive.max_pairs = ptcproc.cleaner.adaptive_selector.max_pairs;
    typed_adaptive.seed =
        static_cast<int>(ptcproc.cleaner.adaptive_selector.seed);
    typed_adaptive.clip_z = ptcproc.cleaner.adaptive_selector.clip_z;
    typed_adaptive.low_weight = ptcproc.cleaner.adaptive_selector.low_weight;
    typed_adaptive.tail_weight = ptcproc.cleaner.adaptive_selector.tail_weight;
    typed_adaptive.topmode_weight =
        ptcproc.cleaner.adaptive_selector.topmode_weight;
    typed_adaptive.reg_weight = ptcproc.cleaner.adaptive_selector.reg_weight;
    typed_adaptive.low_band_Hz =
        ptcproc.cleaner.adaptive_selector.low_band_Hz;
    typed_adaptive.mid_band_Hz =
        ptcproc.cleaner.adaptive_selector.mid_band_Hz;
    typed_adaptive.candidate_offsets =
        ptcproc.cleaner.adaptive_selector.candidate_offsets;
    typed_adaptive.grouping = ptcproc.cleaner.adaptive_selector.grouping;
    typed_adaptive.log_candidates =
        ptcproc.cleaner.adaptive_selector.log_candidates;
}

