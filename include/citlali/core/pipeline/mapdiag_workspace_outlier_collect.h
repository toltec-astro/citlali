#pragma once

// Included by mapdiag_workspace.h inside namespace citlali::pipeline.

template <class MapBuffer, class ReductionLearning>
MapdiagOutlierMaskContext make_mapdiag_outlier_mask_context(
    const MapBuffer &mb, const Eigen::ArrayXXd &core_mask,
    const ReductionLearning &reduction_learning, double rad_to_arcsec,
    double fill_double) {
    const auto source_distance =
        mapdiag_source_distance_context(mb, rad_to_arcsec, fill_double);
    const double protect_radius =
        mapdiag_source_protect_radius_arcsec(reduction_learning);
    return {
        source_distance,
        mapdiag_off_source_core_mask(
            core_mask, source_distance, protect_radius)};
}

template <class MapBuffer, class Matrix, class Mask,
          class RobustStats, class ReductionLearning>
auto collect_mapdiag_pixel_candidates_for_map(
    MapBuffer &mb, Eigen::Index map_index, const Matrix &sig2noise,
    const Mask &off_source_core_mask,
    const MapdiagSourceDistanceContext &source_distance_context,
    const RobustStats &robust_stats,
    const ReductionLearning &reduction_learning, double ptc_fs_hz,
    int fill_int, double fill_double) {
    auto candidates = make_mapdiag_pixel_candidates();
    const bool has_contribution_products =
        mapdiag_has_contribution_products(mb, map_index);
    const Eigen::Index n_mapdiag_rows = mapdiag_n_rows(mb);
    const Eigen::Index n_mapdiag_cols = mapdiag_n_cols(mb);
    const double min_effective_samples =
        mapdiag_min_effective_samples(reduction_learning);
    const double min_abs_z = mapdiag_min_abs_z(reduction_learning);

    for (Eigen::Index r = 0; r < n_mapdiag_rows; ++r) {
        for (Eigen::Index c = 0; c < n_mapdiag_cols; ++c) {
            if (!mapdiag_mask_pixel_is_selected(
                    off_source_core_mask, r, c)) {
                continue;
            }

            const double value =
                mapdiag_matrix_double_value(mb->signal[map_index], r, c);
            const double wt =
                mapdiag_matrix_double_value(mb->weight[map_index], r, c);
            const double sn = mapdiag_matrix_double_value(sig2noise, r, c);
            if (!mapdiag_is_valid_outlier_pixel_value(value, wt, sn)) {
                continue;
            }

            const double n_eff = mapdiag_effective_samples_or_fill(
                mb->coverage, map_index, r, c, mb->n_rows, mb->n_cols,
                ptc_fs_hz, fill_double);
            if (!mapdiag_passes_min_effective_samples(
                    n_eff, min_effective_samples)) {
                continue;
            }

            const double z = mapdiag_robust_z(sn, robust_stats);
            if (!mapdiag_passes_min_abs_z(z, min_abs_z)) {
                continue;
            }

            const double source_distance_arcsec =
                mapdiag_source_distance_arcsec(r, c, source_distance_context);
            auto candidate = make_mapdiag_map_pixel_candidate(
                r, c, value, wt, n_eff, z, source_distance_arcsec, fill_int,
                fill_double);

            if (has_contribution_products) {
                const auto contribution_map_index =
                    mapdiag_contribution_map_index(map_index);
                const int uid = mapdiag_matrix_value(
                    mb->contribution_uid[contribution_map_index], r, c);
                const double contrib_signal = mapdiag_matrix_double_value(
                    mb->contribution_signal[contribution_map_index], r, c);
                const double contrib_weight = mapdiag_matrix_double_value(
                    mb->contribution_weight[contribution_map_index], r, c);
                const double contrib_variance_weight =
                    mapdiag_matrix_double_value(
                        mb->contribution_variance_weight[
                            contribution_map_index],
                        r, c);
                if (mapdiag_has_valid_contributor(
                        uid, fill_int, contrib_signal)) {
                    assign_mapdiag_candidate_contributor_from_products(
                        candidate, uid,
                        mb->contribution_scan[contribution_map_index],
                        mb->contribution_sample[contribution_map_index],
                        r, c);
                    const double total_signal = mapdiag_matrix_double_value(
                        mb->contribution_total_signal[contribution_map_index],
                        r, c);
                    const double total_weight = mapdiag_matrix_double_value(
                        mb->contribution_total_weight[contribution_map_index],
                        r, c);
                    const double total_variance_weight =
                        mapdiag_matrix_double_value(
                            mb->contribution_total_variance_weight[
                                contribution_map_index],
                            r, c);
                    const double remaining_weight =
                        mapdiag_remaining_contribution_weight(
                            total_weight, contrib_weight);
                    if (mapdiag_has_full_leave_one_out_inputs(
                            total_signal, total_weight, contrib_weight,
                            contrib_variance_weight, total_variance_weight,
                            remaining_weight)) {
                        const double loo_value =
                            mapdiag_full_leave_one_out_value(
                                total_signal, contrib_signal,
                                remaining_weight);
                        mapdiag_assign_leave_one_out_z(
                            value, wt, loo_value,
                            candidate.leave_one_out_z);
                    }
                    else if (mapdiag_has_fallback_leave_one_out_inputs(
                                 wt, contrib_weight)) {
                        const double raw_sum =
                            mapdiag_raw_weighted_signal(value, wt);
                        const double loo_value =
                            mapdiag_fallback_leave_one_out_value(
                                raw_sum, contrib_signal, wt,
                                contrib_weight);
                        mapdiag_assign_leave_one_out_z(
                            value, wt, loo_value,
                            candidate.leave_one_out_z);
                    }
                }
            }
            append_mapdiag_pixel_candidate(candidates, candidate);
        }
    }
    return candidates;
}

