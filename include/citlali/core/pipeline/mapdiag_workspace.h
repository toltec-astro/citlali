#pragma once

#include <citlali/core/pipeline/mapdiag_edge_guard.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_netcdf.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/pipeline/mapdiag_stats.h>

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

struct MapdiagMapWorkspace {
    explicit MapdiagMapWorkspace(std::size_t n_maps, double fill_double,
                                 int fill_int)
        : label_storage{n_maps},
          median_err(n_maps, fill_double),
          median_rms(n_maps, fill_double),
          weight_thresholds(n_maps, fill_double),
          weight_sum(n_maps, fill_double),
          core_weight_sum(n_maps, fill_double),
          coverage_sum(n_maps, fill_double),
          coverage_max(n_maps, fill_double),
          coverage_median_core(n_maps, fill_double),
          coverage_refs{coverage_sum, coverage_max, coverage_median_core},
          empirical_to_formal_noise_ratio(n_maps, fill_double),
          formal_noise_refs{
              median_err, median_rms, empirical_to_formal_noise_ratio},
          noise_weight_median_ratio(n_maps, fill_double),
          noise_weight_scale(n_maps, fill_double),
          noise_products_s2n_sigma(n_maps, fill_double),
          noise_products_valid_pixels(n_maps, fill_double),
          noise_product_refs{
              noise_weight_median_ratio, noise_weight_scale,
              noise_products_s2n_sigma, noise_products_valid_pixels},
          peak_signal(n_maps, fill_double),
          peak_abs_sig2noise(n_maps, fill_double),
          core_peak_abs_sig2noise(n_maps, fill_double),
          noise_rms_p16(n_maps, fill_double),
          noise_rms_p84(n_maps, fill_double),
          core_tail_frac_abs3(n_maps, fill_double),
          core_tail_frac_pos3(n_maps, fill_double),
          core_tail_frac_neg3(n_maps, fill_double),
          core_tail_excess_abs3(n_maps, fill_double),
          core_tail_excess_pos3(n_maps, fill_double),
          core_tail_excess_neg3(n_maps, fill_double),
          core_sig2noise_skew(n_maps, fill_double),
          core_tail_refs{
              core_tail_frac_abs3, core_tail_frac_pos3,
              core_tail_frac_neg3, core_tail_excess_abs3,
              core_tail_excess_pos3, core_tail_excess_neg3,
              core_sig2noise_skew},
          noise_tail_frac_abs3(n_maps, fill_double),
          noise_tail_frac_pos3(n_maps, fill_double),
          noise_tail_frac_neg3(n_maps, fill_double),
          noise_tail_excess_abs3(n_maps, fill_double),
          noise_tail_excess_pos3(n_maps, fill_double),
          noise_tail_excess_neg3(n_maps, fill_double),
          noise_sig2noise_skew(n_maps, fill_double),
          noise_tail_refs{
              noise_rms_p16, noise_rms_p84, noise_tail_frac_abs3,
              noise_tail_frac_pos3, noise_tail_frac_neg3,
              noise_tail_excess_abs3, noise_tail_excess_pos3,
              noise_tail_excess_neg3, noise_sig2noise_skew},
          edge_guard_weight_thresholds(n_maps, fill_double),
          edge_guard_hits_thresholds(n_maps, fill_double),
          edge_guard_background_levels(n_maps, fill_double),
          edge_guard_science_frac(n_maps, fill_double),
          edge_guard_support_frac(n_maps, fill_double),
          edge_guard_guardband_rms_pre(n_maps, fill_double),
          edge_guard_guardband_rms_post(n_maps, fill_double),
          edge_guard_exterior_rms_pre(n_maps, fill_double),
          edge_guard_exterior_rms_post(n_maps, fill_double),
          edge_guard_exterior_max_abs_pre(n_maps, fill_double),
          edge_guard_exterior_max_abs_post(n_maps, fill_double),
          edge_guard_double_refs{
              edge_guard_weight_thresholds, edge_guard_hits_thresholds,
              edge_guard_background_levels, edge_guard_science_frac,
              edge_guard_support_frac, edge_guard_guardband_rms_pre,
              edge_guard_guardband_rms_post, edge_guard_exterior_rms_pre,
              edge_guard_exterior_rms_post, edge_guard_exterior_max_abs_pre,
              edge_guard_exterior_max_abs_post},
          n_valid_pixels(n_maps, 0),
          n_core_pixels(n_maps, 0),
          weight_refs{
              weight_sum, core_weight_sum, n_valid_pixels, n_core_pixels},
          peak_row(n_maps, fill_int),
          peak_col(n_maps, fill_int),
          peak_refs{
              peak_abs_sig2noise, core_peak_abs_sig2noise, peak_row,
              peak_col},
          edge_guard_applied(n_maps, 0),
          edge_guard_support_radius_pix(n_maps, 0),
          edge_guard_science_npix(n_maps, 0),
          edge_guard_support_npix(n_maps, 0),
          edge_guard_guardband_npix(n_maps, 0),
          edge_guard_int_refs{
              edge_guard_applied, edge_guard_support_radius_pix,
              edge_guard_science_npix, edge_guard_support_npix,
              edge_guard_guardband_npix},
          map_int_values{
              n_valid_pixels, n_core_pixels, peak_row, peak_col,
              edge_guard_applied, edge_guard_support_radius_pix,
              edge_guard_science_npix, edge_guard_support_npix,
              edge_guard_guardband_npix},
          map_double_values{
              median_err, median_rms, weight_thresholds, weight_sum,
              core_weight_sum, coverage_sum, coverage_max,
              coverage_median_core, empirical_to_formal_noise_ratio,
              noise_weight_median_ratio, noise_weight_scale,
              noise_products_s2n_sigma, noise_products_valid_pixels,
              peak_signal, peak_abs_sig2noise, core_peak_abs_sig2noise,
              noise_rms_p16, noise_rms_p84, core_tail_frac_abs3,
              core_tail_frac_pos3, core_tail_frac_neg3,
              core_tail_excess_abs3, core_tail_excess_pos3,
              core_tail_excess_neg3, core_sig2noise_skew,
              noise_tail_frac_abs3, noise_tail_frac_pos3,
              noise_tail_frac_neg3, noise_tail_excess_abs3,
              noise_tail_excess_pos3, noise_tail_excess_neg3,
              noise_sig2noise_skew, edge_guard_weight_thresholds,
              edge_guard_hits_thresholds, edge_guard_background_levels,
              edge_guard_science_frac, edge_guard_support_frac,
              edge_guard_guardband_rms_pre, edge_guard_guardband_rms_post,
              edge_guard_exterior_rms_pre, edge_guard_exterior_rms_post,
              edge_guard_exterior_max_abs_pre,
              edge_guard_exterior_max_abs_post} {}

    MapdiagMapLabelStorage label_storage;
    std::vector<double> median_err;
    std::vector<double> median_rms;
    std::vector<double> weight_thresholds;
    std::vector<double> weight_sum;
    std::vector<double> core_weight_sum;
    std::vector<double> coverage_sum;
    std::vector<double> coverage_max;
    std::vector<double> coverage_median_core;
    MapdiagCoverageRefs coverage_refs;
    std::vector<double> empirical_to_formal_noise_ratio;
    MapdiagFormalNoiseRefs formal_noise_refs;
    std::vector<double> noise_weight_median_ratio;
    std::vector<double> noise_weight_scale;
    std::vector<double> noise_products_s2n_sigma;
    std::vector<double> noise_products_valid_pixels;
    MapdiagNoiseProductRefs noise_product_refs;
    std::vector<double> peak_signal;
    std::vector<double> peak_abs_sig2noise;
    std::vector<double> core_peak_abs_sig2noise;
    std::vector<double> noise_rms_p16;
    std::vector<double> noise_rms_p84;
    std::vector<double> core_tail_frac_abs3;
    std::vector<double> core_tail_frac_pos3;
    std::vector<double> core_tail_frac_neg3;
    std::vector<double> core_tail_excess_abs3;
    std::vector<double> core_tail_excess_pos3;
    std::vector<double> core_tail_excess_neg3;
    std::vector<double> core_sig2noise_skew;
    MapdiagCoreTailRefs core_tail_refs;
    std::vector<double> noise_tail_frac_abs3;
    std::vector<double> noise_tail_frac_pos3;
    std::vector<double> noise_tail_frac_neg3;
    std::vector<double> noise_tail_excess_abs3;
    std::vector<double> noise_tail_excess_pos3;
    std::vector<double> noise_tail_excess_neg3;
    std::vector<double> noise_sig2noise_skew;
    MapdiagNoiseTailRefs noise_tail_refs;
    std::vector<double> edge_guard_weight_thresholds;
    std::vector<double> edge_guard_hits_thresholds;
    std::vector<double> edge_guard_background_levels;
    std::vector<double> edge_guard_science_frac;
    std::vector<double> edge_guard_support_frac;
    std::vector<double> edge_guard_guardband_rms_pre;
    std::vector<double> edge_guard_guardband_rms_post;
    std::vector<double> edge_guard_exterior_rms_pre;
    std::vector<double> edge_guard_exterior_rms_post;
    std::vector<double> edge_guard_exterior_max_abs_pre;
    std::vector<double> edge_guard_exterior_max_abs_post;
    MapdiagEdgeGuardDoubleRefs edge_guard_double_refs;
    std::vector<int> n_valid_pixels;
    std::vector<int> n_core_pixels;
    MapdiagWeightRefs weight_refs;
    std::vector<int> peak_row;
    std::vector<int> peak_col;
    MapdiagPeakRefs peak_refs;
    std::vector<int> edge_guard_applied;
    std::vector<int> edge_guard_support_radius_pix;
    std::vector<int> edge_guard_science_npix;
    std::vector<int> edge_guard_support_npix;
    std::vector<int> edge_guard_guardband_npix;
    MapdiagEdgeGuardIntRefs edge_guard_int_refs;
    MapdiagMapIntValues map_int_values;
    MapdiagMapDoubleValues map_double_values;
};

struct MapdiagObservationWorkspace {
    explicit MapdiagObservationWorkspace(std::size_t table_size,
                                         double fill_double, int fill_int)
        : weight_sum(table_size, fill_double),
          weight_frac(table_size, fill_double),
          core_weight_sum(table_size, fill_double),
          core_weight_frac(table_size, fill_double),
          valid_pixels(table_size, fill_int),
          core_pixels(table_size, fill_int),
          tables{weight_sum, core_weight_sum, valid_pixels, core_pixels},
          double_values{
              weight_sum, weight_frac, core_weight_sum, core_weight_frac},
          int_values{valid_pixels, core_pixels} {}

    std::vector<double> weight_sum;
    std::vector<double> weight_frac;
    std::vector<double> core_weight_sum;
    std::vector<double> core_weight_frac;
    std::vector<int> valid_pixels;
    std::vector<int> core_pixels;
    MapdiagObsTableRefs tables;
    MapdiagObservationDoubleValues double_values;
    MapdiagObservationIntValues int_values;
};

struct MapdiagOutlierMaskContext {
    MapdiagSourceDistanceContext source_distance;
    Eigen::ArrayXXd off_source_core_mask;
};

template <class ArraysToMaps, class MapsToStokes, class MapsToArrays,
          class ArrayNameMap, class Arrays, class StokesParams,
          class MapNameForIndex>
auto assign_mapdiag_label_entry(
    Eigen::Index map_index, const ArraysToMaps &arrays_to_maps,
    const MapsToStokes &maps_to_stokes, const MapsToArrays &maps_to_arrays,
    ArrayNameMap &array_name_map, const Arrays &arrays,
    StokesParams &stokes_params,
    const MapNameForIndex &map_name_for_index,
    MapdiagMapLabelStorage &label_storage) {
    const std::size_t idx = mapdiag_size_index(map_index);
    const auto write_indices =
        map_write_indices(
            map_index, arrays_to_maps, maps_to_stokes, maps_to_arrays);
    assign_mapdiag_map_labels_from_indices(
        idx, map_index, write_indices, array_name_map, arrays, stokes_params,
        map_name_for_index, label_storage.refs());
    return write_indices;
}

template <class MapBuffer>
auto assign_mapdiag_basic_map_stats(
    Eigen::Index map_index, std::size_t idx, MapBuffer &mb,
    double fill_double, MapdiagMapWorkspace &workspace) {
    const double weight_threshold =
        mapdiag_weight_threshold_for_map(mb, map_index);
    workspace.weight_thresholds[idx] = weight_threshold;
    assign_mapdiag_edge_guard_entry(
        idx, *mb, workspace.edge_guard_int_refs,
        workspace.edge_guard_double_refs);

    const auto weight_arr = mb->weight[map_index].array();
    const auto valid_mask = mapdiag_valid_weight_mask(weight_arr);
    const auto core_mask =
        mapdiag_core_weight_mask(weight_arr, weight_threshold);
    assign_mapdiag_weight_stats(
        idx, mapdiag_weight_stats(weight_arr, valid_mask, core_mask),
        workspace.weight_refs);

    assign_mapdiag_formal_noise_stats_or_fill(
        idx, mb, map_index, fill_double, workspace.formal_noise_refs);
    assign_mapdiag_noise_product_stats_or_fill(
        idx, mb, map_index, fill_double, workspace.noise_product_refs);
    assign_mapdiag_coverage_stats_if_present(
        idx, mb->coverage, map_index, core_mask, fill_double,
        workspace.coverage_refs);
    assign_mapdiag_peak_signal_or_fill(
        idx, mb->signal, map_index, fill_double, workspace.peak_signal);
    return core_mask;
}

template <class MapBuffer>
Eigen::MatrixXd assign_mapdiag_signal_stats_for_map(
    Eigen::Index map_index, std::size_t idx, MapBuffer &mb,
    const Eigen::ArrayXXd &core_mask, double fill_double,
    const MapdiagStatsContext &stats, MapdiagMapWorkspace &workspace) {
    return assign_mapdiag_signal_stats(
        idx, mb->signal[map_index], mb->weight[map_index], core_mask,
        workspace.n_core_pixels[idx], fill_double, stats,
        workspace.peak_refs, workspace.core_tail_refs);
}

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

template <class MapPixelOutlier, class DetectorPenalty, class Candidates,
          class MapBuffer, class ReductionLearning, class Arrays,
          class Logger>
void emit_mapdiag_outlier_learning(
    Candidates &candidates, const MapBuffer &mb, Eigen::Index map_index,
    Eigen::Index write_map_index, const Arrays &arrays,
    const std::string &obsnum, const std::string &record_producer,
    const std::string &stage_name, int fruit_iter, int fill_int,
    ReductionLearning &reduction_learning, const Logger &logger) {
    sort_mapdiag_pixel_candidates(candidates);
    const std::size_t candidate_top_n =
        mapdiag_candidate_top_n(reduction_learning);
    const std::size_t n_emitted_candidates =
        mapdiag_candidate_emit_count(candidates.size(), candidate_top_n);
    auto dominance = make_mapdiag_detector_dominance_list();

    for (std::size_t ci = 0; ci < n_emitted_candidates; ++ci) {
        const auto &candidate = mapdiag_emitted_candidate(candidates, ci);
        const auto outlier_reason =
            mapdiag_map_pixel_outlier_reason(candidate, mb);
        const auto record_map_index = mapdiag_record_map_index(map_index);
        auto record = make_mapdiag_outlier_record<MapPixelOutlier>(
            obsnum, record_producer, outlier_reason, fruit_iter,
            record_map_index, candidate);
        reduction_learning.record_map_pixel_outlier(std::move(record));
        update_mapdiag_detector_dominance(dominance, candidate, fill_int);
    }

    if (!mapdiag_detector_exclusion_enabled(reduction_learning)) {
        return;
    }

    const int detector_exclusion_min_pixels =
        mapdiag_detector_exclusion_min_pixels(reduction_learning);
    const int array_id =
        mapdiag_array_id_or_default(write_map_index, arrays, -1);
    for (const auto &entry : dominance) {
        if (!mapdiag_dominance_meets_min_pixels(
                entry, detector_exclusion_min_pixels)) {
            continue;
        }
        const auto penalty_reason =
            mapdiag_detector_dominance_penalty_reason();
        auto penalty = make_mapdiag_detector_penalty<DetectorPenalty>(
            obsnum, record_producer, penalty_reason, fruit_iter, entry,
            array_id);
        reduction_learning.record_detector_penalty(std::move(penalty), true);
        const auto display_scan_index =
            mapdiag_display_scan_index(entry.scan);
        logger->info(
            "mapdiag learned scan-local detector exclusion candidate stage={} iter={} map={} uid={} scan={} outlier_pixels={} max_abs_value={:.4g} max_abs_leave_one_out_z={:.4g}",
            stage_name, fruit_iter, map_index, entry.uid,
            display_scan_index, entry.count, entry.max_abs_value,
            entry.max_abs_leave_one_out_z);
    }
}

template <class MapPixelOutlier, class DetectorPenalty, class MapBuffer,
          class ReductionLearning, class Arrays, class Logger>
void assign_mapdiag_signal_diagnostics_for_map(
    Eigen::Index map_index, std::size_t storage_index,
    Eigen::Index write_map_index, MapBuffer &mb,
    const Eigen::ArrayXXd &core_mask, double fill_double, int fill_int,
    const MapdiagStatsContext &stats, double rad_to_arcsec,
    double ptc_fs_hz, const Arrays &arrays, const std::string &obsnum,
    const std::string &record_producer, const std::string &stage_name,
    int fruit_iter, ReductionLearning &reduction_learning,
    MapdiagMapWorkspace &workspace, const Logger &logger) {
    if (!mapdiag_has_signal_weight_samples(
            mb->signal[map_index], mb->weight[map_index])) {
        return;
    }

    const Eigen::MatrixXd sig2noise =
        assign_mapdiag_signal_stats_for_map(
            map_index, storage_index, mb, core_mask, fill_double, stats,
            workspace);

    if (mapdiag_outlier_diagnostics_enabled(reduction_learning)) {
        const auto outlier_mask_context =
            make_mapdiag_outlier_mask_context(
                mb, core_mask, reduction_learning, rad_to_arcsec,
                fill_double);
        const auto &source_distance_context =
            outlier_mask_context.source_distance;
        const auto &off_source_core_mask =
            outlier_mask_context.off_source_core_mask;

        const auto off_source_values =
            stats.collect_masked_values(sig2noise, off_source_core_mask);
        if (mapdiag_has_enough_off_source_values(off_source_values)) {
            const auto robust_stats =
                mapdiag_robust_center_stats(stats, off_source_values);
            if (mapdiag_has_valid_robust_center_stats(robust_stats)) {
                auto candidates =
                    collect_mapdiag_pixel_candidates_for_map(
                        mb, map_index, sig2noise, off_source_core_mask,
                        source_distance_context, robust_stats,
                        reduction_learning, ptc_fs_hz, fill_int,
                        fill_double);
                emit_mapdiag_outlier_learning<MapPixelOutlier,
                                              DetectorPenalty>(
                    candidates, mb, map_index, write_map_index, arrays,
                    obsnum, record_producer, stage_name, fruit_iter,
                    fill_int, reduction_learning, logger);
            }
        }
    }

    assign_mapdiag_noise_tail_for_map(
        storage_index, mb, map_index, stats, core_mask,
        workspace.noise_tail_refs);
}

}  // namespace citlali::pipeline
