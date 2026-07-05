#pragma once

#include <citlali/core/pipeline/mapdiag_edge_guard.h>
#include <citlali/core/pipeline/fits_image_metadata.h>
#include <citlali/core/pipeline/mapdiag_labels.h>
#include <citlali/core/pipeline/mapdiag_netcdf.h>
#include <citlali/core/pipeline/mapdiag_observation_weight.h>
#include <citlali/core/pipeline/mapdiag_stats.h>

#include <cstddef>
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

}  // namespace citlali::pipeline
