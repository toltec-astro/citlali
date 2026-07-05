#pragma once

// Included by mapdiag_workspace.h inside namespace citlali::pipeline.

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

