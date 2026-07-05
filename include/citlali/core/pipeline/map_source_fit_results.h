#pragma once

// Included by map_source_finding.h inside namespace citlali::pipeline.

template <class MapBuffer, class MapIndex, class SourceIndex>
SourceInitialPosition source_initial_position(
    const MapBuffer &map_buffer, MapIndex map_index,
    SourceIndex source_index) {
    return {
        static_cast<double>(
            map_buffer.row_source_locs[map_index](source_index)),
        static_cast<double>(
            map_buffer.col_source_locs[map_index](source_index))};
}

template <class SourceRow, class SourceIndex>
auto source_fit_result_row(SourceRow source_row_start,
                           SourceIndex source_index) {
    return source_row_start + source_index;
}

template <class MapBuffer, class SourceRow, class SourceIndex,
          class Params, class PErrors>
void store_source_fit_result(MapBuffer &map_buffer,
                             SourceRow source_row_start,
                             SourceIndex source_index,
                             const Params &params,
                             const PErrors &perrors) {
    const auto source_row =
        source_fit_result_row(source_row_start, source_index);
    map_buffer.source_params.row(source_row) = params;
    map_buffer.source_perror.row(source_row) = perrors;
}

template <class MapBuffer, class SourceRow, class SourceIndex,
          class Params, class PErrors, class TangentToAbs>
void normalize_and_store_source_fit_result(
    MapBuffer &map_buffer, SourceRow source_row_start,
    SourceIndex source_index, Params &params, PErrors &perrors,
    const std::string &pixel_axes,
    const SourceFitUnitConstants &constants,
    const TangentToAbs &tangent_to_abs) {
    rescale_source_fit_result(
        params, perrors, map_buffer.n_rows, map_buffer.n_cols,
        map_buffer.pixel_size_rad, pixel_axes, map_buffer.wcs,
        constants, tangent_to_abs);
    store_source_fit_result(
        map_buffer, source_row_start, source_index, params, perrors);
}

template <class SourceRow, class SourceCount>
auto next_source_fit_row_start(SourceRow source_row_start,
                               SourceCount n_map_sources) {
    return source_row_start + n_map_sources;
}

template <class MapBuffer, class MapCount, class SourceFitCallbacks>
void fit_detected_map_sources(MapBuffer &map_buffer, MapCount n_maps,
                              const SourceFitCallbacks &callbacks) {
    Eigen::Index source_row_start = 0;

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const auto n_map_sources = map_buffer.n_sources[i];
        if (!has_sources(n_map_sources)) {
            continue;
        }

        const auto array = callbacks.maps_to_arrays(i);
        const auto init_fwhm = callbacks.init_fwhm_for_array(array);
        callbacks.fit_map_sources(
            i, n_map_sources, init_fwhm, source_row_start);
        source_row_start =
            next_source_fit_row_start(source_row_start, n_map_sources);
    }
}

