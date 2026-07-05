#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/engine/detail/mapdiag_observation_contribution.h>
#include <citlali/core/pipeline/mapdiag_workspace.h>

template <mapmaking::MapType map_t, class map_buffer_t>
void Engine::write_mapdiag(map_buffer_t &mb, std::string dir_name) {
    const std::string filename =
        setup_filenames<map_t, engine_utils::toltecIO::toltec,
                        engine_utils::toltecIO::mapdiag>(dir_name);
    const auto mapdiag_context = citlali::pipeline::make_mapdiag_size_context(
        static_cast<std::size_t>(n_maps),
        std::max<std::size_t>(1, mb->obsnums.size()),
        map_t == mapmaking::RawCoadd || map_t == mapmaking::FilteredCoadd);
    const double fill_double = citlali::pipeline::mapdiag_fill_double();
    const int fill_int = citlali::pipeline::mapdiag_fill_int();
    const auto n_mapdiag_maps = mapdiag_context.n_maps;

    citlali::pipeline::MapdiagMapWorkspace map_workspace{
        n_mapdiag_maps, fill_double, fill_int};
    auto &mapdiag_label_storage = map_workspace.label_storage;
    auto &weight_sum = map_workspace.weight_sum;
    auto &core_weight_sum = map_workspace.core_weight_sum;
    auto &n_valid_pixels = map_workspace.n_valid_pixels;
    auto &n_core_pixels = map_workspace.n_core_pixels;
    auto &core_tail_refs = map_workspace.core_tail_refs;
    auto &noise_tail_refs = map_workspace.noise_tail_refs;
    auto &peak_refs = map_workspace.peak_refs;
    const std::size_t obs_table_size =
        citlali::pipeline::mapdiag_obs_table_size(mapdiag_context);
    citlali::pipeline::MapdiagObservationWorkspace obs_workspace{
        obs_table_size, fill_double, fill_int};
    auto &obs_weight_sum = obs_workspace.weight_sum;
    auto &obs_core_weight_sum = obs_workspace.core_weight_sum;
    auto &obs_weight_frac = obs_workspace.weight_frac;
    auto &obs_core_weight_frac = obs_workspace.core_weight_frac;
    auto &obs_tables = obs_workspace.tables;

    const std::string stage_name =
        citlali::pipeline::mapdiag_stage_name<map_t>();
    const auto mapdiag_metadata =
        citlali::pipeline::make_mapdiag_metadata_vars(
            stage_name, mb, map_regime, telescope.source_name,
            telescope.project_id, telescope.obs_goal, wiener_filter);
    const auto mapdiag_labels =
        citlali::pipeline::make_mapdiag_label_vars(
            mapdiag_label_storage, mb->obsnums, obsnum, date_obs,
            mapdiag_context);
    const auto mapdiag_values =
        citlali::pipeline::make_mapdiag_value_vars(
            map_workspace.map_double_values, map_workspace.map_int_values,
            obs_workspace.double_values, obs_workspace.int_values);

    const citlali::pipeline::MapdiagStatsContext mapdiag_stats{fill_double};
    const std::string mapdiag_record_producer =
        citlali::pipeline::mapdiag_record_producer(stage_name);
    auto map_name_for_index = [&](Eigen::Index map_i) {
        return get_map_name(map_i);
    };

    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const std::size_t idx = citlali::pipeline::mapdiag_size_index(i);
        const auto write_indices =
            citlali::pipeline::assign_mapdiag_label_entry(
                i, arrays_to_maps, maps_to_stokes, maps_to_arrays,
                toltec_io.array_name_map, calib.arrays,
                rtcproc.polarization.stokes_params, map_name_for_index,
                mapdiag_label_storage);
        const auto core_mask = citlali::pipeline::assign_mapdiag_basic_map_stats(
            i, idx, mb, fill_double, map_workspace);
        if (citlali::pipeline::mapdiag_has_signal_weight_samples(
                mb->signal[i], mb->weight[i])) {
            const Eigen::MatrixXd sig2noise =
                citlali::pipeline::assign_mapdiag_signal_stats_for_map(
                    i, idx, mb, core_mask, fill_double, mapdiag_stats,
                    map_workspace);

            if (citlali::pipeline::mapdiag_outlier_diagnostics_enabled(
                    reduction_learning)) {
                const auto outlier_mask_context =
                    citlali::pipeline::make_mapdiag_outlier_mask_context(
                        mb, core_mask, reduction_learning, RAD_TO_ASEC,
                        fill_double);
                const auto &source_distance_context =
                    outlier_mask_context.source_distance;
                const auto &off_source_core_mask =
                    outlier_mask_context.off_source_core_mask;

                const auto off_source_values =
                    mapdiag_stats.collect_masked_values(
                        sig2noise, off_source_core_mask);
                if (citlali::pipeline::mapdiag_has_enough_off_source_values(
                        off_source_values)) {
                    const auto robust_stats =
                        citlali::pipeline::mapdiag_robust_center_stats(
                            mapdiag_stats, off_source_values);
                    if (citlali::pipeline::
                            mapdiag_has_valid_robust_center_stats(
                                robust_stats)) {
                        auto candidates =
                            citlali::pipeline::
                                collect_mapdiag_pixel_candidates_for_map(
                                    mb, i, sig2noise, off_source_core_mask,
                                    source_distance_context, robust_stats,
                                    reduction_learning,
                                    processed_time_chunk_fs_hz(), fill_int,
                                    fill_double);
                        citlali::pipeline::emit_mapdiag_outlier_learning<
                            ReductionLearningState::MapPixelOutlier,
                            ReductionLearningState::DetectorPenalty>(
                            candidates, mb, i, write_indices.map_index,
                            calib.arrays, obsnum, mapdiag_record_producer,
                            stage_name, fruit_iter, fill_int,
                            reduction_learning, logger);
                    }
                }
            }

            citlali::pipeline::assign_mapdiag_noise_tail_for_map(
                idx, mb, i, mapdiag_stats, core_mask, noise_tail_refs);
        }

        citlali::engine_detail::
            assign_mapdiag_observation_contributions_for_map(
                mapdiag_context, i, idx, mb, core_mask, weight_sum[idx],
                core_weight_sum[idx], n_valid_pixels[idx],
                n_core_pixels[idx], toltec_io, redu_dir_name, redu_type,
                telescope.sim_obs, mapdiag_label_storage, obs_tables,
                logger);
        citlali::pipeline::assign_mapdiag_obs_fractions_for_map(
            obs_weight_sum, obs_core_weight_sum, fill_double,
            mapdiag_context, idx, obs_weight_frac, obs_core_weight_frac);
    }

    write_netcdf_atomic(
        citlali::pipeline::mapdiag_netcdf_filename(filename),
        [&](netCDF::NcFile &fo) {
            const auto mapdiag_netcdf_vars =
                citlali::pipeline::make_mapdiag_netcdf_vars(
                    mapdiag_context, obsnum, mapdiag_metadata,
                    mapdiag_labels, mapdiag_values);
            citlali::pipeline::add_mapdiag_netcdf_vars(
                fo, mapdiag_netcdf_vars);
        });
}
