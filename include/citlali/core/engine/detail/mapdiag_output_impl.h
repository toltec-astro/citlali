#pragma once

// Engine output implementation detail.
// Include this only after Engine has been declared.

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
                            citlali::pipeline::make_mapdiag_pixel_candidates();
                        const bool has_contribution_products =
                            citlali::pipeline::
                                mapdiag_has_contribution_products(mb, i);
                        const double ptc_fs_hz = processed_time_chunk_fs_hz();
                        const Eigen::Index n_mapdiag_rows =
                            citlali::pipeline::mapdiag_n_rows(mb);
                        const Eigen::Index n_mapdiag_cols =
                            citlali::pipeline::mapdiag_n_cols(mb);
                        const double min_effective_samples =
                            citlali::pipeline::mapdiag_min_effective_samples(
                                reduction_learning);
                        const double min_abs_z =
                            citlali::pipeline::mapdiag_min_abs_z(
                                reduction_learning);

                        for (Eigen::Index r = 0; r < n_mapdiag_rows; ++r) {
                            for (Eigen::Index c = 0; c < n_mapdiag_cols; ++c) {
                                if (!citlali::pipeline::
                                        mapdiag_mask_pixel_is_selected(
                                            off_source_core_mask, r, c)) {
                                    continue;
                                }

                                const double value =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->signal[i], r, c);
                                const double wt =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            mb->weight[i], r, c);
                                const double sn =
                                    citlali::pipeline::
                                        mapdiag_matrix_double_value(
                                            sig2noise, r, c);
                                if (!citlali::pipeline::
                                        mapdiag_is_valid_outlier_pixel_value(
                                            value, wt, sn)) {
                                    continue;
                                }

                                const double n_eff =
                                    citlali::pipeline::
                                        mapdiag_effective_samples_or_fill(
                                            mb->coverage, i, r, c,
                                            mb->n_rows, mb->n_cols,
                                            ptc_fs_hz, fill_double);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_effective_samples(
                                            n_eff, min_effective_samples)) {
                                    continue;
                                }

                                const double z =
                                    citlali::pipeline::mapdiag_robust_z(
                                        sn, robust_stats);
                                if (!citlali::pipeline::
                                        mapdiag_passes_min_abs_z(z,
                                                                 min_abs_z)) {
                                    continue;
                                }

                                const double source_distance_arcsec =
                                    citlali::pipeline::
                                        mapdiag_source_distance_arcsec(
                                            r, c, source_distance_context);
                                auto candidate =
                                    citlali::pipeline::
                                        make_mapdiag_map_pixel_candidate(
                                            r, c, value, wt, n_eff, z,
                                            source_distance_arcsec,
                                            fill_int, fill_double);

                                if (has_contribution_products) {
                                    const auto contribution_map_index =
                                        citlali::pipeline::
                                            mapdiag_contribution_map_index(i);
                                    const int uid =
                                        citlali::pipeline::
                                            mapdiag_matrix_value(
                                                mb->contribution_uid[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_signal =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_signal[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_weight[
                                                    contribution_map_index],
                                                r, c);
                                    const double contrib_variance_weight =
                                        citlali::pipeline::
                                            mapdiag_matrix_double_value(
                                                mb->contribution_variance_weight[
                                                    contribution_map_index],
                                                r, c);
                                    if (citlali::pipeline::
                                            mapdiag_has_valid_contributor(
                                                uid, fill_int,
                                                contrib_signal)) {
                                        citlali::pipeline::
                                            assign_mapdiag_candidate_contributor_from_products(
                                                candidate, uid,
                                                mb->contribution_scan[
                                                    contribution_map_index],
                                                mb->contribution_sample[
                                                    contribution_map_index],
                                                r, c);
                                        const double total_signal =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_signal[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double total_variance_weight =
                                            citlali::pipeline::
                                                mapdiag_matrix_double_value(
                                                    mb->contribution_total_variance_weight[
                                                        contribution_map_index],
                                                    r, c);
                                        const double remaining_weight =
                                            citlali::pipeline::
                                                mapdiag_remaining_contribution_weight(
                                                    total_weight,
                                                    contrib_weight);
                                        if (citlali::pipeline::
                                                mapdiag_has_full_leave_one_out_inputs(
                                                    total_signal,
                                                    total_weight,
                                                    contrib_weight,
                                                    contrib_variance_weight,
                                                    total_variance_weight,
                                                    remaining_weight)) {
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_full_leave_one_out_value(
                                                        total_signal,
                                                        contrib_signal,
                                                        remaining_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                        else if (citlali::pipeline::
                                                     mapdiag_has_fallback_leave_one_out_inputs(
                                                         wt, contrib_weight)) {
                                            const double raw_sum =
                                                citlali::pipeline::
                                                    mapdiag_raw_weighted_signal(
                                                        value, wt);
                                            const double loo_value =
                                                citlali::pipeline::
                                                    mapdiag_fallback_leave_one_out_value(
                                                        raw_sum,
                                                        contrib_signal, wt,
                                                        contrib_weight);
                                            citlali::pipeline::
                                                mapdiag_assign_leave_one_out_z(
                                                    value, wt, loo_value,
                                                    candidate.leave_one_out_z);
                                        }
                                    }
                                }
                                citlali::pipeline::
                                    append_mapdiag_pixel_candidate(
                                        candidates, candidate);
                            }
                        }

                        citlali::pipeline::sort_mapdiag_pixel_candidates(
                            candidates);
                        const std::size_t candidate_top_n =
                            citlali::pipeline::mapdiag_candidate_top_n(
                                reduction_learning);
                        const std::size_t n_emitted_candidates =
                            citlali::pipeline::mapdiag_candidate_emit_count(
                                candidates.size(), candidate_top_n);
                        auto dominance =
                            citlali::pipeline::
                                make_mapdiag_detector_dominance_list();

                        for (std::size_t ci = 0; ci < n_emitted_candidates;
                             ++ci) {
                            const auto &candidate =
                                citlali::pipeline::mapdiag_emitted_candidate(
                                    candidates, ci);
                            const auto outlier_reason =
                                citlali::pipeline::
                                    mapdiag_map_pixel_outlier_reason(
                                        candidate, mb);
                            const auto record_map_index =
                                citlali::pipeline::mapdiag_record_map_index(i);
                            auto record =
                                citlali::pipeline::make_mapdiag_outlier_record<
                                    ReductionLearningState::MapPixelOutlier>(
                                    obsnum, mapdiag_record_producer,
                                    outlier_reason, fruit_iter,
                                    record_map_index, candidate);
                            reduction_learning.record_map_pixel_outlier(
                                std::move(record));
                            citlali::pipeline::
                                update_mapdiag_detector_dominance(
                                    dominance, candidate, fill_int);
                        }

                        const bool detector_exclusion_enabled =
                            citlali::pipeline::
                                mapdiag_detector_exclusion_enabled(
                                    reduction_learning);
                        if (detector_exclusion_enabled) {
                            const int detector_exclusion_min_pixels =
                                citlali::pipeline::
                                    mapdiag_detector_exclusion_min_pixels(
                                        reduction_learning);
                            const int array_id =
                                citlali::pipeline::mapdiag_array_id_or_default(
                                    write_indices.map_index, calib.arrays,
                                    -1);
                            for (const auto &entry : dominance) {
                                if (!citlali::pipeline::
                                        mapdiag_dominance_meets_min_pixels(
                                            entry,
                                            detector_exclusion_min_pixels)) {
                                    continue;
                                }
                                const auto penalty_reason =
                                    citlali::pipeline::
                                        mapdiag_detector_dominance_penalty_reason();
                                auto penalty =
                                    citlali::pipeline::
                                        make_mapdiag_detector_penalty<
                                            ReductionLearningState::
                                                DetectorPenalty>(
                                            obsnum, mapdiag_record_producer,
                                            penalty_reason,
                                            fruit_iter, entry, array_id);
                                reduction_learning.record_detector_penalty(
                                    std::move(penalty), true);
                                const auto display_scan_index =
                                    citlali::pipeline::
                                        mapdiag_display_scan_index(entry.scan);
                                logger->info(
                                    "mapdiag learned scan-local detector exclusion candidate stage={} iter={} map={} uid={} scan={} outlier_pixels={} max_abs_value={:.4g} max_abs_leave_one_out_z={:.4g}",
                                    stage_name, fruit_iter, i, entry.uid,
                                    display_scan_index,
                                    entry.count, entry.max_abs_value,
                                    entry.max_abs_leave_one_out_z);
                            }
                        }
                    }
                }
            }

            citlali::pipeline::assign_mapdiag_noise_tail_for_map(
                idx, mb, i, mapdiag_stats, core_mask, noise_tail_refs);
        }

        if (citlali::pipeline::mapdiag_is_single_observation_context(
                mapdiag_context)) {
            citlali::pipeline::assign_mapdiag_single_obs_entry(
                mapdiag_context, idx, weight_sum[idx],
                core_weight_sum[idx], n_valid_pixels[idx],
                n_core_pixels[idx], obs_tables);
        }
        else {
            const auto n_obsnums = mb->obsnums.size();
            for (std::size_t obs_idx = 0; obs_idx < n_obsnums; ++obs_idx) {
                const auto &coadd_obsnum = mb->obsnums[obs_idx];
                const auto obs_dir =
                    citlali::pipeline::mapdiag_obs_raw_dir(
                        redu_dir_name, coadd_obsnum);
                const auto obs_weight_path =
                    toltec_io
                        .create_filename<engine_utils::toltecIO::toltec,
                                         engine_utils::toltecIO::map,
                                         engine_utils::toltecIO::raw>(
                        obs_dir, redu_type,
                        mapdiag_label_storage.array_names[idx], coadd_obsnum,
                        telescope.sim_obs) +
                    ".fits";
                const auto weight_hdu_name =
                    citlali::pipeline::mapdiag_weight_hdu_name(
                        mapdiag_label_storage.map_names[idx],
                        mapdiag_label_storage.stokes_names[idx]);
                try {
                    fitsIO<file_type_enum::read_fits, CCfits::ExtHDU*>
                        obs_fits(obs_weight_path);
                    const auto obs_weight = obs_fits.get_hdu(weight_hdu_name);
                    citlali::pipeline::accumulate_mapdiag_obs_weight(
                        i, mapdiag_context.n_obsnums, mb->n_rows, mb->n_cols,
                        core_mask, obs_weight, obs_idx, obs_tables);
                } catch (const std::exception &e) {
                    logger->warn(
                        "failed to derive mapdiag contribution from {} [{}]: {}",
                        obs_weight_path, weight_hdu_name, e.what());
                    citlali::pipeline::zero_mapdiag_obs_entry(
                        mapdiag_context, idx, obs_idx, obs_tables);
                }
            }
        }
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
