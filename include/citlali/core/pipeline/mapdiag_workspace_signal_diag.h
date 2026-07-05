#pragma once

// Included by mapdiag_workspace.h inside namespace citlali::pipeline.

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

