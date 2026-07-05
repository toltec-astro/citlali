#pragma once

// Included by mapdiag_workspace.h inside namespace citlali::pipeline.

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

