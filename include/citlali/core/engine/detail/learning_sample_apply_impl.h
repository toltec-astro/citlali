#pragma once

// Engine learned sample-mask application detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/map_grouping_policy.h>

template <class tc_t, class calib_t>
void Engine::apply_learned_sample_masks(tc_t &tcdata, calib_t &calib_scan,
                                        bool apply_pre_rtc,
                                        const std::string &stage,
                                        bool source_protection_enabled,
                                        double source_protection_radius_arcsec) {
    if (!learning.is_enabled() ||
        !learning.options.apply_sample_masks_enabled ||
        !learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    const auto records = learning.effective_sample_masks_for(
        observation_identity.obsnum, scan_id, apply_pre_rtc,
        iteration.fruit_iter);
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = observation_identity.obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = iteration.fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    summary.max_new_flagged_fraction =
        learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> proposed(n_pts, n_dets);
    proposed.setZero();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_source_protection = false;
    if (source_protection_enabled && source_protection_radius_arcsec > 0.0) {
        const auto map_grouping =
            citlali::pipeline::active_map_grouping_name(*this);
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", source_protection_radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (source_protection_enabled && !have_source_protection) {
            logger->warn(
                "learned mask {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto &record : records) {
        const Eigen::Index det = citlali::pipeline::learning_find_det_by_uid(calib_scan.apt, record.uid);
        const long long raw_start = record.start;
        const long long raw_stop = record.stop;
        if (det < 0 || det >= n_dets || raw_start < 0 || raw_stop < raw_start ||
            raw_stop < 0 || raw_start >= n_pts) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index start =
            std::max<Eigen::Index>(0, static_cast<Eigen::Index>(raw_start));
        const Eigen::Index stop =
            std::min<Eigen::Index>(n_pts - 1, static_cast<Eigen::Index>(raw_stop));
        if (stop < start) {
            ++summary.invalid_records;
            continue;
        }

        ++summary.matched_records;
        for (Eigen::Index sample = start; sample <= stop; ++sample) {
            if (have_source_protection && source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            if (!proposed(sample, det)) {
                proposed(sample, det) = true;
                ++summary.proposed_samples;
            }
        }
    }

    if (summary.proposed_samples > 0) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (!proposed(sample, det)) {
                    continue;
                }
                if (tcdata.flags.data(sample, det)) {
                    ++summary.already_flagged_samples;
                }
                else {
                    ++summary.newly_flagged_samples;
                }
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        learning.options.apply_max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            learning.options.apply_max_new_flagged_fraction;
    if (!over_cap) {
        for (Eigen::Index det = 0; det < n_dets; ++det) {
            for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                if (proposed(sample, det)) {
                    tcdata.flags.data(sample, det) = true;
                }
            }
        }
        summary.applied = true;
    }

    learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} sample-mask application rejected scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, iteration.fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            learning.options.apply_max_new_flagged_fraction);
    }
    else if (summary.proposed_samples > 0) {
        logger->info(
            "learned {} sample masks applied scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} already_flagged={} source_protected={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, iteration.fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.source_protected_samples, summary.newly_flagged_fraction);
    }
}
