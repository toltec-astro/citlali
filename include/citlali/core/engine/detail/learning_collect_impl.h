#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

template <class rtc_t, class ptc_t, class calib_t>
void Engine::collect_rtc_learning_diagnostics(rtc_t &rtcdata, ptc_t &ptcdata,
                                              calib_t &calib_scan,
                                              const std::vector<timestream::RTCProc::RTCDetectorDiagSummary> &det_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;
    if (det_summary.empty()) {
        return;
    }

    const auto rtc_source_summary =
        rtcproc.snapshot_source_protection_diag_summary(scan_id);
    if (rtc_source_summary.enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "rtc_despike";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.protected_samples = rtc_source_summary.protected_samples;
        source_summary.total_samples = rtc_source_summary.total_samples;
        source_summary.radius_arcsec = rtc_source_summary.radius_arcsec;
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    auto record_event = [&](const auto &event, Eigen::Index det,
                            const std::string &reason) {
        const auto uid_it = calib_scan.apt.find("uid");
        if (!event.valid() || !event.accepted || uid_it == calib_scan.apt.end() ||
            det < 0 || det >= uid_it->second.size()) {
            return;
        }
        ReductionLearningState::LearnedSampleMask record;
        record.obsnum = obsnum;
        record.producer = "rtc_despike";
        record.reason = reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = citlali_learning_apt_int(calib_scan.apt, "uid", det,
                                              static_cast<int>(det));
        record.nw = citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
        record.array = citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
        record.raw_start = event.start_sample;
        record.raw_stop = event.end_sample;
        record.score = event.score;
        record.z = event.score;
        record.confidence = 1.0;
        record.source_protected = false;
        record.apply_pre_rtc = true;
        reduction_learning.record_learned_sample_mask(std::move(record));
    };

    for (const auto &row : det_summary) {
        const Eigen::Index det = row.det;
        record_event(row.local_raw_event, det, "local_raw_accepted");
        record_event(row.local_delta_event, det, "local_delta_accepted");
    }
}

template <class ptc_t, class calib_t>
void Engine::collect_ptc_learning_diagnostics(
    ptc_t &ptcdata, calib_t &calib_scan,
    const std::vector<timestream::PTCProc::SecondPassDiagSummary> &second_pass_summary,
    const std::vector<timestream::PTCProc::HighWeightDiagSummary> &high_weight_summary) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled()) {
        return;
    }

    const auto scan_id = ptcdata.index.data;

    if (ptcproc.second_pass_local.source_protection_enabled) {
        ReductionLearningState::SourceProtectionSummary source_summary;
        source_summary.obsnum = obsnum;
        source_summary.producer = "ptc_second_pass";
        source_summary.mode = "map_center_radius";
        source_summary.iter = fruit_iter;
        source_summary.scan = static_cast<int>(scan_id);
        source_summary.total_samples =
            static_cast<int>(ptcdata.scans.data.rows() * ptcdata.scans.data.cols());
        source_summary.radius_arcsec =
            ptcproc.second_pass_local.source_protection_radius_arcsec;
        auto [source_mask, source_info] = engine_utils::calc_source_protection_mask(
            ptcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius",
            ptcproc.second_pass_local.source_protection_radius_arcsec);
        (void) source_mask;
        source_summary.protected_samples =
            static_cast<int>(source_info.protected_samples);
        reduction_learning.record_source_protection_summary(std::move(source_summary));
    }

    for (const auto &summary : high_weight_summary) {
        ReductionLearningState::HighWeightDetector record;
        record.obsnum = obsnum;
        record.grouping = summary.grouping;
        record.reason = summary.reason;
        record.iter = fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = summary.uid;
        record.nw = static_cast<int>(summary.nw);
        record.array = static_cast<int>(summary.array);
        record.weight = summary.approximate_weight;
        record.final_weight = summary.final_weight;
        record.group_median = summary.group_median_weight;
        record.robust_z = summary.robust_z;
        record.cap = summary.applied_cap;
        record.validation_factor = summary.validation_factor;
        record.cap_recommended = summary.cap_recommended;
        record.cap_applied = summary.cap_applied;
        record.validated = summary.validated;
        reduction_learning.record_high_weight_detector(std::move(record));
    }

    if (second_pass_summary.empty()) {
        return;
    }

    for (const auto &summary : second_pass_summary) {
        const bool has_candidate = summary.n_candidate_clusters > 0 ||
                                   summary.n_candidate_events > 0;
        const bool has_residual =
            std::isfinite(summary.max_unflagged_residual_z) &&
            summary.max_unflagged_residual_uid != timestream::kTransientFillInt;
        const bool selective_acceptance_recommended =
            summary.busy_network_vetoed &&
            ((std::isfinite(summary.top_candidate_cluster_peak_score) &&
              summary.top_candidate_cluster_peak_score >=
                  ptcproc.second_pass_local.high_score_cluster_override) ||
             (std::isfinite(summary.max_unflagged_residual_z) &&
              summary.max_unflagged_residual_z >=
                  ptcproc.second_pass_local.high_score_event_override));
        if (has_candidate || has_residual || summary.busy_network_vetoed) {
            ReductionLearningState::BusyNetworkSummary record;
            record.obsnum = obsnum;
            record.producer = "ptc_second_pass";
            record.reason = summary.busy_network_vetoed
                ? "busy_network_vetoed"
                : "candidate_or_residual";
            record.iter = fruit_iter;
            record.scan = static_cast<int>(scan_id);
            record.nw = static_cast<int>(summary.nw);
            record.n_candidate_clusters =
                static_cast<int>(summary.n_candidate_clusters);
            record.n_candidate_events =
                static_cast<int>(summary.n_candidate_events);
            record.n_accepted_clusters =
                static_cast<int>(summary.n_accepted_clusters);
            record.n_accepted_events =
                static_cast<int>(summary.n_accepted_events);
            record.n_rejected_clusters =
                static_cast<int>(summary.n_rejected_clusters);
            record.n_rejected_events =
                static_cast<int>(summary.n_rejected_events);
            record.n_source_protected_clusters =
                static_cast<int>(summary.n_source_protected_clusters);
            record.n_source_protected_events =
                static_cast<int>(summary.n_source_protected_events);
            record.max_unflagged_residual_uid = summary.max_unflagged_residual_uid;
            record.top_candidate_sample = summary.top_candidate_cluster_sample;
            record.top_candidate_score = summary.top_candidate_cluster_peak_score;
            record.max_unflagged_residual_z = summary.max_unflagged_residual_z;
            record.busy_vetoed = summary.busy_network_vetoed;
            record.selective_acceptance_recommended = selective_acceptance_recommended;
            reduction_learning.record_busy_network_summary(std::move(record));
        }

        if (reduction_learning.options.scan_network_pathology_enabled &&
            summary.nw >= 0) {
            const int off_source_candidate_events = std::max<Eigen::Index>(
                0, summary.n_candidate_events - summary.n_source_protected_events);
            const double max_residual_z = std::isfinite(summary.max_unflagged_residual_z)
                ? summary.max_unflagged_residual_z
                : 0.0;
            const bool busy_pathology =
                summary.busy_network_vetoed &&
                summary.n_candidate_clusters >=
                    reduction_learning.options.scan_network_pathology_min_candidate_clusters &&
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_min_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_min_max_residual_z;
            const bool severe_pathology =
                off_source_candidate_events >=
                    reduction_learning.options.scan_network_pathology_severe_candidate_events &&
                max_residual_z >=
                    reduction_learning.options.scan_network_pathology_severe_max_residual_z;
            if (busy_pathology || severe_pathology) {
                ReductionLearningState::DetectorPenalty penalty;
                penalty.obsnum = obsnum;
                penalty.producer = "ptc_second_pass";
                penalty.reason = "busy_network_pathology";
                penalty.iter = fruit_iter;
                penalty.scan = static_cast<int>(scan_id);
                penalty.uid = -1;
                penalty.nw = static_cast<int>(summary.nw);
                penalty.array = citlali_learning_array_for_nw(
                    calib_scan.apt, penalty.nw, -1);
                penalty.factor = 0.0;
                penalty.score = std::max(
                    max_residual_z,
                    std::isfinite(summary.top_candidate_cluster_peak_score)
                        ? summary.top_candidate_cluster_peak_score
                        : 0.0);
                penalty.scan_local = true;
                reduction_learning.record_detector_penalty(std::move(penalty));
            }
        }

        for (const auto &event : summary.candidate_events) {
            if (event.uid == timestream::kTransientFillInt ||
                event.start_sample < 0 ||
                event.end_sample < event.start_sample) {
                continue;
            }
            if (!event.accepted || event.source_protected) {
                continue;
            }
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, event.uid);
            ReductionLearningState::LearnedSampleMask candidate_record;
            candidate_record.obsnum = obsnum;
            candidate_record.producer = "ptc_second_pass";
            candidate_record.reason = event.busy_network_vetoed
                ? "busy_selective_accepted_event"
                : "candidate_event";
            candidate_record.iter = fruit_iter;
            candidate_record.scan = static_cast<int>(scan_id);
            candidate_record.uid = event.uid;
            candidate_record.nw = static_cast<int>(summary.nw);
            candidate_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            candidate_record.ptc_start = event.start_sample;
            candidate_record.ptc_stop = event.end_sample;
            candidate_record.score = event.score;
            candidate_record.z = event.score;
            candidate_record.value = event.cluster_score;
            candidate_record.confidence = event.busy_network_vetoed ? 0.8 : 1.0;
            candidate_record.source_protected = event.source_protected;
            candidate_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(candidate_record));
        }

        if (summary.top_event.valid() && summary.top_event.accepted &&
            summary.top_event_uid != timestream::kTransientFillInt) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, summary.top_event_uid);
            ReductionLearningState::LearnedSampleMask sample_record;
            sample_record.obsnum = obsnum;
            sample_record.producer = "ptc_second_pass";
            sample_record.reason = "accepted_event";
            sample_record.iter = fruit_iter;
            sample_record.scan = static_cast<int>(scan_id);
            sample_record.uid = summary.top_event_uid;
            sample_record.nw = static_cast<int>(summary.nw);
            sample_record.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            sample_record.ptc_start = summary.top_event.start_sample;
            sample_record.ptc_stop = summary.top_event.end_sample;
            sample_record.score = summary.top_event.score;
            sample_record.z = summary.top_event.score;
            sample_record.confidence = 1.0;
            sample_record.source_protected = false;
            sample_record.apply_pre_rtc = false;
            reduction_learning.record_learned_sample_mask(std::move(sample_record));
        }

        if (summary.busy_network_vetoed && has_residual &&
            summary.max_unflagged_residual_z >=
                ptcproc.second_pass_local.high_score_event_override) {
            const Eigen::Index det = citlali_learning_find_det_by_uid(
                calib_scan.apt, summary.max_unflagged_residual_uid);
            ReductionLearningState::DetectorPenalty penalty;
            penalty.obsnum = obsnum;
            penalty.producer = "ptc_second_pass";
            penalty.reason = "busy_vetoed_residual";
            penalty.iter = fruit_iter;
            penalty.scan = static_cast<int>(scan_id);
            penalty.uid = summary.max_unflagged_residual_uid;
            penalty.nw = static_cast<int>(summary.nw);
            penalty.array =
                citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
            penalty.factor = 0.0;
            penalty.score = summary.max_unflagged_residual_z;
            penalty.scan_local = true;
            reduction_learning.record_detector_penalty(std::move(penalty));
        }
    }
}

