#pragma once

// Engine RTC learning diagnostic collection detail.
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
        source_summary.iter = iteration.fruit_iter;
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
        record.iter = iteration.fruit_iter;
        record.scan = static_cast<int>(scan_id);
        record.uid = citlali::pipeline::learning_apt_int(calib_scan.apt, "uid", det,
                                              static_cast<int>(det));
        record.nw = citlali::pipeline::learning_apt_int(calib_scan.apt, "nw", det, -1);
        record.array = citlali::pipeline::learning_apt_int(calib_scan.apt, "array", det, -1);
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

