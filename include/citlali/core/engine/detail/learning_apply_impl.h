#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

template <class rtc_t, class calib_t>
void Engine::apply_learned_rtc_sample_masks(rtc_t &rtcdata, calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        rtcdata, calib_scan, "pre_rtc_detector_exclusion", true, false,
        true, true);
    apply_learned_sample_masks(
        rtcdata, calib_scan, true, "pre_rtc",
        rtcproc.despiker.source_protection_enabled,
        rtcproc.despiker.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_sample_masks(ptc_t &ptcdata, calib_t &calib_scan) {
    apply_learned_sample_masks(
        ptcdata, calib_scan, false, "pre_ptc",
        ptcproc.second_pass_local.source_protection_enabled,
        ptcproc.second_pass_local.source_protection_radius_arcsec);
}

template <class ptc_t, class calib_t>
void Engine::apply_learned_ptc_detector_exclusions(ptc_t &ptcdata,
                                                   calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        ptcdata, calib_scan, "pre_ptc_detector_exclusion", false, true,
        true, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_mapmaking_detector_exclusions(tc_t &tcdata,
                                                         calib_t &calib_scan) {
    apply_learned_detector_exclusions(
        tcdata, calib_scan, "pre_mapmaking_detector_exclusion", false, false,
        false, true);
}

template <class tc_t, class calib_t>
void Engine::apply_learned_detector_exclusions(tc_t &tcdata,
                                               calib_t &calib_scan,
                                               const std::string &stage,
                                               bool pre_rtc,
                                               bool update_apt_flags,
                                               bool include_detector_records,
                                               bool include_network_records) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const bool mapdiag_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.map_pixel_outlier_detector_exclusion_enabled;
    const bool busy_detector_exclusion =
        include_detector_records &&
        reduction_learning.options.busy_detector_exclusion_enabled;
    const bool network_exclusion =
        include_network_records &&
        reduction_learning.options.scan_network_pathology_enabled &&
        (stage == "pre_mapmaking_detector_exclusion"
             ? reduction_learning.options.scan_network_pathology_apply_pre_mapmaking
             : ((!pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_ptc) ||
                (pre_rtc && reduction_learning.options.scan_network_pathology_apply_pre_rtc)));
    if (!mapdiag_detector_exclusion && !busy_detector_exclusion &&
        !network_exclusion) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::DetectorPenalty> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.detector_penalties) {
            if (record.obsnum != obsnum ||
                !record.scan_local ||
                record.scan != scan_id ||
                record.iter < 0 ||
                record.iter >= fruit_iter ||
                !std::isfinite(record.factor) ||
                record.factor > 0.0) {
                continue;
            }
            const bool is_mapdiag_detector =
                mapdiag_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "map_pixel_outlier_detector_dominance" &&
                record.producer.rfind("mapdiag:", 0) == 0;
            const bool is_busy_detector =
                busy_detector_exclusion &&
                record.uid >= 0 &&
                record.reason == "busy_vetoed_residual" &&
                record.producer == "ptc_second_pass";
            const bool is_network =
                network_exclusion &&
                record.uid < 0 &&
                record.nw >= 0 &&
                record.reason == "busy_network_pathology" &&
                record.producer == "ptc_second_pass";
            if (is_mapdiag_detector || is_busy_detector || is_network) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    const bool has_network_record = std::any_of(
        records.begin(), records.end(),
        [](const auto &record) {
            return record.uid < 0 &&
                   record.reason == "busy_network_pathology";
        });
    summary.max_new_flagged_fraction = has_network_record
        ? reduction_learning.options.scan_network_pathology_max_new_flagged_fraction
        : reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    std::set<Eigen::Index> proposed_dets;
    std::set<Eigen::Index> network_proposed_dets;
    for (const auto &record : records) {
        if (record.uid >= 0) {
            const Eigen::Index det =
                citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
            if (det < 0 || det >= n_dets) {
                ++summary.invalid_records;
                continue;
            }
            ++summary.matched_records;
            proposed_dets.insert(det);
        }
        else if (record.nw >= 0) {
            bool matched_network = false;
            for (Eigen::Index det = 0; det < n_dets; ++det) {
                const int det_nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                if (det_nw == record.nw) {
                    matched_network = true;
                    proposed_dets.insert(det);
                    network_proposed_dets.insert(det);
                }
            }
            if (matched_network) {
                ++summary.matched_records;
            }
            else {
                ++summary.invalid_records;
            }
        }
    }
    if (proposed_dets.empty()) {
        reduction_learning.record_learned_mask_application(summary);
        return;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_network_source_protection = false;
    if (!network_proposed_dets.empty() &&
        stage == "pre_mapmaking_detector_exclusion") {
        const double radius_arcsec =
            std::max(20.0, ptcproc.second_pass_local.source_protection_radius_arcsec);
        auto [mask, source_info] = engine_utils::calc_source_protection_mask(
            tcdata, calib_scan.apt, telescope.pixel_axes, map_grouping,
            "map_center_radius", radius_arcsec);
        (void) source_info;
        source_mask = std::move(mask);
        have_network_source_protection =
            source_mask.rows() == n_pts && source_mask.cols() == n_dets;
        if (!have_network_source_protection) {
            logger->warn(
                "learned {} source-protection mask shape mismatch scan {}: mask=({}, {}) flags=({}, {})",
                stage, scan_id, source_mask.rows(), source_mask.cols(), n_pts, n_dets);
        }
    }

    for (const auto det : proposed_dets) {
        for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end() &&
                source_mask(sample, det)) {
                ++summary.source_protected_samples;
                continue;
            }
            ++summary.proposed_samples;
            if (tcdata.flags.data(sample, det)) {
                ++summary.already_flagged_samples;
            }
            else {
                ++summary.newly_flagged_samples;
            }
        }
    }

    const double denom = static_cast<double>(std::max<Eigen::Index>(1, n_pts * n_dets));
    summary.newly_flagged_fraction =
        static_cast<double>(summary.newly_flagged_samples) / denom;
    const bool over_cap =
        summary.max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            summary.max_new_flagged_fraction;
    if (!over_cap) {
        auto flag_it = calib_scan.apt.find("flag");
        std::set<Eigen::Index> apt_flag_dets;
        Eigen::Index apt_flag_preserved = 0;
        if (update_apt_flags &&
            flag_it != calib_scan.apt.end() &&
            flag_it->second.size() > 0) {
            std::map<int, Eigen::Index> unflagged_by_nw;
            std::map<int, Eigen::Index> unflagged_by_array;
            const Eigen::Index n_apt =
                std::min<Eigen::Index>(n_dets, flag_it->second.size());
            for (Eigen::Index det = 0; det < n_apt; ++det) {
                if (flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                if (nw >= 0) {
                    ++unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    ++unflagged_by_array[array];
                }
            }

            for (const auto det : proposed_dets) {
                if (network_proposed_dets.find(det) != network_proposed_dets.end()) {
                    continue;
                }
                if (det < 0 ||
                    det >= n_apt ||
                    flag_it->second(det) != 0.0) {
                    continue;
                }
                const int nw =
                    citlali_learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali_learning_apt_int(calib_scan.apt, "array", det, -1);
                const bool preserves_nw =
                    nw < 0 ||
                    unflagged_by_nw.find(nw) == unflagged_by_nw.end() ||
                    unflagged_by_nw[nw] > 1;
                const bool preserves_array =
                    array < 0 ||
                    unflagged_by_array.find(array) == unflagged_by_array.end() ||
                    unflagged_by_array[array] > 1;
                if (!preserves_nw || !preserves_array) {
                    ++apt_flag_preserved;
                    continue;
                }
                apt_flag_dets.insert(det);
                if (nw >= 0) {
                    --unflagged_by_nw[nw];
                }
                if (array >= 0) {
                    --unflagged_by_array[array];
                }
            }
        }

        for (const auto det : proposed_dets) {
            if (have_network_source_protection &&
                network_proposed_dets.find(det) != network_proposed_dets.end()) {
                for (Eigen::Index sample = 0; sample < n_pts; ++sample) {
                    if (!source_mask(sample, det)) {
                        tcdata.flags.data(sample, det) = true;
                    }
                }
            }
            else {
                tcdata.flags.data.col(det).setOnes();
            }
            if (apt_flag_dets.find(det) != apt_flag_dets.end()) {
                flag_it->second(det) = 1.0;
            }
        }
        summary.applied = true;
        if (apt_flag_preserved > 0) {
            logger->info(
                "learned {} preserved {} scan-local APT flags in scan {} iter {} to keep nw/array groups valid",
                stage, apt_flag_preserved, scan_id + 1, fruit_iter);
        }
    }

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} rejected scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            summary.max_new_flagged_fraction);
    }
    else {
        logger->info(
            "learned {} applied scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} already_flagged={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.newly_flagged_fraction);
    }
}

template <class tc_t, class calib_t>
void Engine::apply_learned_sample_masks(tc_t &tcdata, calib_t &calib_scan,
                                        bool apply_pre_rtc,
                                        const std::string &stage,
                                        bool source_protection_enabled,
                                        double source_protection_radius_arcsec) {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.options.apply_sample_masks_enabled ||
        !reduction_learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::LearnedSampleMask> records;
    {
        std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
        for (const auto &record : reduction_learning.learned_sample_masks) {
            if (record.obsnum == obsnum &&
                record.scan == scan_id &&
                record.iter >= 0 &&
                record.iter < fruit_iter &&
                record.apply_pre_rtc == apply_pre_rtc) {
                records.push_back(record);
            }
        }
    }
    if (records.empty()) {
        return;
    }

    ReductionLearningState::LearnedMaskApplicationSummary summary;
    summary.obsnum = obsnum;
    summary.producer = "learning_state";
    summary.stage = stage;
    summary.iter = fruit_iter;
    summary.scan = scan_id;
    summary.candidate_records = static_cast<int>(records.size());
    summary.max_new_flagged_fraction =
        reduction_learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> proposed(n_pts, n_dets);
    proposed.setZero();

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_source_protection = false;
    if (source_protection_enabled && source_protection_radius_arcsec > 0.0) {
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
        if (record.source_protected) {
            ++summary.invalid_records;
            continue;
        }
        const Eigen::Index det = citlali_learning_find_det_by_uid(calib_scan.apt, record.uid);
        const long long raw_start = apply_pre_rtc ? record.raw_start : record.ptc_start;
        const long long raw_stop = apply_pre_rtc ? record.raw_stop : record.ptc_stop;
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
        reduction_learning.options.apply_max_new_flagged_fraction > 0.0 &&
        summary.newly_flagged_fraction >
            reduction_learning.options.apply_max_new_flagged_fraction;
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

    reduction_learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} sample-mask application rejected scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            reduction_learning.options.apply_max_new_flagged_fraction);
    }
    else if (summary.proposed_samples > 0) {
        logger->info(
            "learned {} sample masks applied scan {} iter {}: candidates={} matched={} proposed={} newly_flagged={} already_flagged={} source_protected={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, fruit_iter, summary.candidate_records,
            summary.matched_records, summary.proposed_samples,
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.source_protected_samples, summary.newly_flagged_fraction);
    }
}

