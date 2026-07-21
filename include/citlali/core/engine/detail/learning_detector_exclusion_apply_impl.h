#pragma once

// Engine learned detector-exclusion application detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/map_grouping_policy.h>

template <class tc_t, class calib_t>
void Engine::apply_learned_detector_exclusions(tc_t &tcdata,
                                               calib_t &calib_scan,
                                               const std::string &stage,
                                               bool pre_rtc,
                                               bool update_apt_flags,
                                               bool include_detector_records,
                                               bool include_network_records) {
    if (!learning.is_enabled() ||
        !learning.apply_active()) {
        return;
    }
    if (tcdata.flags.data.rows() <= 0 || tcdata.flags.data.cols() <= 0) {
        return;
    }

    const bool mapdiag_detector_exclusion =
        include_detector_records &&
        learning.options.map_pixel_outlier_detector_exclusion_enabled;
    const bool busy_detector_exclusion =
        include_detector_records &&
        learning.options.busy_detector_exclusion_enabled;
    const bool network_exclusion =
        include_network_records &&
        learning.options.scan_network_pathology_enabled &&
        (stage == "pre_mapmaking_detector_exclusion"
             ? learning.options.scan_network_pathology_apply_pre_mapmaking
             : ((!pre_rtc && learning.options.scan_network_pathology_apply_pre_ptc) ||
                (pre_rtc && learning.options.scan_network_pathology_apply_pre_rtc)));
    if (!mapdiag_detector_exclusion && !busy_detector_exclusion &&
        !network_exclusion) {
        return;
    }

    const int scan_id = static_cast<int>(tcdata.index.data);
    std::vector<ReductionLearningState::DetectorPenalty> records;
    for (const auto &record :
         learning.effective_detector_penalty_records()) {
            if (record.obsnum != observation_identity.obsnum ||
                !record.scan_local ||
                record.scan != scan_id ||
                record.iter < 0 ||
                record.iter >= iteration.fruit_iter ||
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
    const bool has_network_record = std::any_of(
        records.begin(), records.end(),
        [](const auto &record) {
            return record.uid < 0 &&
                   record.reason == "busy_network_pathology";
        });
    summary.max_new_flagged_fraction = has_network_record
        ? learning.options.scan_network_pathology_max_new_flagged_fraction
        : learning.options.apply_max_new_flagged_fraction;

    const Eigen::Index n_pts = tcdata.flags.data.rows();
    const Eigen::Index n_dets = tcdata.flags.data.cols();
    std::set<Eigen::Index> proposed_dets;
    std::set<Eigen::Index> network_proposed_dets;
    for (const auto &record : records) {
        if (record.uid >= 0) {
            const Eigen::Index det =
                citlali::pipeline::learning_find_det_by_uid(calib_scan.apt, record.uid);
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
                    citlali::pipeline::learning_apt_int(calib_scan.apt, "nw", det, -1);
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
        learning.record_learned_mask_application(summary);
        return;
    }

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> source_mask;
    bool have_network_source_protection = false;
    if (!network_proposed_dets.empty() &&
        stage == "pre_mapmaking_detector_exclusion") {
        const auto &source_protection =
            citlali::pipeline::processed_time_chunk_config(*this)
                .flagging.second_pass_local.source_protection;
        const double radius_arcsec =
            std::max(20.0, source_protection.radius_arcsec);
        const auto map_grouping =
            citlali::pipeline::active_map_grouping_name(*this);
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
                    citlali::pipeline::learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali::pipeline::learning_apt_int(calib_scan.apt, "array", det, -1);
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
                    citlali::pipeline::learning_apt_int(calib_scan.apt, "nw", det, -1);
                const int array =
                    citlali::pipeline::learning_apt_int(calib_scan.apt, "array", det, -1);
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
                stage, apt_flag_preserved, scan_id + 1, iteration.fruit_iter);
        }
    }

    learning.record_learned_mask_application(summary);
    if (over_cap) {
        logger->warn(
            "learned {} rejected scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} newly_flagged_fraction={:.4f} cap={:.4f}",
            stage, scan_id + 1, iteration.fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.newly_flagged_fraction,
            summary.max_new_flagged_fraction);
    }
    else {
        logger->info(
            "learned {} applied scan {} iter {}: candidates={} matched={} dets={} newly_flagged={} already_flagged={} newly_flagged_fraction={:.4f}",
            stage, scan_id + 1, iteration.fruit_iter, summary.candidate_records,
            summary.matched_records, proposed_dets.size(),
            summary.newly_flagged_samples, summary.already_flagged_samples,
            summary.newly_flagged_fraction);
    }
}
