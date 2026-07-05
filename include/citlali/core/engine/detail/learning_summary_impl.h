#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

inline void Engine::write_learning_summary() {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        redu_dir_name.empty()) {
        return;
    }

    std::ostringstream filename;
    filename << redu_dir_name << "/learning_iter_" << fruit_iter << ".csv";
    std::ofstream out(filename.str());
    if (!out) {
        logger->warn("failed to open learning summary output {}", filename.str());
        return;
    }

    auto csv = [](const std::string &s) {
        std::string escaped = "\"";
        for (const char ch : s) {
            if (ch == '"') {
                escaped += "\"\"";
            }
            else {
                escaped += ch;
            }
        }
        escaped += "\"";
        return escaped;
    };

    enum {
        ColRecordType,
        ColIter,
        ColObsnum,
        ColProducer,
        ColReason,
        ColScan,
        ColUid,
        ColNw,
        ColArray,
        ColRawStart,
        ColRawStop,
        ColPtcStart,
        ColPtcStop,
        ColScore,
        ColZ,
        ColValue,
        ColConfidence,
        ColSourceDistanceArcsec,
        ColSourceProtected,
        ColApplyPreRtc,
        ColCandidateClusters,
        ColCandidateEvents,
        ColAcceptedClusters,
        ColAcceptedEvents,
        ColRejectedClusters,
        ColRejectedEvents,
        ColSourceProtectedClusters,
        ColSourceProtectedEvents,
        ColMaxResidualUid,
        ColTopCandidateSample,
        ColTopCandidateScore,
        ColMaxResidualZ,
        ColBusyVetoed,
        ColSelectiveAcceptanceRecommended,
        ColFactor,
        ColScanLocal,
        ColProtectedSamples,
        ColTotalSamples,
        ColRadiusArcsec,
        ColSupportNpix,
        ColApplicationStage,
        ColCandidateRecords,
        ColMatchedRecords,
        ColInvalidRecords,
        ColProposedSamples,
        ColNewlyFlaggedSamples,
        ColAlreadyFlaggedSamples,
        ColSourceProtectedSamples,
        ColNewlyFlaggedFraction,
        ColMaxNewFlaggedFraction,
        ColApplied,
        ColGrouping,
        ColWeight,
        ColFinalWeight,
        ColGroupMedian,
        ColRobustZ,
        ColCap,
        ColValidationFactor,
        ColCapRecommended,
        ColCapApplied,
        ColValidated,
        ColMapIndex,
        ColRow,
        ColCol,
        ColSample,
        ColNEff,
        ColLeaveOneOutZ,
        ColCount
    };

    const std::vector<std::string> header = {
        "record_type", "iter", "obsnum", "producer", "reason", "scan", "uid",
        "nw", "array", "raw_start", "raw_stop", "ptc_start", "ptc_stop",
        "score", "z", "value", "confidence", "source_distance_arcsec",
        "source_protected", "apply_pre_rtc", "n_candidate_clusters",
        "n_candidate_events", "n_accepted_clusters", "n_accepted_events",
        "n_rejected_clusters", "n_rejected_events",
        "n_source_protected_clusters", "n_source_protected_events",
        "max_unflagged_residual_uid", "top_candidate_sample",
        "top_candidate_score", "max_unflagged_residual_z", "busy_vetoed",
        "selective_acceptance_recommended", "factor", "scan_local",
        "protected_samples", "total_samples", "radius_arcsec", "support_npix",
        "application_stage", "candidate_records", "matched_records",
        "invalid_records", "proposed_samples", "newly_flagged_samples",
        "already_flagged_samples", "source_protected_samples",
        "newly_flagged_fraction", "max_new_flagged_fraction", "applied",
        "grouping", "weight", "final_weight", "group_median", "robust_z",
        "cap", "validation_factor", "cap_recommended", "cap_applied",
        "validated", "map_index", "row", "col", "sample", "n_eff",
        "leave_one_out_z"
    };

    auto text = [](const auto &value) {
        std::ostringstream stream;
        stream << value;
        return stream.str();
    };

    auto write_row = [&](const std::vector<std::string> &row) {
        for (std::size_t i = 0; i < row.size(); ++i) {
            if (i > 0) {
                out << ',';
            }
            out << row[i];
        }
        out << '\n';
    };

    auto new_row = [&]() {
        return std::vector<std::string>(ColCount);
    };

    auto write_common_header = [&]() {
        write_row(header);
    };

    auto write_base = [&](std::vector<std::string> &row,
                          const std::string &record_type, int iter,
                          const std::string &obsnum_value,
                          const std::string &producer,
                          const std::string &reason, int scan, int uid,
                          int nw, int array) {
        row[ColRecordType] = csv(record_type);
        row[ColIter] = text(iter);
        row[ColObsnum] = csv(obsnum_value);
        row[ColProducer] = csv(producer);
        row[ColReason] = csv(reason);
        row[ColScan] = text(scan);
        row[ColUid] = text(uid);
        row[ColNw] = text(nw);
        row[ColArray] = text(array);
    };

    std::lock_guard<std::mutex> lock(*reduction_learning.mutex);
    write_common_header();

    for (const auto &record : reduction_learning.learned_sample_masks) {
        auto row = new_row();
        write_base(row, "sample_mask", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, record.uid, record.nw, record.array);
        row[ColRawStart] = text(record.raw_start);
        row[ColRawStop] = text(record.raw_stop);
        row[ColPtcStart] = text(record.ptc_start);
        row[ColPtcStop] = text(record.ptc_stop);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.z);
        row[ColValue] = text(record.value);
        row[ColConfidence] = text(record.confidence);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        row[ColApplyPreRtc] = text(record.apply_pre_rtc ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.busy_network_summaries) {
        auto row = new_row();
        write_base(row, "busy_network", record.iter, record.obsnum, record.producer,
                   record.reason, record.scan, -1, record.nw, -1);
        row[ColScore] = text(record.top_candidate_score);
        row[ColZ] = text(record.max_unflagged_residual_z);
        row[ColCandidateClusters] = text(record.n_candidate_clusters);
        row[ColCandidateEvents] = text(record.n_candidate_events);
        row[ColAcceptedClusters] = text(record.n_accepted_clusters);
        row[ColAcceptedEvents] = text(record.n_accepted_events);
        row[ColRejectedClusters] = text(record.n_rejected_clusters);
        row[ColRejectedEvents] = text(record.n_rejected_events);
        row[ColSourceProtectedClusters] = text(record.n_source_protected_clusters);
        row[ColSourceProtectedEvents] = text(record.n_source_protected_events);
        row[ColMaxResidualUid] = text(record.max_unflagged_residual_uid);
        row[ColTopCandidateSample] = text(record.top_candidate_sample);
        row[ColTopCandidateScore] = text(record.top_candidate_score);
        row[ColMaxResidualZ] = text(record.max_unflagged_residual_z);
        row[ColBusyVetoed] = text(record.busy_vetoed ? 1 : 0);
        row[ColSelectiveAcceptanceRecommended] =
            text(record.selective_acceptance_recommended ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.detector_penalties) {
        auto row = new_row();
        write_base(row, "detector_penalty", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.score);
        row[ColZ] = text(record.score);
        row[ColFactor] = text(record.factor);
        row[ColScanLocal] = text(record.scan_local ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.high_weight_detectors) {
        auto row = new_row();
        write_base(row, "high_weight_detector", record.iter, record.obsnum,
                   "weight_validation", record.reason, record.scan, record.uid,
                   record.nw, record.array);
        row[ColScore] = text(record.robust_z);
        row[ColZ] = text(record.robust_z);
        row[ColValue] = text(record.weight);
        row[ColFactor] = text(record.validation_factor);
        row[ColGrouping] = csv(record.grouping);
        row[ColWeight] = text(record.weight);
        row[ColFinalWeight] = text(record.final_weight);
        row[ColGroupMedian] = text(record.group_median);
        row[ColRobustZ] = text(record.robust_z);
        row[ColCap] = text(record.cap);
        row[ColValidationFactor] = text(record.validation_factor);
        row[ColCapRecommended] = text(record.cap_recommended ? 1 : 0);
        row[ColCapApplied] = text(record.cap_applied ? 1 : 0);
        row[ColValidated] = text(record.validated ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.map_pixel_outliers) {
        auto row = new_row();
        write_base(row, "map_pixel_outlier", record.iter, record.obsnum,
                   record.producer, record.reason, record.scan, record.uid,
                   -1, -1);
        row[ColScore] = text(record.leave_one_out_z);
        row[ColZ] = text(record.leave_one_out_z);
        row[ColValue] = text(record.value);
        row[ColWeight] = text(record.weight);
        row[ColMapIndex] = text(record.map_index);
        row[ColRow] = text(record.row);
        row[ColCol] = text(record.col);
        row[ColSample] = text(record.sample);
        row[ColNEff] = text(record.n_eff);
        row[ColLeaveOneOutZ] = text(record.leave_one_out_z);
        row[ColSourceDistanceArcsec] = text(record.source_distance_arcsec);
        row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
        write_row(row);
    }

    for (const auto &record : reduction_learning.source_protection_summaries) {
        auto row = new_row();
        write_base(row, "source_protection", record.iter, record.obsnum,
                   record.producer, record.mode, record.scan, -1, -1, -1);
        row[ColSourceProtected] = text(1);
        row[ColApplyPreRtc] = text(0);
        row[ColProtectedSamples] = text(record.protected_samples);
        row[ColTotalSamples] = text(record.total_samples);
        row[ColRadiusArcsec] = text(record.radius_arcsec);
        row[ColSupportNpix] = text(record.support_npix);
        write_row(row);
    }

    for (const auto &record : reduction_learning.learned_mask_applications) {
        auto row = new_row();
        const bool detector_exclusion =
            record.stage.find("detector_exclusion") != std::string::npos;
        write_base(row,
                   detector_exclusion
                       ? "detector_penalty_application"
                       : "sample_mask_application",
                   record.iter, record.obsnum, record.producer,
                   detector_exclusion
                       ? "apply_learned_detector_exclusion"
                       : "apply_learned_sample_mask",
                   record.scan, -1, -1, -1);
        row[ColApplicationStage] = csv(record.stage);
        row[ColCandidateRecords] = text(record.candidate_records);
        row[ColMatchedRecords] = text(record.matched_records);
        row[ColInvalidRecords] = text(record.invalid_records);
        row[ColProposedSamples] = text(record.proposed_samples);
        row[ColNewlyFlaggedSamples] = text(record.newly_flagged_samples);
        row[ColAlreadyFlaggedSamples] = text(record.already_flagged_samples);
        row[ColSourceProtectedSamples] = text(record.source_protected_samples);
        row[ColNewlyFlaggedFraction] = text(record.newly_flagged_fraction);
        row[ColMaxNewFlaggedFraction] = text(record.max_new_flagged_fraction);
        row[ColApplied] = text(record.applied ? 1 : 0);
        write_row(row);
    }

    logger->info("wrote reduction learning summary {}", filename.str());
}
