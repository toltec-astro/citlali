#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/csv_output.h>
#include <citlali/core/pipeline/learning_summary_csv.h>

inline void Engine::write_learning_summary() {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        redu_dir_name.empty()) {
        return;
    }

    const auto filename =
        citlali::pipeline::learning_summary_filename(redu_dir_name, fruit_iter);
    std::ofstream out(filename);
    if (!out) {
        logger->warn("failed to open learning summary output {}", filename);
        return;
    }

    auto csv = citlali::pipeline::csv_escaped;
    using namespace citlali::pipeline::learning_summary_columns;

    const auto header = citlali::pipeline::learning_summary_csv_header();

    auto text = [](const auto &value) {
        return citlali::pipeline::csv_text(value);
    };

    auto write_row = [&](const std::vector<std::string> &row) {
        citlali::pipeline::write_csv_row(out, row);
    };

    auto new_row = [&]() {
        return citlali::pipeline::learning_summary_empty_row();
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
        citlali::pipeline::write_learning_summary_base_fields(
            row, record_type, iter, obsnum_value, producer, reason, scan,
            uid, nw, array, text, csv);
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

    logger->info("wrote reduction learning summary {}", filename);
}
