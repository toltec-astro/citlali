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
        write_row(citlali::pipeline::learning_summary_sample_mask_row(
            record, text, csv));
    }

    for (const auto &record : reduction_learning.busy_network_summaries) {
        write_row(citlali::pipeline::learning_summary_busy_network_row(
            record, text, csv));
    }

    for (const auto &record : reduction_learning.detector_penalties) {
        write_row(citlali::pipeline::learning_summary_detector_penalty_row(
            record, text, csv));
    }

    for (const auto &record : reduction_learning.high_weight_detectors) {
        write_row(
            citlali::pipeline::learning_summary_high_weight_detector_row(
                record, text, csv));
    }

    for (const auto &record : reduction_learning.map_pixel_outliers) {
        write_row(citlali::pipeline::learning_summary_map_pixel_outlier_row(
            record, text, csv));
    }

    for (const auto &record : reduction_learning.source_protection_summaries) {
        write_row(citlali::pipeline::learning_summary_source_protection_row(
            record, text, csv));
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
