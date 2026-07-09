#pragma once

// Engine learning implementation detail.
// Include this only after Engine has been declared.

#include <citlali/core/pipeline/csv_output.h>
#include <citlali/core/pipeline/learning_summary_csv.h>

inline void Engine::write_learning_summary() {
    if (!reduction_learning.is_enabled() ||
        !reduction_learning.diagnostics_enabled() ||
        output_paths.redu_dir_name.empty()) {
        return;
    }

    const auto filename =
        citlali::pipeline::learning_summary_filename(output_paths.redu_dir_name, iteration.fruit_iter);
    std::ofstream out(filename);
    if (!out) {
        logger->warn("failed to open learning summary output {}", filename);
        return;
    }

    auto csv = citlali::pipeline::csv_escaped;

    const auto header = citlali::pipeline::learning_summary_csv_header();

    auto text = [](const auto &value) {
        return citlali::pipeline::csv_text(value);
    };

    auto write_row = [&](const std::vector<std::string> &row) {
        citlali::pipeline::write_csv_row(out, row);
    };

    auto write_common_header = [&]() {
        write_row(header);
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
        write_row(citlali::pipeline::learning_summary_mask_application_row(
            record, text, csv));
    }

    logger->info("wrote reduction learning summary {}", filename);
}
