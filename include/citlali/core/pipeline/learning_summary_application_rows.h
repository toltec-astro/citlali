#pragma once

// Included by learning_summary_csv.h inside namespace citlali::pipeline.

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_mask_application_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    const bool detector_exclusion =
        record.stage.find("detector_exclusion") != std::string::npos;
    write_learning_summary_base_fields(
        row,
        detector_exclusion ? "detector_penalty_application"
                           : "sample_mask_application",
        record.iter, record.obsnum, record.producer,
        detector_exclusion ? "apply_learned_detector_exclusion"
                           : "apply_learned_sample_mask",
        record.scan, -1, -1, -1, text, csv);
    row[ColApplicationStage] = csv(record.stage);
    row[ColCandidateRecords] = text(record.candidate_records);
    row[ColMatchedRecords] = text(record.matched_records);
    row[ColInvalidRecords] = text(record.invalid_records);
    row[ColProposedSamples] = text(record.proposed_samples);
    row[ColNewlyFlaggedSamples] = text(record.newly_flagged_samples);
    row[ColAlreadyFlaggedSamples] = text(record.already_flagged_samples);
    row[ColSourceProtectedSamples] =
        text(record.source_protected_samples);
    row[ColNewlyFlaggedFraction] = text(record.newly_flagged_fraction);
    row[ColMaxNewFlaggedFraction] =
        text(record.max_new_flagged_fraction);
    row[ColApplied] = text(record.applied ? 1 : 0);
    return row;
}

