#pragma once

// Included by learning_summary_csv.h inside namespace citlali::pipeline.

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_detector_penalty_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "detector_penalty", record.iter, record.obsnum,
        record.producer, record.reason, record.scan, record.uid, record.nw,
        record.array, text, csv);
    row[ColScore] = text(record.score);
    row[ColZ] = text(record.score);
    row[ColFactor] = text(record.factor);
    row[ColScanLocal] = text(record.scan_local ? 1 : 0);
    return row;
}

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_high_weight_detector_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "high_weight_detector", record.iter, record.obsnum,
        "weight_validation", record.reason, record.scan, record.uid,
        record.nw, record.array, text, csv);
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
    return row;
}

