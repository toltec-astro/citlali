#pragma once

// Included by learning_summary_csv.h inside namespace citlali::pipeline.

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_map_pixel_outlier_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "map_pixel_outlier", record.iter, record.obsnum,
        record.producer, record.reason, record.scan, record.uid, -1, -1,
        text, csv);
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
    row[ColSourceDistanceArcsec] =
        text(record.source_distance_arcsec);
    row[ColSourceProtected] = text(record.source_protected ? 1 : 0);
    return row;
}

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_source_protection_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "source_protection", record.iter, record.obsnum,
        record.producer, record.mode, record.scan, -1, -1, -1, text, csv);
    row[ColSourceProtected] = text(1);
    row[ColApplyPreRtc] = text(0);
    row[ColProtectedSamples] = text(record.protected_samples);
    row[ColTotalSamples] = text(record.total_samples);
    row[ColRadiusArcsec] = text(record.radius_arcsec);
    row[ColSupportNpix] = text(record.support_npix);
    return row;
}

