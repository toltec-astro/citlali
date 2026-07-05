#pragma once

// Included by learning_summary_csv.h inside namespace citlali::pipeline.

template <class TextFormatter, class CsvFormatter>
void write_learning_summary_base_fields(
    std::vector<std::string> &row, const std::string &record_type, int iter,
    const std::string &obsnum_value, const std::string &producer,
    const std::string &reason, int scan, int uid, int nw, int array,
    TextFormatter &&text, CsvFormatter &&csv) {
    using namespace learning_summary_columns;
    row[ColRecordType] = csv(record_type);
    row[ColIter] = text(iter);
    row[ColObsnum] = csv(obsnum_value);
    row[ColProducer] = csv(producer);
    row[ColReason] = csv(reason);
    row[ColScan] = text(scan);
    row[ColUid] = text(uid);
    row[ColNw] = text(nw);
    row[ColArray] = text(array);
}

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_sample_mask_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "sample_mask", record.iter, record.obsnum, record.producer,
        record.reason, record.scan, record.uid, record.nw, record.array,
        text, csv);
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
    return row;
}

template <class Record, class TextFormatter, class CsvFormatter>
std::vector<std::string> learning_summary_busy_network_row(
    const Record &record, const TextFormatter &text,
    const CsvFormatter &csv) {
    using namespace learning_summary_columns;
    auto row = learning_summary_empty_row();
    write_learning_summary_base_fields(
        row, "busy_network", record.iter, record.obsnum, record.producer,
        record.reason, record.scan, -1, record.nw, -1, text, csv);
    row[ColScore] = text(record.top_candidate_score);
    row[ColZ] = text(record.max_unflagged_residual_z);
    row[ColCandidateClusters] = text(record.n_candidate_clusters);
    row[ColCandidateEvents] = text(record.n_candidate_events);
    row[ColAcceptedClusters] = text(record.n_accepted_clusters);
    row[ColAcceptedEvents] = text(record.n_accepted_events);
    row[ColRejectedClusters] = text(record.n_rejected_clusters);
    row[ColRejectedEvents] = text(record.n_rejected_events);
    row[ColSourceProtectedClusters] =
        text(record.n_source_protected_clusters);
    row[ColSourceProtectedEvents] =
        text(record.n_source_protected_events);
    row[ColMaxResidualUid] = text(record.max_unflagged_residual_uid);
    row[ColTopCandidateSample] = text(record.top_candidate_sample);
    row[ColTopCandidateScore] = text(record.top_candidate_score);
    row[ColMaxResidualZ] = text(record.max_unflagged_residual_z);
    row[ColBusyVetoed] = text(record.busy_vetoed ? 1 : 0);
    row[ColSelectiveAcceptanceRecommended] =
        text(record.selective_acceptance_recommended ? 1 : 0);
    return row;
}

