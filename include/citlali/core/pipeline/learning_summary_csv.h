#pragma once

#include <cstddef>
#include <sstream>
#include <string>
#include <vector>

namespace citlali::pipeline {

namespace learning_summary_columns {

enum : std::size_t {
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

}  // namespace learning_summary_columns

inline std::string learning_summary_filename(
    const std::string &redu_dir_name, int fruit_iter) {
    std::ostringstream filename;
    filename << redu_dir_name << "/learning_iter_" << fruit_iter << ".csv";
    return filename.str();
}

inline std::vector<std::string> learning_summary_csv_header() {
    return {
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
}

inline std::vector<std::string> learning_summary_empty_row() {
    return std::vector<std::string>(
        learning_summary_columns::ColCount);
}

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

}  // namespace citlali::pipeline
