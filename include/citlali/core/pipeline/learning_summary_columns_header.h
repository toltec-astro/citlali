#pragma once

// Included by learning_summary_csv.h inside namespace citlali::pipeline.

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

