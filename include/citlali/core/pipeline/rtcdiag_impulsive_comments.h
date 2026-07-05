#pragma once

// Included by rtcdiag_impulsive_capture.h inside namespace citlali::pipeline.

struct RtcDiagImpulsiveCaptureComments {
    std::string peak_abs_z;
    std::string peak_delta_abs_z;
    std::string added_flagged_frac;
    std::string raw_exceed_count;
    std::string local_raw_candidate_count;
    std::string local_raw_accepted_event_count;
    std::string local_flagged_sample_count;
    std::string local_raw_reject_count;
    std::string delta_spike_count;
    std::string local_delta_candidate_count;
    std::string local_delta_accepted_event_count;
    std::string local_delta_reject_count;
    std::string snippet_flag;
};

inline RtcDiagImpulsiveCaptureComments
rtcdiag_impulsive_capture_stream_comments() {
    return {
        "maximum per-sample absolute robust-z for a captured scan/network detector slot",
        "maximum adjacent-sample delta robust-z for a captured scan/network detector slot",
        "fraction of samples newly flagged by RTC despiking for a captured detector slot",
        "count of raw-sample MAD exceedances for a captured detector slot",
        "count of locally detrended raw candidate events considered by the compact-raw gate for a captured detector slot",
        "count of locally detrended raw candidate events accepted by the compact-raw gate for a captured detector slot",
        "count of samples flagged by accepted compact-raw local-residual events for a captured detector slot",
        "count of locally detrended raw candidate events rejected by the compact-raw gate for a captured detector slot",
        "count of delta-domain spikes for a captured detector slot",
        "count of locally detrended delta candidate events considered by the compact-delta gate for a captured detector slot",
        "count of locally detrended delta candidate events accepted by the compact-delta gate for a captured detector slot",
        "count of locally detrended delta candidate events rejected by the compact-delta gate for a captured detector slot",
        "final RTC flag state for each sample in a captured impulsive snippet",
    };
}

inline RtcDiagImpulsiveCaptureComments
rtcdiag_impulsive_capture_file_comments() {
    return {
        "absolute robust-z peak of a captured impulsive RTC event",
        "absolute delta robust-z peak of a captured impulsive RTC event",
        "newly added flagged-sample fraction for the captured detector",
        "native raw-threshold exceedance count for the captured detector",
        "compact-raw local candidate count for the captured detector",
        "accepted compact-raw local-event count for the captured detector",
        "samples flagged by accepted compact-raw local events for the captured detector",
        "rejected compact-raw local-event count for the captured detector",
        "native delta-spike count for the captured detector",
        "compact-delta local candidate count for the captured detector",
        "accepted compact-delta local-event count for the captured detector",
        "rejected compact-delta local-event count for the captured detector",
        "RTC flag state for each sample in the captured impulsive-event snippet",
    };
}

