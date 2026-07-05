#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

inline std::vector<int> rtcdiag_impulsive_snippet_offsets(
    std::size_t n_snippet, std::size_t snippet_pre, int fill_value) {
    std::vector<int> offsets(n_snippet, fill_value);
    for (std::size_t i=0; i<n_snippet; ++i) {
        offsets[i] = static_cast<int>(i) - static_cast<int>(snippet_pre);
    }
    return offsets;
}

inline std::size_t rtcdiag_impulsive_window_samples(
    double window_sec, double sample_rate_hz) {
    return static_cast<std::size_t>(
        std::max(0.0, std::round(window_sec * sample_rate_hz)));
}

inline std::size_t rtcdiag_impulsive_snippet_sample_count(
    std::size_t snippet_pre, std::size_t snippet_post) {
    return snippet_pre + snippet_post + 1;
}

inline void add_rtcdiag_impulsive_slot_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &slot_dims,
    const std::vector<std::size_t> &slot_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, slot_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, slot_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_slot_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &slot_dims,
    const std::vector<std::size_t> &slot_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, slot_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, slot_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_snippet_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &snippet_dims,
    const std::vector<std::size_t> &snippet_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, snippet_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, snippet_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_impulsive_snippet_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment,
    const std::vector<netCDF::NcDim> &snippet_dims,
    const std::vector<std::size_t> &snippet_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, snippet_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, snippet_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

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

template <class ImpulsiveCapture>
bool rtcdiag_impulsive_capture_requested(
    const ImpulsiveCapture &impulsive_capture) {
    return impulsive_capture.enabled;
}

inline void add_rtcdiag_impulsive_snippet_offset_var(
    netCDF::NcFile &fo, netCDF::NcDim n_samples_dim,
    std::size_t n_snippet, std::size_t snippet_pre, int fill_int) {
    netCDF::NcVar offset_v =
        fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt,
                  n_samples_dim);
    offset_v.putAtt("units", "samples");
    offset_v.putAtt(
        "comment",
        "sample offsets relative to rtc_impulsive_slot_event_sample");
    const auto offsets =
        rtcdiag_impulsive_snippet_offsets(n_snippet, snippet_pre, fill_int);
    offset_v.putVar(offsets.data());
}

template <class AddSlotInt, class AddSlotDouble, class AddSnippetDouble,
          class AddSnippetInt>
void add_rtcdiag_impulsive_capture_diag(
    const AddSlotInt &add_slot_int, const AddSlotDouble &add_slot_double,
    const AddSnippetDouble &add_snippet_double,
    const AddSnippetInt &add_snippet_int,
    const RtcDiagImpulsiveCaptureComments &comments) {
    add_slot_int("rtc_impulsive_slot_det_index",
                 "detector index of a captured impulsive RTC event for each scan/network/slot");
    add_slot_int("rtc_impulsive_slot_event_sample",
                 "sample index of a captured impulsive RTC event; -2147483647 means unavailable");
    add_slot_int("rtc_impulsive_slot_event_kind",
                 "0=raw-sample peak, 1=delta peak, -2147483647 means unavailable");
    add_slot_double("rtc_impulsive_slot_event_score",
                    "impulsive event score for a captured scan/network detector slot");
    add_slot_double("rtc_impulsive_slot_peak_abs_z", comments.peak_abs_z);
    add_slot_double("rtc_impulsive_slot_peak_delta_abs_z",
                    comments.peak_delta_abs_z);
    add_slot_double("rtc_impulsive_slot_added_flagged_frac",
                    comments.added_flagged_frac);
    add_slot_int("rtc_impulsive_slot_raw_exceed_count",
                 comments.raw_exceed_count);
    add_slot_int("rtc_impulsive_slot_local_raw_candidate_count",
                 comments.local_raw_candidate_count);
    add_slot_int("rtc_impulsive_slot_local_raw_accepted_event_count",
                 comments.local_raw_accepted_event_count);
    add_slot_int("rtc_impulsive_slot_local_flagged_sample_count",
                 comments.local_flagged_sample_count);
    add_slot_int("rtc_impulsive_slot_local_exceed_count",
                 "legacy alias for rtc_impulsive_slot_local_flagged_sample_count");
    add_slot_int("rtc_impulsive_slot_local_raw_reject_count",
                 comments.local_raw_reject_count);
    add_slot_int("rtc_impulsive_slot_delta_spike_count",
                 comments.delta_spike_count);
    add_slot_int("rtc_impulsive_slot_local_delta_candidate_count",
                 comments.local_delta_candidate_count);
    add_slot_int("rtc_impulsive_slot_local_delta_accepted_event_count",
                 comments.local_delta_accepted_event_count);
    add_slot_int("rtc_impulsive_slot_local_delta_exceed_count",
                 "legacy alias for rtc_impulsive_slot_local_delta_accepted_event_count");
    add_slot_int("rtc_impulsive_slot_local_delta_reject_count",
                 comments.local_delta_reject_count);
    add_snippet_double("rtc_impulsive_slot_snippet_z",
                       "standardized RTC snippet around each captured impulsive event");
    add_snippet_int("rtc_impulsive_slot_snippet_flag",
                    comments.snippet_flag);
}

template <class ImpulsiveCapture>
void add_rtcdiag_impulsive_capture_file_outputs_if_needed(
    netCDF::NcFile &fo, const ImpulsiveCapture &impulsive_capture,
    netCDF::NcDim n_scans_dim, netCDF::NcDim n_nws_dim,
    Eigen::Index n_scans, Eigen::Index n_networks, double sample_rate_hz,
    int fill_int, double fill_double) {
    if (!rtcdiag_impulsive_capture_requested(impulsive_capture)) {
        return;
    }

    const auto max_events_per_network =
        impulsive_capture.max_events_per_network;
    const auto n_slots =
        static_cast<std::size_t>(
            std::max<Eigen::Index>(max_events_per_network, 1));
    const double snippet_pre_window_sec =
        impulsive_capture.snippet_pre_window_sec;
    const auto snippet_pre =
        rtcdiag_impulsive_window_samples(
            snippet_pre_window_sec, sample_rate_hz);
    const double snippet_post_window_sec =
        impulsive_capture.snippet_post_window_sec;
    const auto snippet_post =
        rtcdiag_impulsive_window_samples(
            snippet_post_window_sec, sample_rate_hz);
    const auto n_snippet =
        rtcdiag_impulsive_snippet_sample_count(snippet_pre, snippet_post);
    netCDF::NcDim n_rtc_impulsive_slots_dim =
        fo.addDim("n_rtc_impulsive_slots", n_slots);
    netCDF::NcDim n_rtc_impulsive_samples_dim =
        fo.addDim("n_rtc_impulsive_samples", n_snippet);

    add_rtcdiag_impulsive_snippet_offset_var(
        fo, n_rtc_impulsive_samples_dim, n_snippet, snippet_pre, fill_int);

    const auto n_impulsive_networks =
        static_cast<std::size_t>(n_networks);
    std::vector<netCDF::NcDim> rtc_impulsive_slot_dims = {
        n_scans_dim, n_nws_dim, n_rtc_impulsive_slots_dim};
    std::vector<netCDF::NcDim> rtc_impulsive_snippet_dims = {
        n_scans_dim, n_nws_dim, n_rtc_impulsive_slots_dim,
        n_rtc_impulsive_samples_dim};
    const std::vector<std::size_t> rtc_impulsive_slot_chunks = {
        1, n_impulsive_networks, n_slots};
    const std::vector<std::size_t> rtc_impulsive_snippet_chunks = {
        1, n_impulsive_networks, n_slots, n_snippet};
    const auto n_rtc_impulsive_slot_values =
        static_cast<std::size_t>(n_scans) * n_impulsive_networks * n_slots;
    const auto n_rtc_impulsive_snippet_values =
        n_rtc_impulsive_slot_values * n_snippet;

    auto add_rtc_imp_slot_double = [&](const std::string &name,
                                       const std::string &comment) {
        add_rtcdiag_impulsive_slot_double(
            fo, name, comment, rtc_impulsive_slot_dims,
            rtc_impulsive_slot_chunks, n_rtc_impulsive_slot_values,
            fill_double);
    };
    auto add_rtc_imp_slot_int = [&](const std::string &name,
                                    const std::string &comment) {
        add_rtcdiag_impulsive_slot_int(
            fo, name, comment, rtc_impulsive_slot_dims,
            rtc_impulsive_slot_chunks, n_rtc_impulsive_slot_values, fill_int);
    };
    auto add_rtc_imp_snip_double = [&](const std::string &name,
                                       const std::string &comment) {
        add_rtcdiag_impulsive_snippet_double(
            fo, name, comment, rtc_impulsive_snippet_dims,
            rtc_impulsive_snippet_chunks, n_rtc_impulsive_snippet_values,
            fill_double);
    };
    auto add_rtc_imp_snip_int = [&](const std::string &name,
                                    const std::string &comment) {
        add_rtcdiag_impulsive_snippet_int(
            fo, name, comment, rtc_impulsive_snippet_dims,
            rtc_impulsive_snippet_chunks, n_rtc_impulsive_snippet_values,
            fill_int);
    };

    add_rtcdiag_impulsive_capture_diag(
        add_rtc_imp_slot_int, add_rtc_imp_slot_double,
        add_rtc_imp_snip_double, add_rtc_imp_snip_int,
        rtcdiag_impulsive_capture_file_comments());
}

