#pragma once

// Included by rtcdiag_impulsive_capture.h inside namespace citlali::pipeline.

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

