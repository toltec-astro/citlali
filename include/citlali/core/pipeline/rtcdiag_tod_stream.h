#pragma once

// Included by rtcdiag_netcdf.h inside namespace citlali::pipeline.

template <class Calib, class Rtcproc>
void add_rtcdiag_tod_stream_diag(netCDF::NcFile &fo, const Calib &calib,
                                 const Rtcproc &rtcproc,
                                 netCDF::NcDim n_scans_dim,
                                 netCDF::NcDim n_dets_dim,
                                 Eigen::Index n_scans,
                                 double sample_rate_hz,
                                 int fill_int,
                                 double fill_double) {
    const std::vector<std::size_t> no_chunks;
    std::vector<netCDF::NcDim> rtc_det_dims = {n_scans_dim, n_dets_dim};
    const auto n_det_values =
        static_cast<std::size_t>(n_scans) *
        static_cast<std::size_t>(calib.n_dets);

    auto add_det_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_det_double(
            fo, name, comment, rtc_det_dims, no_chunks,
            n_det_values, fill_double);
    };
    auto add_det_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_det_int(
            fo, name, comment, rtc_det_dims, no_chunks,
            n_det_values, fill_int);
    };

    add_rtcdiag_detector_core_diag(add_det_int, add_det_double);

    netCDF::NcDim n_nws_rtcdiag_dim =
        fo.addDim("n_nws_rtcdiag", calib.n_nws);
    add_rtcdiag_network_ids(fo, calib, n_nws_rtcdiag_dim, fill_int);

    std::vector<netCDF::NcDim> rtc_nw_dims = {
        n_scans_dim, n_nws_rtcdiag_dim};
    const auto n_nw_values =
        static_cast<std::size_t>(n_scans) *
        static_cast<std::size_t>(calib.n_nws);
    auto add_nw_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_network_double(
            fo, name, comment, rtc_nw_dims, no_chunks,
            n_nw_values, fill_double);
    };
    auto add_nw_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_network_int(
            fo, name, comment, rtc_nw_dims, no_chunks,
            n_nw_values, fill_int);
    };

    add_rtcdiag_standard_network_diag(add_nw_int, add_nw_double);

    if (!rtcproc.impulsive_capture.enabled) {
        return;
    }

    const auto n_slots = static_cast<std::size_t>(
        std::max<Eigen::Index>(rtcproc.impulsive_capture.max_events_per_network, 1));
    const auto snippet_pre =
        rtcdiag_impulsive_window_samples(
            rtcproc.impulsive_capture.snippet_pre_window_sec,
            sample_rate_hz);
    const auto snippet_post =
        rtcdiag_impulsive_window_samples(
            rtcproc.impulsive_capture.snippet_post_window_sec,
            sample_rate_hz);
    const auto n_snippet =
        rtcdiag_impulsive_snippet_sample_count(snippet_pre, snippet_post);
    netCDF::NcDim n_rtc_impulsive_slots_dim =
        fo.addDim("n_rtc_impulsive_slots", n_slots);
    netCDF::NcDim n_rtc_impulsive_samples_dim =
        fo.addDim("n_rtc_impulsive_samples", n_snippet);

    netCDF::NcVar offset_v =
        fo.addVar("rtc_impulsive_snippet_offset_samples", netCDF::ncInt,
                  n_rtc_impulsive_samples_dim);
    offset_v.putAtt("units", "samples");
    offset_v.putAtt(
        "comment", "sample offsets relative to rtc_impulsive_slot_event_sample");
    const auto offsets =
        rtcdiag_impulsive_snippet_offsets(n_snippet, snippet_pre, fill_int);
    offset_v.putVar(offsets.data());

    std::vector<netCDF::NcDim> slot_dims = {
        n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim};
    std::vector<netCDF::NcDim> snippet_dims = {
        n_scans_dim, n_nws_rtcdiag_dim, n_rtc_impulsive_slots_dim,
        n_rtc_impulsive_samples_dim};
    const auto n_slot_values = n_nw_values * n_slots;
    const auto n_snippet_values = n_slot_values * n_snippet;

    auto add_slot_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_slot_double(
            fo, name, comment, slot_dims, no_chunks,
            n_slot_values, fill_double);
    };
    auto add_slot_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_slot_int(
            fo, name, comment, slot_dims, no_chunks,
            n_slot_values, fill_int);
    };
    auto add_snippet_double = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_snippet_double(
            fo, name, comment, snippet_dims, no_chunks,
            n_snippet_values, fill_double);
    };
    auto add_snippet_int = [&](const std::string &name, const std::string &comment) {
        add_rtcdiag_impulsive_snippet_int(
            fo, name, comment, snippet_dims, no_chunks,
            n_snippet_values, fill_int);
    };

    add_rtcdiag_impulsive_capture_diag(
        add_slot_int, add_slot_double, add_snippet_double, add_snippet_int,
        rtcdiag_impulsive_capture_stream_comments());
}

