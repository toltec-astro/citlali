#pragma once

// Included by rtcdiag_impulsive_capture.h inside namespace citlali::pipeline.

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

