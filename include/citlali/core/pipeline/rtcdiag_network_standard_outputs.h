#pragma once

// Included by rtcdiag_network_outputs.h inside namespace citlali::pipeline.

template <class AddInt, class AddDouble>
void add_rtcdiag_standard_network_diag(const AddInt &add_int,
                                       const AddDouble &add_double) {
    add_rtcdiag_network_detector_count_diag(add_int);
    add_rtcdiag_network_line_audit_base_diag(add_int, add_double);
    add_rtcdiag_network_line_audit_diag(
        add_int, add_double, "rtc_network_post_line_audit", "post-filter");
    add_rtcdiag_network_step_summary_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_summary_diag(add_int, add_double);
    add_rtcdiag_network_common_mode_diag(add_double);
    add_rtcdiag_network_step_mask_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_window_diag(add_int, add_double);
    add_rtcdiag_network_impulsive_mask_trigger_diag(add_int);
    add_rtcdiag_network_impulsive_mask_candidate_diag(add_int, add_double);
}

inline void add_rtcdiag_standard_network_outputs(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_nw_values,
    int fill_int, double fill_double) {
    auto add_rtc_nw_double = [&](const std::string &name,
                                 const std::string &comment) {
        add_rtcdiag_network_double(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_double);
    };
    auto add_rtc_nw_int = [&](const std::string &name,
                              const std::string &comment) {
        add_rtcdiag_network_int(
            fo, name, comment, nw_dims, nw_chunks, n_nw_values,
            fill_int);
    };

    add_rtcdiag_standard_network_diag(add_rtc_nw_int, add_rtc_nw_double);
}

