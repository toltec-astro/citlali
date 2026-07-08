#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

template <class AddInt, class AddDouble>
void add_tod_filter_edge_guard_scan_vars(const AddInt &add_int,
                                         const AddDouble &add_double) {
    add_int("tod_filter_edge_guard_pre_samples",
            "samples flagged at the start of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_post_samples",
            "samples flagged at the end of this output scan by the TOD filter edge guard");
    add_int("tod_filter_edge_guard_flagged_samples",
            "detector-samples flagged by the TOD filter edge guard");
    add_double("tod_filter_edge_guard_flagged_frac", "N/A",
               "fraction of time samples guarded at this output scan edge");
}

inline void add_tod_filter_edge_guard_scan_placeholders(
    netCDF::NcFile &fo, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, int fill_int, double fill_double) {
    auto add_scan_int_var = [&](const std::string &name,
                                const std::string &comment) {
        add_tod_scan_int_placeholder_var(
            fo, name, comment, n_scans_dim, n_output_scans, fill_int);
    };
    auto add_scan_double_var = [&](const std::string &name,
                                   const std::string &units,
                                   const std::string &comment) {
        add_tod_scan_double_placeholder_var(
            fo, name, units, comment, n_scans_dim, n_output_scans,
            fill_double);
    };
    add_tod_filter_edge_guard_scan_vars(add_scan_int_var,
                                        add_scan_double_var);
}

template <class RtcProc, class PtcProc, class ScanIndices>
TodPreparedLayout prepare_tod_file_layout(
    netCDF::NcFile &fo, citlali::config::TodOutputStream stream,
    Eigen::Index n_rtc_output_scans, Eigen::Index n_ptc_output_scans,
    const RtcProc &rtcproc, const PtcProc &ptcproc,
    const ScanIndices &scan_indices, Eigen::Index n_dets) {
    auto stream_layout = tod_stream_layout(
        stream, n_rtc_output_scans, n_ptc_output_scans, rtcproc,
        ptcproc);
    auto counts = tod_file_counts(
        stream_layout.n_output_scans, scan_indices.rows(), n_dets);
    auto dims = add_tod_file_dims(
        fo, counts.n_output_scans, counts.n_raw_scan_indices, counts.n_dets);
    add_tod_scan_index_placeholders(
        fo, dims.raw_scans, dims.scans, dims.n_scans,
        counts.n_output_scans, counts.n_raw_scan_indices,
        stream_layout.outer_output, tod_output_fill_int());
    add_tod_filter_edge_guard_scan_placeholders(
        fo, dims.n_scans, counts.n_output_scans, tod_output_fill_int(),
        tod_output_fill_double());
    auto chunking = tod_data_chunking(scan_indices, counts.n_dets);

    return {stream_layout, counts, dims, chunking};
}
