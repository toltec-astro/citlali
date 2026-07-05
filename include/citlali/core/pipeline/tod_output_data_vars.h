#pragma once

// TOD output NetCDF metadata implementation detail.
// Include this only from output_netcdf_metadata.h inside citlali::pipeline.

inline void add_tod_scan_index_placeholders(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &raw_scans_dims,
    const std::vector<netCDF::NcDim> &scans_dims,
    netCDF::NcDim n_scans_dim, std::size_t n_output_scans,
    std::size_t n_raw_scan_indices, bool tod_output_outer, int fill_value) {
    netCDF::NcVar raw_scan_indices_v =
        fo.addVar("raw_scan_indices", netCDF::ncInt, raw_scans_dims);
    raw_scan_indices_v.putAtt("units", "N/A");
    raw_scan_indices_v.putAtt(
        "comment",
        tod_output_outer
            ? "indices in output timebase: inner_start, inner_end, outer_start, outer_end"
            : "indices in output timebase; outer=inner (output stores inner scans only)");
    std::vector<int> raw_scan_init(n_output_scans * n_raw_scan_indices,
                                   fill_value);
    raw_scan_indices_v.putVar(raw_scan_init.data());

    netCDF::NcVar scan_indices_v =
        fo.addVar("scan_indices", netCDF::ncInt, scans_dims);
    scan_indices_v.putAtt("units", "N/A");
    std::vector<int> scan_init(n_output_scans * 2, fill_value);
    scan_indices_v.putVar(scan_init.data());

    netCDF::NcVar output_scan_index_v =
        fo.addVar("output_scan_index", netCDF::ncInt, n_scans_dim);
    output_scan_index_v.putAtt("units", "N/A");
    output_scan_index_v.putAtt(
        "comment", "1-based original scan index from the full observation");
    std::vector<int> output_scan_init(n_output_scans, fill_value);
    output_scan_index_v.putVar(output_scan_init.data());
}

inline void add_tod_scan_int_placeholder_var(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_scans_dim);
    v.putAtt("units", "samples");
    v.putAtt("comment", comment);
    std::vector<int> init(n_output_scans, fill_value);
    v.putVar(init.data());
}

inline void add_tod_scan_double_placeholder_var(
    netCDF::NcFile &fo, const std::string &name, const std::string &units,
    const std::string &comment, netCDF::NcDim n_scans_dim,
    std::size_t n_output_scans, double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, n_scans_dim);
    v.putAtt("units", units);
    v.putAtt("comment", comment);
    std::vector<double> init(n_output_scans, fill_value);
    v.putVar(init.data());
}

inline void set_tod_var_chunking(
    netCDF::NcVar &var, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    auto chunks = chunk_sizes;
    var.setChunking(chunk_mode, chunks);
}

inline void add_tod_signal_var(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &signal_unit,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    netCDF::NcVar signal_v;
    if (mini_output) {
        signal_v = fo.addVar("signal", netCDF::ncFloat, dims);
    }
    else {
        signal_v = fo.addVar("signal", netCDF::ncDouble, dims);
    }
    signal_v.putAtt("units", signal_unit);
    set_tod_var_chunking(signal_v, chunk_mode, chunk_sizes);
}

inline void add_tod_flags_var(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    netCDF::NcVar flags_v;
    if (mini_output) {
        flags_v = fo.addVar("flags", netCDF::ncByte, dims);
    }
    else {
        flags_v = fo.addVar("flags", netCDF::ncDouble, dims);
    }
    flags_v.putAtt("units", "N/A");
    if (mini_output) {
        flags_v.putAtt("comment", "0=good,1=flagged");
    }
    set_tod_var_chunking(flags_v, chunk_mode, chunk_sizes);
}

inline void add_tod_kernel_var_if_requested(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool run_kernel, bool mini_output,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    if (!run_kernel || mini_output) {
        return;
    }
    netCDF::NcVar kernel_v = fo.addVar("kernel", netCDF::ncDouble, dims);
    kernel_v.putAtt("units", "N/A");
    set_tod_var_chunking(kernel_v, chunk_mode, chunk_sizes);
}

inline void add_tod_detector_pointing_vars(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &pixel_axes,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    if (mini_output) {
        return;
    }

    netCDF::NcVar det_lat_v = fo.addVar("det_lat", netCDF::ncDouble, dims);
    det_lat_v.putAtt("units", "rad");
    set_tod_var_chunking(det_lat_v, chunk_mode, chunk_sizes);

    netCDF::NcVar det_lon_v = fo.addVar("det_lon", netCDF::ncDouble, dims);
    det_lon_v.putAtt("units", "rad");
    set_tod_var_chunking(det_lon_v, chunk_mode, chunk_sizes);

    if (pixel_axes == "radec") {
        netCDF::NcVar det_ra_v = fo.addVar("det_ra", netCDF::ncDouble, dims);
        det_ra_v.putAtt("units", "rad");
        set_tod_var_chunking(det_ra_v, chunk_mode, chunk_sizes);

        netCDF::NcVar det_dec_v =
            fo.addVar("det_dec", netCDF::ncDouble, dims);
        det_dec_v.putAtt("units", "rad");
        set_tod_var_chunking(det_dec_v, chunk_mode, chunk_sizes);
    }
}

inline void add_tod_core_data_vars(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &dims,
    bool mini_output, const std::string &signal_unit, bool run_kernel,
    const std::string &pixel_axes, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    add_tod_signal_var(fo, dims, mini_output, signal_unit, chunk_mode,
                       chunk_sizes);
    add_tod_flags_var(fo, dims, mini_output, chunk_mode, chunk_sizes);
    add_tod_kernel_var_if_requested(
        fo, dims, run_kernel, mini_output, chunk_mode, chunk_sizes);
    add_tod_detector_pointing_vars(
        fo, dims, mini_output, pixel_axes, chunk_mode, chunk_sizes);
}

template <class AptTable, class AptUnits>
void add_tod_apt_table_vars(netCDF::NcFile &fo, const AptTable &apt,
                            const AptUnits &apt_header_units,
                            netCDF::NcDim n_dets_dim) {
    for (const auto &item : apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + item.first, netCDF::ncDouble, n_dets_dim);
        const auto units_it = apt_header_units.find(item.first);
        const std::string units =
            (units_it == apt_header_units.end()) ? "" : units_it->second;
        apt_v.putAtt("units", units);
    }
}

template <class TelescopeData>
void add_telescope_data_vars(
    netCDF::NcFile &fo, const TelescopeData &tel_data,
    netCDF::NcDim n_pts_dim, netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : tel_data) {
        netCDF::NcVar tel_data_v =
            fo.addVar(item.first, netCDF::ncDouble, n_pts_dim);
        tel_data_v.putAtt("units", "rad");
        set_tod_var_chunking(tel_data_v, chunk_mode, chunk_sizes);
    }
}

template <class PointingOffsets, class Logger>
void add_tod_pointing_offset_vars(
    netCDF::NcFile &fo, const PointingOffsets &pointing_offsets_arcsec,
    const Logger &logger, netCDF::NcDim n_pts_dim,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    for (const auto &item : pointing_offsets_arcsec) {
        logger->info("pointing_offsets_arcsec.second {} {}", item.first,
                     item.second);
        netCDF::NcVar offsets_v = fo.addVar(
            "pointing_offset_" + item.first, netCDF::ncDouble, n_pts_dim);
        offsets_v.putAtt("units", "arcsec");
        set_tod_var_chunking(offsets_v, chunk_mode, chunk_sizes);
    }
}

template <class AptTable, class AptUnits, class TelescopeData,
          class PointingOffsets, class Logger>
void add_tod_static_metadata_vars(
    netCDF::NcFile &fo, const AptTable &apt, const AptUnits &apt_header_units,
    const TelescopeData &tel_data, const PointingOffsets &pointing_offsets,
    const Logger &logger, netCDF::NcDim n_dets_dim, netCDF::NcDim n_pts_dim,
    netCDF::NcVar::ChunkMode chunk_mode,
    const std::vector<std::size_t> &chunk_sizes) {
    add_tod_apt_table_vars(fo, apt, apt_header_units, n_dets_dim);
    add_telescope_data_vars(fo, tel_data, n_pts_dim, chunk_mode,
                            chunk_sizes);
    add_tod_pointing_offset_vars(
        fo, pointing_offsets, logger, n_pts_dim, chunk_mode, chunk_sizes);
}

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
    netCDF::NcFile &fo, bool is_rtc_stream,
    Eigen::Index n_rtc_output_scans, Eigen::Index n_ptc_output_scans,
    const RtcProc &rtcproc, const PtcProc &ptcproc,
    const ScanIndices &scan_indices, Eigen::Index n_dets) {
    auto stream = tod_stream_layout(
        is_rtc_stream, n_rtc_output_scans, n_ptc_output_scans, rtcproc,
        ptcproc);
    auto counts = tod_file_counts(
        stream.n_output_scans, scan_indices.rows(), n_dets);
    auto dims = add_tod_file_dims(
        fo, counts.n_output_scans, counts.n_raw_scan_indices, counts.n_dets);
    add_tod_scan_index_placeholders(
        fo, dims.raw_scans, dims.scans, dims.n_scans,
        counts.n_output_scans, counts.n_raw_scan_indices,
        stream.outer_output, tod_output_fill_int());
    add_tod_filter_edge_guard_scan_placeholders(
        fo, dims.n_scans, counts.n_output_scans, tod_output_fill_int(),
        tod_output_fill_double());
    auto chunking = tod_data_chunking(scan_indices, counts.n_dets);

    return {stream, counts, dims, chunking};
}

inline void add_tod_hwpr_var(netCDF::NcFile &fo, netCDF::NcDim n_pts_dim) {
    netCDF::NcVar hwpr_v = fo.addVar("hwpr", netCDF::ncDouble, n_pts_dim);
    hwpr_v.putAtt("units", "rad");
}

inline void add_tod_hwpr_var_if_requested(netCDF::NcFile &fo,
                                          bool run_polarization,
                                          bool run_hwpr,
                                          netCDF::NcDim n_pts_dim) {
    if (run_polarization && run_hwpr) {
        add_tod_hwpr_var(fo, n_pts_dim);
    }
}

template <class TelescopeHeader>
void add_telescope_header_vars(netCDF::NcFile &fo,
                               const TelescopeHeader &tel_header) {
    netCDF::NcDim tel_header_dim = fo.addDim("tel_header_n_pts", 1);
    for (const auto &[key, val] : tel_header) {
        netCDF::NcVar tel_header_v =
            fo.addVar(key, netCDF::ncDouble, tel_header_dim);
        tel_header_v.putVar(&val(0));
    }
}
