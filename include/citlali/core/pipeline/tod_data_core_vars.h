#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

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

