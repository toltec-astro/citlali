#pragma once

// Included by rtcdiag_impulsive_capture.h inside namespace citlali::pipeline.

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

