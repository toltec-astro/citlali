#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

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

