#pragma once

// Included by rtcdiag_network_outputs.h inside namespace citlali::pipeline.

inline void add_rtcdiag_network_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_rtcdiag_network_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &nw_dims,
    const std::vector<std::size_t> &nw_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, nw_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    set_netcdf_chunking_and_compression(v, nw_chunks, 1);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class AddInt>
void add_rtcdiag_network_detector_count_diag(const AddInt &add_int) {
    add_int("rtc_network_n_det_input",
            "input detector count in each RTC network block");
    add_int("rtc_network_n_det_used",
            "detectors passing the step-mask valid-sample threshold and finite robust scale");
    add_int("rtc_network_impulsive_n_det_used",
            "detectors passing the impulsive-coincidence valid-sample threshold and finite robust scale");
}

