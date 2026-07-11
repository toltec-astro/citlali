#pragma once

// Included by rtcdiag_layout_config.h inside namespace citlali::pipeline.

inline double rtcdiag_fill_double() {
    return std::numeric_limits<double>::quiet_NaN();
}

constexpr int rtcdiag_fill_int() {
    return -2147483647;
}

inline double rtc_tod_stream_sample_rate(bool downsample_enabled, double fsmp,
                                  double downsampled_fsmp) {
    return downsample_enabled ? downsampled_fsmp : fsmp;
}

struct RtcDiagDims {
    netCDF::NcDim n_scans;
    netCDF::NcDim n_dets;
    netCDF::NcDim n_arrays;
    netCDF::NcDim n_nws;
    std::vector<netCDF::NcDim> scan_array;
    std::vector<netCDF::NcDim> det;
    std::vector<netCDF::NcDim> nw;
    std::vector<std::size_t> scan_chunks;
    std::vector<std::size_t> scan_array_chunks;
    std::vector<std::size_t> det_chunks;
    std::vector<std::size_t> nw_chunks;
    std::size_t n_scan_values;
    std::size_t n_array_values;
    std::size_t n_scan_array_values;
    std::size_t n_det_values;
    std::size_t n_nw_values;
};

inline RtcDiagDims add_rtcdiag_dims(netCDF::NcFile &fo,
                                    Eigen::Index n_scans,
                                    Eigen::Index n_dets,
                                    Eigen::Index n_arrays,
                                    Eigen::Index n_nws) {
    netCDF::NcDim n_scans_dim = fo.addDim("n_scans", n_scans);
    netCDF::NcDim n_dets_dim = fo.addDim("n_dets", n_dets);
    netCDF::NcDim n_arrays_dim = fo.addDim("n_arrays", n_arrays);
    netCDF::NcDim n_nws_dim = fo.addDim("n_nws_rtcdiag", n_nws);

    const auto n_scan_values = static_cast<std::size_t>(n_scans);
    const auto n_array_values = static_cast<std::size_t>(n_arrays);
    const auto n_det_values = static_cast<std::size_t>(n_dets);
    const auto n_nw_values = static_cast<std::size_t>(n_nws);
    const std::vector<netCDF::NcDim> scan_array_dims = {
        n_scans_dim, n_arrays_dim};
    const std::vector<netCDF::NcDim> det_dims = {
        n_scans_dim, n_dets_dim};
    const std::vector<netCDF::NcDim> nw_dims = {
        n_scans_dim, n_nws_dim};
    const std::vector<std::size_t> scan_chunks = {
        static_cast<std::size_t>(std::max<Eigen::Index>(n_scans, 1))};
    const std::vector<std::size_t> scan_array_chunks = {
        1, static_cast<std::size_t>(std::max<Eigen::Index>(n_arrays, 1))};
    const std::vector<std::size_t> det_chunks = {
        1, n_det_values};
    const std::vector<std::size_t> nw_chunks = {
        1, n_nw_values};

    return {
        n_scans_dim,
        n_dets_dim,
        n_arrays_dim,
        n_nws_dim,
        scan_array_dims,
        det_dims,
        nw_dims,
        scan_chunks,
        scan_array_chunks,
        det_chunks,
        nw_chunks,
        n_scan_values,
        n_array_values,
        n_scan_values * n_array_values,
        n_scan_values * n_det_values,
        n_scan_values * n_nw_values};
}

template <class Calib>
std::vector<int> diagnostic_array_ids(const Calib &calib, int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_arrays),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_arrays; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.arrays(i));
    }
    return ids;
}

template <class Calib>
void add_rtcdiag_array_ids(netCDF::NcFile &fo, const Calib &calib,
                           netCDF::NcDim n_arrays_dim, int fill_value) {
    netCDF::NcVar array_ids_v =
        fo.addVar("rtc_diag_array_ids", netCDF::ncInt, n_arrays_dim);
    array_ids_v.putAtt("units", "N/A");
    array_ids_v.putAtt("comment",
                       "array IDs corresponding to n_arrays axis");
    const auto array_ids = diagnostic_array_ids(calib, fill_value);
    array_ids_v.putVar(array_ids.data());
}

template <class Calib>
void add_rtcdiag_network_ids(netCDF::NcFile &fo, const Calib &calib,
                             netCDF::NcDim n_nws_rtcdiag_dim,
                             int fill_value) {
    netCDF::NcVar nw_ids_v =
        fo.addVar("rtc_diag_network_ids", netCDF::ncInt,
                  n_nws_rtcdiag_dim);
    nw_ids_v.putAtt("units", "N/A");
    nw_ids_v.putAtt("comment",
                    "network IDs corresponding to n_nws_rtcdiag axis");
    std::vector<int> nw_ids(static_cast<std::size_t>(calib.n_nws),
                            fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        nw_ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    nw_ids_v.putVar(nw_ids.data());
}

template <class Calib>
void add_rtcdiag_apt_double_vars(netCDF::NcFile &fo, Calib &calib,
                                 netCDF::NcDim n_dets_dim) {
    for (auto const &x : calib.apt) {
        netCDF::NcVar apt_v =
            fo.addVar("apt_" + x.first, netCDF::ncDouble, n_dets_dim);
        apt_v.putAtt("units", calib.apt_header_units[x.first]);
        apt_v.putVar(x.second.data());
    }
}
