#pragma once

// Included by ptcdiag_netcdf.h inside namespace citlali::pipeline.

template <class Calib>
std::vector<int> ptcdiag_apt_int_values(const Calib &calib,
                                        const std::string &key,
                                        int fill_value) {
    std::vector<int> values(static_cast<std::size_t>(calib.n_dets),
                            fill_value);
    const auto it = calib.apt.find(key);
    if (it != calib.apt.end() && it->second.size() == calib.n_dets) {
        for (Eigen::Index i=0; i<calib.n_dets; ++i) {
            values[static_cast<std::size_t>(i)] =
                static_cast<int>(std::lround(it->second(i)));
        }
    }
    return values;
}

template <class Calib>
std::vector<int> diagnostic_network_ids(const Calib &calib,
                                        int fill_value) {
    std::vector<int> ids(static_cast<std::size_t>(calib.n_nws),
                         fill_value);
    for (Eigen::Index i=0; i<calib.n_nws; ++i) {
        ids[static_cast<std::size_t>(i)] = static_cast<int>(calib.nws(i));
    }
    return ids;
}

inline void add_ptcdiag_det_meta_int(netCDF::NcFile &fo,
                                     const std::string &name,
                                     const std::string &comment,
                                     netCDF::NcDim n_dets_dim,
                                     const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, n_dets_dim);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

template <class AddMetaInt, class AptIntValues>
void add_ptcdiag_det_meta_vars(const AddMetaInt &add_meta_int,
                               const AptIntValues &apt_int_values) {
    add_meta_int("ptc_diag_uid", "detector UID along n_dets",
                 apt_int_values("uid"));
    add_meta_int("ptc_diag_array", "array index along n_dets",
                 apt_int_values("array"));
    add_meta_int("ptc_diag_network", "network index along n_dets",
                 apt_int_values("nw"));
    add_meta_int("ptc_diag_apt_flag", "APT detector flag along n_dets",
                 apt_int_values("flag"));
}

template <class Calib>
void add_ptcdiag_detector_metadata_vars(netCDF::NcFile &fo,
                                        const Calib &calib,
                                        netCDF::NcDim n_dets_dim,
                                        int fill_int) {
    auto add_det_meta_int = [&](const std::string &name,
                                const std::string &comment,
                                const std::vector<int> &values) {
        add_ptcdiag_det_meta_int(
            fo, name, comment, n_dets_dim, values);
    };
    auto apt_int_values = [&](const std::string &key) {
        return ptcdiag_apt_int_values(calib, key, fill_int);
    };
    add_ptcdiag_det_meta_vars(add_det_meta_int, apt_int_values);
}

inline void add_ptcdiag_det_double(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    double fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    auto chunks = det_chunks;
    v.setChunking(netCDF::NcVar::nc_CHUNKED, chunks);
    std::vector<double> init(n_values, fill_value);
    v.putVar(init.data());
}

inline void add_ptcdiag_det_int(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_values,
    int fill_value) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, det_dims);
    v.putAtt("units", "N/A");
    v.putAtt("comment", comment);
    auto chunks = det_chunks;
    v.setChunking(netCDF::NcVar::nc_CHUNKED, chunks);
    std::vector<int> init(n_values, fill_value);
    v.putVar(init.data());
}

template <class AddDouble>
void add_ptcdiag_detector_core_diag(const AddDouble &add_double) {
    add_double("ptc_detector_weight",
               "final detector map weight used by PTC for this scan");
    add_double("ptc_detector_rms",
               "per-detector RMS of the PTC timestream written for this scan");
    add_double("ptc_detector_stddev",
               "per-detector standard deviation of the PTC timestream written for this scan");
    add_double("ptc_detector_median",
               "per-detector median of the PTC timestream written for this scan");
    add_double("ptc_detector_flagged_fraction",
               "fraction of detector samples flagged in the PTC timestream for this scan");
}

template <class AddInt, class AddDouble>
void add_ptcdiag_detector_invvar_window_diag(const AddInt &add_int,
                                             const AddDouble &add_double) {
    add_double("ptc_invvar_window_valid_fraction",
               "fraction of remove_bad_dets diagnostic windows with enough unflagged samples to estimate inverse variance in the PTC timestream");
    add_double("ptc_invvar_window_median",
               "median per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_double("ptc_invvar_window_q10",
               "10th percentile of per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_double("ptc_invvar_window_q90",
               "90th percentile of per-window inverse variance used for PTC remove_bad_dets diagnostics");
    add_double("ptc_invvar_window_flagged_frac_median",
               "median flagged fraction across remove_bad_dets diagnostic windows in the PTC timestream");
    add_double("ptc_invvar_window_flagged_frac_max",
               "maximum flagged fraction across remove_bad_dets diagnostic windows in the PTC timestream");
    add_double("ptc_invvar_window_heavy_flagged_fraction",
               "fraction of remove_bad_dets diagnostic windows in the PTC timestream with at least 50 percent flagged samples");
    add_int("ptc_invvar_window_n_total",
            "total number of fixed windows evaluated for PTC remove_bad_dets diagnostics");
    add_int("ptc_invvar_window_n_valid",
            "number of fixed windows with a finite inverse-variance estimate for PTC remove_bad_dets diagnostics");
}

inline void add_ptcdiag_standard_detector_diag(
    netCDF::NcFile &fo, const std::vector<netCDF::NcDim> &det_dims,
    const std::vector<std::size_t> &det_chunks, std::size_t n_det_values,
    int fill_int, double fill_double) {
    auto add_det_double = [&](const std::string &name,
                              const std::string &comment) {
        add_ptcdiag_det_double(
            fo, name, comment, det_dims, det_chunks, n_det_values,
            fill_double);
    };
    auto add_det_int = [&](const std::string &name,
                           const std::string &comment) {
        add_ptcdiag_det_int(
            fo, name, comment, det_dims, det_chunks, n_det_values,
            fill_int);
    };
    add_ptcdiag_detector_core_diag(add_det_double);
    add_ptcdiag_detector_invvar_window_diag(add_det_int, add_det_double);
}

