#pragma once

// Included by tod_output_data_vars.h inside namespace citlali::pipeline.

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

