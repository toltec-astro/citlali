#pragma once

// Included by mapdiag_netcdf.h inside namespace citlali::pipeline.

inline void put_netcdf_string_1d(
    netCDF::NcFile &fo, const std::string &name, netCDF::NcDim dim,
    const std::vector<std::string> &values,
    const std::string &comment = "") {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncString, dim);
    if (!comment.empty()) {
        v.putAtt("comment", comment);
    }
    for (std::size_t i=0; i<values.size(); ++i) {
        const std::vector<std::size_t> idx = {i};
        std::string value = values[i];
        v.putVar(idx, value);
    }
}

inline void add_mapdiag_map_label_vars(
    netCDF::NcFile &fo, netCDF::NcDim maps_dim,
    const std::vector<std::string> &array_names,
    const std::vector<std::string> &stokes_names,
    const std::vector<std::string> &map_names) {
    put_netcdf_string_1d(
        fo, "map_array_name", maps_dim, array_names,
        "array label for each map row");
    put_netcdf_string_1d(
        fo, "map_stokes", maps_dim, stokes_names,
        "stokes parameter label for each map row");
    put_netcdf_string_1d(
        fo, "map_name", maps_dim, map_names,
        "grouping-derived map label prefix for each map row");
}

inline void add_mapdiag_observation_label_vars(
    netCDF::NcFile &fo, netCDF::NcDim obsnums_dim,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    std::size_t n_obsnums) {
    const auto obsnum_strings =
        mapdiag_obsnum_labels(obsnums, fallback_obsnum);
    put_netcdf_string_1d(
        fo, "coadd_obsnum", obsnums_dim, obsnum_strings,
        "obsnum ordering for map x obsnum contribution tables");

    const auto dateobs_strings =
        mapdiag_dateobs_labels(date_obs, n_obsnums);
    put_netcdf_string_1d(
        fo, "coadd_dateobs", obsnums_dim, dateobs_strings,
        "DATEOBS ordering matching coadd_obsnum");
}

inline void add_mapdiag_label_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::vector<std::string> &array_names,
    const std::vector<std::string> &stokes_names,
    const std::vector<std::string> &map_names,
    const std::vector<std::string> &obsnums,
    const std::string &fallback_obsnum,
    const std::vector<std::string> &date_obs,
    std::size_t n_obsnums) {
    add_mapdiag_map_label_vars(
        fo, dims.maps, array_names, stokes_names, map_names);
    add_mapdiag_observation_label_vars(
        fo, dims.obsnums, obsnums, fallback_obsnum, date_obs, n_obsnums);
}

inline void add_mapdiag_label_vars(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const MapdiagLabelVars &values) {
    add_mapdiag_label_vars(
        fo, dims, values.array_names, values.stokes_names,
        values.map_names, values.obsnums, values.fallback_obsnum,
        values.date_obs, values.n_obsnums);
}

