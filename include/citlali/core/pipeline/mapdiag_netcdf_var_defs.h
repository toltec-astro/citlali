#pragma once

// Included by mapdiag_netcdf.h inside namespace citlali::pipeline.

inline void add_mapdiag_double_1d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim dim,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dim);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_int_1d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, netCDF::NcDim dim,
    const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dim);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_double_2d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &dims,
    const std::vector<double> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncDouble, dims);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_int_2d(
    netCDF::NcFile &fo, const std::string &name,
    const std::string &comment, const std::vector<netCDF::NcDim> &dims,
    const std::vector<int> &values) {
    netCDF::NcVar v = fo.addVar(name, netCDF::ncInt, dims);
    v.putAtt("comment", comment);
    v.putVar(values.data());
}

inline void add_mapdiag_map_double_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<double> &values) {
    add_mapdiag_double_1d(fo, name, comment, dims.maps, values);
}

inline void add_mapdiag_map_int_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<int> &values) {
    add_mapdiag_int_1d(fo, name, comment, dims.maps, values);
}

inline void add_mapdiag_obs_double_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<double> &values) {
    add_mapdiag_double_2d(fo, name, comment, dims.map_obs, values);
}

inline void add_mapdiag_obs_int_var(
    netCDF::NcFile &fo, const MapdiagNetcdfDims &dims,
    const std::string &name, const std::string &comment,
    const std::vector<int> &values) {
    add_mapdiag_int_2d(fo, name, comment, dims.map_obs, values);
}

