#pragma once

#include <cstddef>
#include <string>
#include <vector>

#include <netcdf>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
