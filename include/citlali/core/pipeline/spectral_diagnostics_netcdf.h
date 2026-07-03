#pragma once

#include <string>
#include <vector>

#include <netcdf>

namespace citlali::pipeline {

template <class Data>
void add_double_1d_var(netCDF::NcFile &fo, const std::string &name,
                       netCDF::NcDim dim, const Data &data) {
    netCDF::NcVar var = fo.addVar(name, netCDF::ncDouble, dim);
    var.putVar(data.data());
}

template <class Data>
void add_double_2d_var(netCDF::NcFile &fo, const std::string &name,
                       const std::vector<netCDF::NcDim> &dims,
                       const Data &data) {
    netCDF::NcVar var = fo.addVar(name, netCDF::ncDouble, dims);
    var.putVar(data.data());
}

}  // namespace citlali::pipeline
