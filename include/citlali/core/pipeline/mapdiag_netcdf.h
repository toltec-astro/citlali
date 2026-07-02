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

}  // namespace citlali::pipeline
