#pragma once

#include <netcdf>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

// write scalars to netcdf file
template<typename T>
void add_netcdf_var(netCDF::NcFile &fo, std::string name, T data) {
    // create netcdf dimension
    netCDF::NcDim dim;
    // create netcdf variable
    netCDF::NcVar var;

    auto dim_name = name;

    // make dimension name lower case for cleanliness
    std::transform(dim_name.begin(), dim_name.end(), dim_name.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    dim_name += "_dim";

    // if int
    if constexpr (std::is_same_v<T, int>) {
        dim = fo.addDim(dim_name,1);
        var = fo.addVar(name, netCDF::ncInt, dim);
        var.putVar(&data);
    }
    // if bool (netcdf has no bool type)
    if constexpr (std::is_same_v<T, bool>) {
        dim = fo.addDim(dim_name.c_str(),1);
        var = fo.addVar(name, netCDF::ncInt, dim);
        var.putVar(&data);
    }
    // if double
    if constexpr (std::is_same_v<T, double>) {
        dim = fo.addDim(dim_name.c_str(),1);
        var = fo.addVar(name, netCDF::ncDouble, dim);
        var.putVar(&data);
    }
    // if string
    if constexpr (std::is_same_v<T, std::string>) {
        dim = fo.addDim(dim_name.c_str(),1);
        var = fo.addVar(name, netCDF::ncString, dim);
        const std::vector<size_t> index = {0};
        var.putVar(index,data);
    }
}
