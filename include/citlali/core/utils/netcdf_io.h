#pragma once

#include <filesystem>
#include <netcdf>
#include <type_traits>
#include <system_error>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

template <typename Writer>
void write_netcdf_atomic(const std::string &final_path, Writer &&writer) {
    namespace fs = std::filesystem;
    const fs::path final_file(final_path);
    const fs::path temp_file(final_path + ".tmp");
    std::error_code ec;
    fs::remove(temp_file, ec);

    try {
        netCDF::NcFile fo(temp_file.string(), netCDF::NcFile::replace);
        writer(fo);
        fo.sync();
        fo.close();

        ec.clear();
        fs::remove(final_file, ec);
        ec.clear();
        fs::rename(temp_file, final_file, ec);
        if (ec) {
            throw DataIOError(
                "failed to atomically rename netCDF temp file " +
                temp_file.string() + " -> " + final_file.string() + ": " +
                ec.message());
        }
    } catch (...) {
        ec.clear();
        fs::remove(temp_file, ec);
        throw;
    }
}

inline void set_netcdf_chunking_and_compression(
    netCDF::NcVar &var, const std::vector<std::size_t> &chunk_sizes,
    int deflate_level = 1) {
    if (var.isNull() || chunk_sizes.empty()) {
        return;
    }
    auto chunks = chunk_sizes;
    var.setChunking(netCDF::NcVar::nc_CHUNKED, chunks);
    var.setCompression(true, true, deflate_level);
}

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
    // if other integral scalar types (for example Eigen::Index)
    if constexpr (std::is_integral_v<T> &&
                  !std::is_same_v<T, int> &&
                  !std::is_same_v<T, bool>) {
        const long long value = static_cast<long long>(data);
        dim = fo.addDim(dim_name.c_str(),1);
        var = fo.addVar(name, netCDF::ncInt64, dim);
        var.putVar(&value);
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
