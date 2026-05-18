#pragma once

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <netcdf>
#include <string>
#include <system_error>
#include <type_traits>
#include <vector>

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
    netCDF::NcVar var = fo.getVar(name);

    auto put_value = [&](netCDF::NcVar &target) {
        // if int
        if constexpr (std::is_same_v<T, int>) {
            target.putVar(&data);
        }
        // if bool (netcdf has no bool type)
        if constexpr (std::is_same_v<T, bool>) {
            int value = data ? 1 : 0;
            target.putVar(&value);
        }
        // if other integral scalar types (for example Eigen::Index)
        if constexpr (std::is_integral_v<T> &&
                      !std::is_same_v<T, int> &&
                      !std::is_same_v<T, bool>) {
            const long long value = static_cast<long long>(data);
            target.putVar(&value);
        }
        // if double
        if constexpr (std::is_same_v<T, double>) {
            target.putVar(&data);
        }
        // if string
        if constexpr (std::is_same_v<T, std::string>) {
            const std::vector<size_t> index = {0};
            target.putVar(index, data);
        }
    };

    if (!var.isNull()) {
        put_value(var);
        return;
    }

    auto dim_name = name;

    // make dimension name lower case for cleanliness
    std::transform(dim_name.begin(), dim_name.end(), dim_name.begin(),
                   [](unsigned char c){ return std::tolower(c); });

    dim_name += "_dim";
    dim = fo.getDim(dim_name);
    if (dim.isNull()) {
        dim = fo.addDim(dim_name, 1);
    }

    // if int
    if constexpr (std::is_same_v<T, int>) {
        var = fo.addVar(name, netCDF::ncInt, dim);
        put_value(var);
    }
    // if bool (netcdf has no bool type)
    if constexpr (std::is_same_v<T, bool>) {
        var = fo.addVar(name, netCDF::ncInt, dim);
        put_value(var);
    }
    // if other integral scalar types (for example Eigen::Index)
    if constexpr (std::is_integral_v<T> &&
                  !std::is_same_v<T, int> &&
                  !std::is_same_v<T, bool>) {
        var = fo.addVar(name, netCDF::ncInt64, dim);
        put_value(var);
    }
    // if double
    if constexpr (std::is_same_v<T, double>) {
        var = fo.addVar(name, netCDF::ncDouble, dim);
        put_value(var);
    }
    // if string
    if constexpr (std::is_same_v<T, std::string>) {
        var = fo.addVar(name, netCDF::ncString, dim);
        put_value(var);
    }
}
