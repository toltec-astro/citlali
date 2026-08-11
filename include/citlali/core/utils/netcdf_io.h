#pragma once

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cctype>
#include <filesystem>
#include <fcntl.h>
#include <netcdf>
#include <sstream>
#include <string>
#include <system_error>
#include <type_traits>
#include <utility>
#include <unistd.h>
#include <vector>

struct DataIOError : public std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline constexpr const char *netcdf_atomic_staging_marker =
    ".citlali-stage.";

enum class NetcdfAtomicFailureStage {
    none,
    create,
    write,
    sync,
    close,
    publish,
};

inline void inject_netcdf_atomic_failure(
    NetcdfAtomicFailureStage configured, NetcdfAtomicFailureStage current) {
    if (configured == current) {
        throw DataIOError("injected netCDF atomic lifecycle failure");
    }
}

inline bool netcdf_atomic_decimal_component(const std::string &value) {
    return !value.empty() && std::all_of(
        value.begin(), value.end(),
        [](unsigned char c) { return std::isdigit(c) != 0; });
}

inline bool is_netcdf_atomic_staging_path(const std::string &path) {
    const auto marker = path.rfind(netcdf_atomic_staging_marker);
    if (marker == std::string::npos || marker == 0) {
        return false;
    }
    const auto suffix = path.substr(
        marker + std::char_traits<char>::length(
                     netcdf_atomic_staging_marker));
    const auto separator = suffix.find('.');
    return separator != std::string::npos &&
           suffix.find('.', separator + 1) == std::string::npos &&
           netcdf_atomic_decimal_component(suffix.substr(0, separator)) &&
           suffix.substr(0, separator) == std::to_string(::getpid()) &&
           netcdf_atomic_decimal_component(suffix.substr(separator + 1));
}

inline std::string netcdf_atomic_final_path_from_staging(
    const std::string &staging_path) {
    const auto marker = staging_path.rfind(netcdf_atomic_staging_marker);
    if (!is_netcdf_atomic_staging_path(staging_path)) {
        throw DataIOError("not a Citlali netCDF staging path: " +
                          staging_path);
    }
    return staging_path.substr(0, marker);
}

inline std::string reserve_netcdf_atomic_staging_path(
    const std::string &final_path,
    NetcdfAtomicFailureStage failure = NetcdfAtomicFailureStage::none) {
    inject_netcdf_atomic_failure(failure, NetcdfAtomicFailureStage::create);
    static std::atomic<unsigned long long> sequence{0};
    for (int attempt = 0; attempt < 128; ++attempt) {
        std::ostringstream name;
        name << final_path << netcdf_atomic_staging_marker
             << static_cast<unsigned long long>(::getpid()) << "."
             << sequence.fetch_add(1, std::memory_order_relaxed);
        const auto staging = name.str();
        const int descriptor = ::open(
            staging.c_str(), O_CREAT | O_EXCL | O_WRONLY, 0600);
        if (descriptor >= 0) {
            if (::close(descriptor) != 0) {
                std::error_code ec;
                std::filesystem::remove(staging, ec);
                throw DataIOError("failed to close reserved netCDF staging file " +
                                  staging);
            }
            return staging;
        }
        if (errno != EEXIST) {
            throw DataIOError("failed to reserve adjacent netCDF staging file for " +
                              final_path + ": " +
                              std::error_code(errno, std::generic_category()).message());
        }
    }
    throw DataIOError("exhausted unique adjacent netCDF staging names for " +
                      final_path);
}

inline void cleanup_netcdf_atomic_staging(const std::string &staging_path) {
    if (!is_netcdf_atomic_staging_path(staging_path)) {
        return;
    }
    std::error_code ec;
    std::filesystem::remove(staging_path, ec);
}

inline std::string publish_netcdf_atomic_staging(
    const std::string &staging_path) {
    namespace fs = std::filesystem;
    const std::string final_path =
        netcdf_atomic_final_path_from_staging(staging_path);
    std::error_code ec;
    if (!fs::is_regular_file(staging_path, ec) || ec) {
        throw DataIOError("netCDF staging artifact is not a regular file: " +
                          staging_path);
    }
    ec.clear();
    // On the supported local POSIX filesystems this is one atomic replacement.
    // Do not pre-delete the prior final: any failure therefore preserves it.
    fs::rename(staging_path, final_path, ec);
    if (ec) {
        throw DataIOError("failed to atomically publish netCDF staging file " +
                          staging_path + " -> " + final_path + ": " +
                          ec.message());
    }
    return final_path;
}

template <typename Writer>
std::string write_netcdf_staging(const std::string &final_path,
                                 Writer &&writer,
                                 NetcdfAtomicFailureStage failure =
                                     NetcdfAtomicFailureStage::none) {
    const std::string staging_path =
        reserve_netcdf_atomic_staging_path(final_path, failure);
    try {
        netCDF::NcFile fo(staging_path, netCDF::NcFile::replace);
        writer(fo);
        inject_netcdf_atomic_failure(failure,
                                     NetcdfAtomicFailureStage::write);
        inject_netcdf_atomic_failure(failure,
                                     NetcdfAtomicFailureStage::sync);
        fo.sync();
        inject_netcdf_atomic_failure(failure,
                                     NetcdfAtomicFailureStage::close);
        fo.close();
        return staging_path;
    }
    catch (...) {
        cleanup_netcdf_atomic_staging(staging_path);
        throw;
    }
}

template <typename Writer>
void write_netcdf_atomic(
    const std::string &final_path, Writer &&writer,
    NetcdfAtomicFailureStage failure = NetcdfAtomicFailureStage::none) {
    std::string staging_path;
    try {
        staging_path = write_netcdf_staging(
            final_path, std::forward<Writer>(writer), failure);
        inject_netcdf_atomic_failure(failure,
                                     NetcdfAtomicFailureStage::publish);
        (void)publish_netcdf_atomic_staging(staging_path);
    }
    catch (...) {
        cleanup_netcdf_atomic_staging(staging_path);
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
