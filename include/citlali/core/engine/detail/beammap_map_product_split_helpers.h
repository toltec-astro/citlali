#pragma once

// Beammap split-FITS output helpers.

#include <exception>
#include <filesystem>
#include <string>
#include <vector>

namespace beammap_map_product_split_helpers {

inline std::string split_suffix(int flag_value) {
    std::string suffix = "_flag" + std::to_string(flag_value);
    if (flag_value == 0) {
        suffix += "_good";
    }
    else if (flag_value == 1) {
        suffix += "_bad";
    }
    return suffix;
}

template <class FitsIoVec>
std::vector<std::string> filepaths(const FitsIoVec &fits_io) {
    std::vector<std::string> paths;
    paths.reserve(fits_io.size());
    for (const auto &fio : fits_io) {
        paths.push_back(fio.filepath);
    }
    return paths;
}

template <class Logger>
void remove_fits_files(const std::vector<std::string> &base_filepaths,
                       const char *label,
                       const Logger &logger) {
    namespace fs = std::filesystem;
    for (const auto &path : base_filepaths) {
        const auto fits_path = path + ".fits";
        try {
            if (fs::exists(fits_path)) {
                fs::remove(fits_path);
            }
        }
        catch (const std::exception &e) {
            logger->warn("unable to remove unsplit beammap {} file {}: {}",
                         label, fits_path, e.what());
        }
    }
}

template <class SplitIo>
std::vector<SplitIo> make_split_io(
    const std::vector<std::string> &base_filepaths,
    const std::string &suffix) {
    std::vector<SplitIo> split_io;
    split_io.reserve(base_filepaths.size());
    for (const auto &path : base_filepaths) {
        split_io.emplace_back(path + suffix);
    }
    return split_io;
}

} // namespace beammap_map_product_split_helpers
