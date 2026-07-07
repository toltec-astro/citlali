#pragma once

// Beammap split-FITS output helpers.

#include <exception>
#include <filesystem>
#include <cmath>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Core>

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

template <class Flags>
int detector_flag(const Flags &flags, Eigen::Index map_index) {
    return static_cast<int>(std::lround(flags(map_index)));
}

template <class Flags>
Eigen::Index count_maps_with_flag(const Flags &flags,
                                  Eigen::Index n_maps,
                                  int flag_value) {
    Eigen::Index n_flag_maps = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const int det_flag = detector_flag(flags, i);
        if (det_flag == flag_value) {
            ++n_flag_maps;
        }
    }
    return n_flag_maps;
}

template <class Flags>
Eigen::Index count_maps_with_any_flag(const Flags &flags,
                                      Eigen::Index n_maps,
                                      const std::vector<int> &flag_values) {
    const std::set<int> split_values(flag_values.begin(), flag_values.end());
    Eigen::Index n_selected_maps = 0;
    for (Eigen::Index i = 0; i < n_maps; ++i) {
        const int det_flag = detector_flag(flags, i);
        if (split_values.count(det_flag) > 0) {
            ++n_selected_maps;
        }
    }
    return n_selected_maps;
}

template <class FitsIo>
void add_split_primary_header(FitsIo &fits_io, Eigen::Index index,
                              int flag_value) {
    fits_io.at(index).pfits->pHDU().addKey(
        "BEAMMAP.SPLIT_BY", "flag", "Beammap detector split criterion");
    fits_io.at(index).pfits->pHDU().addKey(
        "BEAMMAP.SPLIT_VALUE", flag_value,
        "Beammap detector flag value in this file");
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

template <class Logger, class FitsIoVec>
void log_output_filepaths(const Logger &logger,
                          const FitsIoVec &fits_io) {
    logger->info("maps have been written to:");
    for (Eigen::Index i = 0; i < fits_io.size(); ++i) {
        logger->info("{}.fits", fits_io.at(i).filepath);
    }
}

template <class Logger, class FitsIoVec>
void log_split_output_filepaths(const Logger &logger,
                                const FitsIoVec &fits_io,
                                int flag_value) {
    logger->info("beammap split maps (flag={}) have been written to:",
                 flag_value);
    for (Eigen::Index i = 0; i < fits_io.size(); ++i) {
        logger->info("{}.fits", fits_io.at(i).filepath);
    }
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
