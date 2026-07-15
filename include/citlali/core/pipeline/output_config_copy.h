#pragma once

#include <cstddef>
#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline std::string config_copy_filename(const std::string &config_filepath) {
    const auto last_slash_pos = config_filepath.find_last_of("/");
    if (last_slash_pos == std::string::npos) {
        return config_filepath;
    }
    return config_filepath.substr(last_slash_pos + 1);
}

inline std::string config_copy_destination(const std::string &reduction_dir,
                                           const std::string &config_filepath) {
    return reduction_dir + "/" + config_copy_filename(config_filepath);
}

inline std::string config_copy_filename(
    const std::vector<std::string> &config_filepaths, std::size_t index) {
    const auto basename = config_copy_filename(config_filepaths.at(index));
    std::size_t matching_basenames = 0;
    for (const auto &filepath : config_filepaths) {
        matching_basenames += config_copy_filename(filepath) == basename;
    }
    if (matching_basenames == 1) {
        return basename;
    }
    std::ostringstream stream;
    stream << "source_" << std::setw(3) << std::setfill('0') << index << "_"
           << basename;
    return stream.str();
}

inline std::string config_copy_destination(
    const std::string &reduction_dir,
    const std::vector<std::string> &config_filepaths, std::size_t index) {
    return reduction_dir + "/" +
           config_copy_filename(config_filepaths, index);
}

template <class Logger>
void copy_config_file_to_reduction_dir(const std::string &config_filepath,
                                       const std::string &reduction_dir,
                                       const Logger &logger) {
    namespace fs = std::filesystem;

    logger->debug("copying config files into redu directory");
    fs::copy(config_filepath,
             config_copy_destination(reduction_dir, config_filepath),
             fs::copy_options::overwrite_existing);
}

template <class Logger>
void copy_config_files_to_reduction_dir(
    const std::vector<std::string> &config_filepaths,
    const std::string &reduction_dir, const Logger &logger) {
    namespace fs = std::filesystem;
    for (std::size_t index = 0; index < config_filepaths.size(); ++index) {
        logger->debug("copying config files into redu directory");
        fs::copy(config_filepaths[index],
                 config_copy_destination(reduction_dir, config_filepaths,
                                         index),
                 fs::copy_options::overwrite_existing);
    }
}

}  // namespace citlali::pipeline
