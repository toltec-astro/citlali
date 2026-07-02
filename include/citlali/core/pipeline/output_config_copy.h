#pragma once

#include <filesystem>
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
    for (const auto &config_filepath : config_filepaths) {
        copy_config_file_to_reduction_dir(config_filepath, reduction_dir,
                                          logger);
    }
}

}  // namespace citlali::pipeline
