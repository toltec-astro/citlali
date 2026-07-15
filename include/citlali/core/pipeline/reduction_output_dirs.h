#pragma once

#include <citlali/core/pipeline/stage_profile.h>
#include <citlali/core/utils/compressed_log_sink.h>
#include <citlali_config/gitversion.h>
#include <kidscpp_config/gitversion.h>
#include <tula_config/gitversion.h>

#include <filesystem>
#include <exception>
#include <iomanip>
#include <sstream>
#include <string>

namespace citlali::pipeline {

template <class Logger>
void log_reduction_version_stamp(const Logger &logger) {
    logger->info("citlali version: {}", CITLALI_GIT_VERSION);
    logger->info("kids version: {}", KIDSCPP_GIT_VERSION);
    logger->info("tula version: {}", TULA_GIT_VERSION);
}

inline std::string reduction_subdir_name(int redu_dir_num) {
    std::stringstream ss_redu_dir_num;
    ss_redu_dir_num << std::setfill('0') << std::setw(2) << redu_dir_num;
    return "redu" + ss_redu_dir_num.str();
}

inline std::string next_reduction_subdir_path(const std::string &output_dir,
                                              int &redu_dir_num) {
    redu_dir_num = 0;
    std::string redu_dir_name = reduction_subdir_name(redu_dir_num);

    while (std::filesystem::exists(
        std::filesystem::status(output_dir + "/" + redu_dir_name))) {
        ++redu_dir_num;
        redu_dir_name = reduction_subdir_name(redu_dir_num);
    }

    return output_dir + "/" + redu_dir_name;
}

template <class Logger>
void configure_reduction_logging_and_profile(const std::string &redu_dir_name,
                                             StageProfileCollector &stage_profile,
                                             const Logger &logger) {
    try {
        const auto log_path =
            citlali::logging::enable_reduction_gzip_logs(redu_dir_name);
        logger->info("reduction-local compressed log: {}", log_path);
        log_reduction_version_stamp(logger);
    }
    catch (const std::exception &e) {
        logger->warn(
            "failed to enable reduction-local compressed log in {}: {}",
            redu_dir_name, e.what());
    }
    configure_stage_profile_output(stage_profile, redu_dir_name, logger);
}

template <class Logger>
void create_output_directory_or_warn(const std::string &dir_name,
                                     const Logger &logger) {
    if (!std::filesystem::exists(std::filesystem::status(dir_name))) {
        std::filesystem::create_directories(dir_name);
    }
    else {
        logger->warn("directory {} already exists", dir_name);
    }
}

template <class Logger>
void create_coadd_output_dirs(const std::string &coadd_dir_name,
                              bool map_filter_outputs_enabled,
                              const Logger &logger) {
    create_output_directory_or_warn(coadd_dir_name + "raw/", logger);
    if (map_filter_outputs_enabled) {
        create_output_directory_or_warn(coadd_dir_name + "filtered/",
                                        logger);
    }
}

}  // namespace citlali::pipeline
