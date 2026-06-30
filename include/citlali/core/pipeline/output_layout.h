#pragma once

#include <filesystem>
#include <fstream>
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

template <class Logger>
void copy_config_files_to_reduction_dir(
    const std::vector<std::string> &config_filepaths,
    const std::string &reduction_dir, const Logger &logger) {
    namespace fs = std::filesystem;
    for (const auto &config_filepath : config_filepaths) {
        logger->debug("copying config files into redu directory");
        fs::copy(config_filepath,
                 config_copy_destination(reduction_dir, config_filepath),
                 fs::copy_options::overwrite_existing);
    }
}

inline std::string format_obsnum(int obsnum) {
    std::stringstream ss;
    ss << std::setfill('0') << std::setw(6) << obsnum;
    return ss.str();
}

template <class Engine>
void configure_observation_output_layout(Engine &engine, int obsnum) {
    engine.obsnum = format_obsnum(obsnum);
    engine.obsnum_dir_name = engine.redu_dir_name + "/" + engine.obsnum + "/";

    engine.omb.obsnums.clear();
    engine.omb.obsnums.push_back(engine.obsnum);

    if (engine.run_coadd) {
        engine.cmb.obsnums.push_back(engine.obsnum);
    }
}

template <class Engine, class Logger>
void create_observation_output_dirs(const Engine &engine,
                                    const Logger &logger) {
    namespace fs = std::filesystem;

    logger->debug("creating obsnum directory");
    fs::create_directories(engine.obsnum_dir_name);

    logger->debug("creating obsnum raw directory");
    fs::create_directories(engine.obsnum_dir_name + "raw/");

    if (!engine.run_coadd) {
        if (engine.run_map_filter) {
            logger->debug("creating obsnum filtered directory");
            fs::create_directories(engine.obsnum_dir_name + "filtered/");
        }
    }
    if (engine.verbose_mode) {
        logger->debug("creating obsnum logs directory");
        fs::create_directories(engine.obsnum_dir_name + "logs/");
    }
}

template <class Engine, class Logger>
void prepare_observation_output_layout(Engine &engine, int obsnum,
                                       const Logger &logger) {
    configure_observation_output_layout(engine, obsnum);
    create_observation_output_dirs(engine, logger);
}

template <class RawObsKidsMeta, class Logger>
int obsnum_from_rawobs_meta(const RawObsKidsMeta &rawobs_kids_meta,
                            const Logger &logger) {
    logger->debug("getting obsnum");
    return rawobs_kids_meta.back().template get_typed<int>("obsid");
}

template <class Engine, class RawObsKidsMeta, class Logger>
void prepare_observation_output_layout_from_rawobs_meta(
    Engine &engine, const RawObsKidsMeta &rawobs_kids_meta,
    const Logger &logger) {
    const int obsnum = obsnum_from_rawobs_meta(rawobs_kids_meta, logger);
    prepare_observation_output_layout(engine, obsnum, logger);
}

inline std::string gaps_log_filepath(const std::string &obsnum_dir_name) {
    return obsnum_dir_name + "/logs/gaps.log";
}

template <class Engine, class Logger>
void record_timing_gaps_if_needed(const Engine &engine, const Logger &logger) {
    if (engine.gaps.size() > 0) {
        logger->warn("gaps found in obnsum {} data file timing!",
                     engine.obsnum);
        if (engine.verbose_mode) {
            logger->debug("writing gaps.log file");
            std::ofstream f;
            f.open(gaps_log_filepath(engine.obsnum_dir_name));
            f << "Summary of timing gaps\n";
            for (auto const &[key, val] : engine.gaps) {
                logger->debug("{} gaps: {}", key, val);
                f << "-" + key + " gaps: " << val << "\n";
            }
            f.close();
        }
    }
}

}  // namespace citlali::pipeline
