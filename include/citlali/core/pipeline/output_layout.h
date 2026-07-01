#pragma once

#include <citlali/core/pipeline/output_config_copy.h>

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <vector>

namespace citlali::pipeline {

template <class TodProc, class ConfigFilepaths, class Logger>
void prepare_iteration_output_layout_if_needed(
    TodProc &todproc, const ConfigFilepaths &config_filepaths,
    const Logger &logger) {
    auto &engine = todproc.engine();

    if (engine.ptcproc.save_all_iters || engine.fruit_iter == 0) {
        todproc.create_output_dir();
        copy_config_files_to_reduction_dir(
            config_filepaths, engine.redu_dir_name, logger);
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
