#pragma once

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <string>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
