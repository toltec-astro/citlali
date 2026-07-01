#pragma once

#include <citlali/core/pipeline/obsnum_format.h>
#include <citlali/core/pipeline/observation_output_config.h>
#include <citlali/core/pipeline/observation_output_dirs.h>


namespace citlali::pipeline {

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
