#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/timestream_native_pointing.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::interp_pointing() {
    const auto tel_time_it = engine().telescope.tel_data.find("TelTime");
    if (tel_time_it == engine().telescope.tel_data.end() ||
        tel_time_it->second.size() <= 0) {
        logger->error("cannot interpolate pointing offsets: telescope TelTime is empty");
        throw citlali::error::runtime(
            "cannot interpolate pointing offsets: telescope TelTime is empty");
    }

    try {
        const auto model =
            citlali::pipeline::make_native_pointing_offset_model(
                engine().pointing_offsets, tel_time_it->second);
        auto evaluated = model.evaluate_at(tel_time_it->second);
        for (const auto *axis : {
                 citlali::config::pointing_axis_alt(),
                 citlali::config::pointing_axis_az()}) {
            engine().pointing_offsets.arcsec[axis] =
                std::move(evaluated.at(axis));
        }
    }
    catch (const citlali::error::Error &) {
        throw;
    }
    catch (const std::exception &error) {
        logger->error("cannot interpolate pointing offsets: {}", error.what());
        throw citlali::error::invalid_config(error.what());
    }
}
