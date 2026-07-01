#pragma once

#include <citlali/core/pipeline/array_properties_table.h>
#include <citlali/core/pipeline/flxscale_correction.h>
#include <citlali/core/pipeline/kids_metadata.h>

#include <cmath>
#include <string>
#include <vector>

namespace citlali::pipeline {

template <bool IsBeammap, class TodProc, class RawObs, class Logger>
void configure_observation_calibration(TodProc &todproc, const RawObs &rawobs,
                                       const Logger &logger) {
    auto &engine = todproc.engine();

    logger->debug("getting astrometry config");
    engine.get_astrometry_config(rawobs.astrometry_calib_info().config());

    if constexpr (IsBeammap) {
        engine.get_photometry_config(rawobs.photometry_calib_info().config());
        if (engine.map_grouping == "detector" ||
            engine.map_grouping == "auto") {
            logger->info("making apt file from raw nc files");
            todproc.get_apt_from_files(rawobs);
            return;
        }
    }

    load_array_properties_table(engine, rawobs, logger);
}

template <bool IsBeammap, class TodProc, class RawObs, class RawObsKidsMeta,
          class Logger>
bool configure_reduction_observation_calibration_if_needed(
    TodProc &todproc, const RawObs &rawobs,
    const RawObsKidsMeta &rawobs_kids_meta, bool should_configure,
    const Logger &logger) {
    if (!should_configure) {
        return true;
    }

    auto &engine = todproc.engine();
    configure_observation_calibration<IsBeammap>(todproc, rawobs, logger);
    if (!apply_flxscale_correction(engine, rawobs, logger)) {
        return false;
    }

    update_sample_rate_from_rawobs_meta(engine, rawobs_kids_meta, logger);
    return true;
}

template <class Engine, class Logger>
void calculate_flux_calibration(Engine &engine, const Logger &logger) {
    logger->info("calculating flux calibration");
    engine.calib.calc_flux_calibration(engine.omb.sig_unit,
                                       engine.omb.pixel_size_rad);
}

template <class Engine, class RawObs, class Logger>
void load_hwpr_data_if_requested(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    if (engine.rtcproc.run_polarization) {
        std::string hwpr_filepath;
        if (rawobs.hwpdata().has_value() && engine.calib.ignore_hwpr != "true") {
            hwpr_filepath = rawobs.hwpdata()->filepath();
            if (hwpr_filepath != "null") {
                logger->info("getting hwpr file {}", hwpr_filepath);
                engine.calib.get_hwpr(hwpr_filepath, engine.telescope.sim_obs);
            }
            else {
                engine.calib.run_hwpr = false;
            }
        }
        else {
            engine.calib.run_hwpr = false;
        }
        if (!engine.calib.run_hwpr) {
            logger->info("ignoring hwpr");
        }
    }
}

}  // namespace citlali::pipeline
