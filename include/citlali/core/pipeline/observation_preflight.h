#pragma once

#include <citlali/core/pipeline/kids_metadata.h>
#include <citlali/core/pipeline/observation_calibration.h>
#include <citlali/core/pipeline/rawobs_data_items.h>
#include <citlali/core/pipeline/reduction_config.h>

#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline double degrees_to_radians(double degrees) {
    constexpr double deg_to_rad = 0.017453292519943295769;
    return degrees * deg_to_rad;
}

template <class Engine, class RawObs>
void reset_simulated_observation_indices(Engine &engine,
                                         const RawObs &rawobs) {
    engine.start_indices.clear();
    engine.end_indices.clear();

    for (const auto &data_item : rawobs.kidsdata()) {
        (void)data_item;
        engine.start_indices.push_back(0);
        engine.start_indices.push_back(0);
    }

    if (engine.calib.run_hwpr) {
        engine.hwpr_start_indices = 0;
        engine.hwpr_end_indices = 0;
    }
}

template <class TodProc, class RawObs, class Logger>
void check_observation_inputs(TodProc &todproc, const RawObs &rawobs,
                              const Logger &logger) {
    logger->debug("checking inputs");
    todproc.check_inputs(rawobs);
}

template <class Engine, class Logger>
bool configure_effective_sample_rate(Engine &engine, const Logger &logger) {
    if (engine.rtcproc.run_downsample) {
        if (engine.rtcproc.downsampler.factor <= 0) {
            if (engine.rtcproc.downsampler.downsampled_freq_Hz <= 0) {
                logger->error(
                    "downsampled freq ({} Hz) must be > 0 when downsample "
                    "factor <= 0",
                    engine.rtcproc.downsampler.downsampled_freq_Hz);
                return false;
            }
            if (engine.rtcproc.downsampler.downsampled_freq_Hz >
                engine.telescope.fsmp) {
                logger->error(
                    "downsampled freq ({} Hz) must be less than sample rate "
                    "({} Hz)",
                    engine.rtcproc.downsampler.downsampled_freq_Hz,
                    engine.telescope.fsmp);
                return false;
            }
            engine.rtcproc.downsampler.factor = std::floor(
                engine.telescope.fsmp /
                engine.rtcproc.downsampler.downsampled_freq_Hz);
        }
        if (engine.rtcproc.downsampler.factor <= 0) {
            logger->error("downsample factor ({}) must be > 0",
                          engine.rtcproc.downsampler.factor);
            return false;
        }

        const double downsample_nyquist_Hz =
            engine.telescope.fsmp /
            (2.0 * engine.rtcproc.downsampler.factor);
        if (engine.rtcproc.filter.freq_high_Hz > downsample_nyquist_Hz) {
            logger->error(
                "invalid anti-alias setup: filter freq_high_Hz ({} Hz) "
                "exceeds downsample Nyquist ({} Hz)",
                engine.rtcproc.filter.freq_high_Hz,
                downsample_nyquist_Hz);
            return false;
        }
        engine.telescope.d_fsmp =
            engine.telescope.fsmp / engine.rtcproc.downsampler.factor;
    }
    else {
        engine.telescope.d_fsmp = engine.telescope.fsmp;
    }
    return true;
}

template <class TodProc, class RawObs, class Logger>
void load_raw_detector_diagnostics(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    logger->debug("getting tone frequencies");
    todproc.get_tone_freqs_from_files(rawobs);

    if (!todproc.engine().telescope.sim_obs) {
        logger->debug("getting adc snap data");
        todproc.get_adc_snap_from_files(rawobs);
    }
}

template <class Engine, class Logger>
void overwrite_map_center_if_configured(Engine &engine, const Logger &logger) {
    if (engine.omb.crval_config[0] != 0 && engine.omb.crval_config[1] != 0) {
        logger->info("overwriting map center to ({}, {})",
                     engine.omb.crval_config[0], engine.omb.crval_config[1]);
        const double map_center_ra_rad =
            degrees_to_radians(engine.omb.crval_config[0]);
        const double map_center_dec_rad =
            degrees_to_radians(engine.omb.crval_config[1]);
        engine.telescope.tel_header["Header.Source.Ra"].setConstant(
            map_center_ra_rad);
        engine.telescope.tel_header["Header.Source.Dec"].setConstant(
            map_center_dec_rad);
    }
}

template <class TodProc, class RawObs, class Logger>
void load_and_align_telescope_data(TodProc &todproc, const RawObs &rawobs,
                                   const Logger &logger) {
    auto &engine = todproc.engine();

    auto tel_path = rawobs.teldata().filepath();
    logger->info("getting telescope file {}", tel_path);
    engine.telescope.get_tel_data(tel_path);

    overwrite_map_center_if_configured(engine, logger);

    if (!engine.telescope.sim_obs) {
        logger->info("aligning timestreams");
        if (engine.interp_over_gaps) {
            todproc.align_timestreams_gaps(rawobs);
        }
        else {
            todproc.align_timestreams(rawobs);
        }
    }
    else {
        reset_simulated_observation_indices(engine, rawobs);
    }
}

template <class TodProc, class Logger>
void calculate_telescope_pointing(TodProc &todproc, const Logger &logger) {
    auto &engine = todproc.engine();

    logger->info("calculating tangent plane pointing");
    engine.telescope.calc_tan_pointing();

    logger->info("calculating pointing offsets");
    todproc.interp_pointing();
}

template <class TodProc, class RawObs, class Logger>
void load_and_point_telescope_data_if_needed(TodProc &todproc,
                                             const RawObs &rawobs,
                                             bool should_load,
                                             const Logger &logger) {
    if (!should_load) {
        return;
    }

    load_and_align_telescope_data(todproc, rawobs, logger);
    calculate_telescope_pointing(todproc, logger);
}

template <class Engine, class Logger>
void calculate_scan_indices(Engine &engine, const Logger &logger) {
    logger->info("calculating scan indices");
    engine.telescope.calc_scan_indices();
}

template <class Engine, class Logger>
void calculate_scan_indices_if_needed(Engine &engine, bool should_calculate,
                                      const Logger &logger) {
    if (!should_calculate) {
        return;
    }

    calculate_scan_indices(engine, logger);
}

template <class Engine>
void update_observation_exposure_time(Engine &engine) {
    auto t0 = engine.telescope.tel_data["TelTime"](0);
    auto tn = engine.telescope.tel_data["TelTime"](
        engine.telescope.tel_data["TelTime"].size() - 1);

    engine.omb.exposure_time = tn - t0;
    if (engine.run_coadd) {
        engine.cmb.exposure_time =
            engine.cmb.exposure_time + engine.omb.exposure_time;
    }
}

template <class Engine, class DateObs>
void append_observation_date(Engine &engine, DateObs &&date_obs) {
    engine.date_obs.push_back(std::forward<DateObs>(date_obs));
}

template <class Engine, class ConvertUnixToUtc>
auto date_obs_from_telescope_time(Engine &engine,
                                  ConvertUnixToUtc &&convert_unix_to_utc) {
    return convert_unix_to_utc(engine.telescope.tel_data["TelTime"](0));
}

template <class Engine, class Logger>
void configure_fruit_loop_iteration_policy(Engine &engine,
                                           const Logger &logger) {
    if (engine.ptcproc.run_fruit_loops && !engine.run_noise) {
        logger->warn("noise maps are not enabled for fruit loops");
    }

    if (!engine.ptcproc.run_fruit_loops || engine.redu_type == "beammap") {
        engine.ptcproc.fruit_loops_iters = 1;
        engine.ptcproc.save_all_iters = true;
    }
}

}  // namespace citlali::pipeline
