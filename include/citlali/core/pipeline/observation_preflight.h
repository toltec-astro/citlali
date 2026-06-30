#pragma once

#include <cmath>
#include <string>
#include <vector>

namespace citlali::pipeline {

inline double degrees_to_radians(double degrees) {
    constexpr double deg_to_rad = 0.017453292519943295769;
    return degrees * deg_to_rad;
}

template <class Engine, class RawObs, class Logger>
bool apply_flxscale_correction(Engine &engine, const RawObs &rawobs,
                               const Logger &logger) {
    const auto *flxscale_corr = rawobs.flxscale_correction();
    if (flxscale_corr == nullptr) {
        return true;
    }

    const double factor = flxscale_corr->value();
    if (!std::isfinite(factor) || factor <= 0.0) {
        logger->error(
            "invalid flxscale_correction={} for observation {}; "
            "factor must be finite and > 0",
            factor, rawobs.name());
        return false;
    }
    if (engine.calib.apt.count("flxscale") == 0) {
        logger->error(
            "flxscale column missing from APT while applying "
            "flxscale_correction for observation {}",
            rawobs.name());
        return false;
    }

    engine.calib.apt["flxscale"].array() *= factor;
    logger->info("applied flxscale correction factor={} for observation {}",
                 factor, rawobs.name());
    return true;
}

template <class Engine, class RawObs, class Logger>
void load_array_properties_table(Engine &engine, const RawObs &rawobs,
                                 const Logger &logger) {
    auto apt_path = rawobs.array_prop_table().filepath();
    logger->info("getting array properties table {}", apt_path);

    std::vector<std::string> raw_filenames, interfaces;
    for (const auto &data_item : rawobs.kidsdata()) {
        raw_filenames.push_back(data_item.filepath());
        interfaces.push_back(data_item.interface());
    }

    engine.calib.get_apt(apt_path, raw_filenames, interfaces);
}

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

template <class Engine, class Logger>
void calculate_scan_indices(Engine &engine, const Logger &logger) {
    logger->info("calculating scan indices");
    engine.telescope.calc_scan_indices();
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
