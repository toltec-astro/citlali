#pragma once

#include <cmath>
#include <string>

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

}  // namespace citlali::pipeline
