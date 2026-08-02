#pragma once

// Implementation detail included by todproc.h.

#include <citlali/core/config/calibration_config.h>
#include <citlali/core/error/error.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

template <class EngineType>
void TimeOrderedDataProc<EngineType>::interp_pointing() {
    const auto az_it = engine().pointing_offsets.arcsec.find(
        citlali::config::pointing_axis_az());
    const auto alt_it = engine().pointing_offsets.arcsec.find(
        citlali::config::pointing_axis_alt());
    if (az_it == engine().pointing_offsets.arcsec.end() || alt_it == engine().pointing_offsets.arcsec.end()) {
        logger->error("pointing_offsets must include both az and alt vectors");
        throw citlali::error::invalid_config(
            "pointing_offsets must include both az and alt vectors");
    }

    // how many offsets in config file
    Eigen::Index n_offsets = az_it->second.size();
    if (n_offsets != alt_it->second.size()) {
        logger->error("pointing_offsets az/alt lengths differ (az={} alt={})",
                      n_offsets, alt_it->second.size());
        throw citlali::error::invalid_config(
            "pointing_offsets az/alt lengths differ");
    }
    if (n_offsets != 1 && n_offsets != 2) {
        logger->error("only one or two values for altaz offsets are supported");
        throw citlali::error::invalid_config(
            "only one or two values for altaz offsets are supported");
    }

    const Eigen::Index ni = engine().telescope.tel_data["TelTime"].size();
    if (ni <= 0) {
        logger->error("cannot interpolate pointing offsets: telescope TelTime is empty");
        throw citlali::error::runtime(
            "cannot interpolate pointing offsets: telescope TelTime is empty");
    }
    const auto &tel_time = engine().telescope.tel_data["TelTime"];
    const double governing_start_time = engine().alignment.grid.initialized
        ? citlali::pipeline::governing_compatibility_start_value(
              tel_time, engine().alignment)
        : tel_time(0);
    const double governing_stop_time = engine().alignment.grid.initialized
        ? citlali::pipeline::governing_compatibility_stop_value(
              tel_time, engine().alignment)
        : tel_time(ni - 1);

    // keys for pointing offsets
    std::vector<std::string> altaz_keys = {
        citlali::config::pointing_axis_alt(),
        citlali::config::pointing_axis_az()};

    for (const auto &key: altaz_keys) {
        // if only one value given
        if (n_offsets==1) {
            double offset = engine().pointing_offsets.arcsec[key](0);
            engine().pointing_offsets.arcsec[key].resize(ni);
            engine().pointing_offsets.arcsec[key].setConstant(offset);
        }
        else if (n_offsets==2) {
            // size of offset data
            Eigen::Matrix<Eigen::Index,1,1> nd;
            nd << n_offsets;

            // vector to store interpolation
            Eigen::VectorXd yi(ni);

            // start and end times of observation
            Eigen::VectorXd xd(n_offsets);
            const bool use_mjd = (engine().pointing_offsets.modified_julian_date.size() == 2) &&
                                 (engine().pointing_offsets.modified_julian_date > 0).all();

            // use start and end of current obs if MJD values are not specified
            if (!use_mjd) {
                xd << governing_start_time, governing_stop_time;
            }
            // else use specified modified julian dates, convert to julian dates, and calc unix time
            else {
                xd << engine_utils::modified_julian_date_to_unix(engine().pointing_offsets.modified_julian_date(0)),
                    engine_utils::modified_julian_date_to_unix(engine().pointing_offsets.modified_julian_date(1));

                if (xd(1) <= xd(0)) {
                    logger->error("MJD range is invalid: end <= start");
                    throw citlali::error::invalid_config(
                        "pointing offset MJD range is invalid: end <= start");
                }
                // make sure offsets are before and after the observation
                if (xd(0) > governing_start_time ||
                    xd(1) < governing_stop_time) {
                    logger->error("MJD range is invalid");
                    throw citlali::error::invalid_config(
                        "pointing offset MJD range does not bracket the observation");
                }
            }

            // interpolate offset onto time vector
            mlinterp::interp(nd.data(), ni, // nd, ni
                             engine().pointing_offsets.arcsec[key].data(), yi.data(), // yd, yi
                             xd.data(), tel_time.data()); // xd, xi

            // overwrite pointing offsets
            engine().pointing_offsets.arcsec[key] = yi;
        }
    }
}
