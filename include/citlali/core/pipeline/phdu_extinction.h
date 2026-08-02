#pragma once

#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <cmath>

#include <Eigen/Core>

namespace citlali::pipeline {

template <class RTCProc, class Telescope, class Calib, class Logger>
double phdu_mean_tau(bool extinction_enabled, RTCProc &rtcproc,
                     const Telescope &telescope,
                     const TimestreamAlignmentState &alignment,
                     const Calib &calib, Eigen::Index array_slot,
                     const Logger &logger) {
    double mean_tau = 0.0;
    if (!extinction_enabled) {
        return mean_tau;
    }

    const auto tel_it = telescope.tel_data.find("TelElAct");
    if (tel_it == telescope.tel_data.end() || tel_it->second.size() <= 0) {
        logger->warn("MEAN_TAU unavailable (TelElAct missing/empty); defaulting to 0");
        return mean_tau;
    }

    Eigen::VectorXd tau_el(1);
    tau_el << governing_compatibility_mean(
        tel_it->second, alignment);
    auto tau_freq = rtcproc.calibration.calc_tau(tau_el, telescope.tau_225_GHz);
    const auto array_id = calib.arrays(array_slot);
    const auto tau_it = tau_freq.find(array_id);
    if (tau_it != tau_freq.end() && tau_it->second.size() > 0 &&
        std::isfinite(tau_it->second(0))) {
        mean_tau = tau_it->second(0);
    }
    else {
        logger->warn(
            "MEAN_TAU unavailable for array {} (tau_freq missing/empty); defaulting to 0",
            array_id);
    }
    return mean_tau;
}

}  // namespace citlali::pipeline
