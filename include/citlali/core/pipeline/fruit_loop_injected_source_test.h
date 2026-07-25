#pragma once

#include <citlali/core/config/timestream_config.h>
#include <citlali/core/error/error.h>

#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace citlali::pipeline {

struct FruitLoopInjectedSourceArraySummary {
    Eigen::Index array_id = -1;
    double amplitude_mjy_beam = 0.0;
    long long projected_samples = 0;
    double projected_sum = 0.0;
    double projected_square_sum = 0.0;
    double projected_peak = 0.0;
};

struct FruitLoopInjectedSourceSummary {
    bool applied = false;
    long long projected_samples = 0;
    std::vector<FruitLoopInjectedSourceArraySummary> arrays;
};

inline bool fruit_loop_injected_source_test_active(
    const citlali::config::FruitLoopsInjectedSourceTestConfig &config,
    int iteration) {
    return config.enabled && iteration >= config.start_iteration;
}

template <class PtcData, class Calib>
FruitLoopInjectedSourceSummary inject_fruit_loop_test_source(
    PtcData &ptcdata, const Calib &calib,
    const citlali::config::FruitLoopsInjectedSourceTestConfig &config,
    int iteration, const std::string &signal_unit) {
    FruitLoopInjectedSourceSummary summary;
    if (!fruit_loop_injected_source_test_active(config, iteration)) {
        return summary;
    }
    if (signal_unit != "mJy/beam") {
        throw citlali::error::runtime(
            "fruit-loop injected-source test requires mJy/beam signal units");
    }
    if (ptcdata.kernel.data.rows() != ptcdata.scans.data.rows() ||
        ptcdata.kernel.data.cols() != ptcdata.scans.data.cols() ||
        ptcdata.kernel.data.size() == 0) {
        throw citlali::error::runtime(
            "fruit-loop injected-source test requires a kernel timestream "
            "matching the signal timestream");
    }
    if (!ptcdata.kernel.data.allFinite()) {
        throw citlali::error::runtime(
            "fruit-loop injected-source test kernel contains non-finite values");
    }
    if (calib.apt.at("array").size() != ptcdata.scans.data.cols()) {
        throw citlali::error::runtime(
            "fruit-loop injected-source test detector array metadata does not "
            "match the signal timestream");
    }

    summary.applied = true;
    summary.arrays.resize(config.array_amplitude_mjy_beam.size());
    for (std::size_t i = 0; i < summary.arrays.size(); ++i) {
        auto &array = summary.arrays[i];
        array.array_id = static_cast<Eigen::Index>(i);
        array.amplitude_mjy_beam = config.array_amplitude_mjy_beam[i];
    }

    for (Eigen::Index detector = 0;
         detector < ptcdata.scans.data.cols(); ++detector) {
        const double raw_array = calib.apt.at("array")(detector);
        if (!std::isfinite(raw_array)) {
            throw citlali::error::runtime(
                "fruit-loop injected-source test encountered non-finite "
                "detector array identity");
        }
        const auto array_id =
            static_cast<Eigen::Index>(std::llround(raw_array));
        if (std::abs(raw_array - static_cast<double>(array_id)) > 1.0e-9 ||
            array_id < 0 ||
            array_id >= static_cast<Eigen::Index>(
                config.array_amplitude_mjy_beam.size())) {
            throw citlali::error::runtime(
                "fruit-loop injected-source test encountered invalid detector "
                "array identity");
        }

        const double amplitude =
            config.array_amplitude_mjy_beam[
                static_cast<std::size_t>(array_id)];
        const auto projected =
            amplitude * ptcdata.kernel.data.col(detector).array();
        ptcdata.scans.data.col(detector).array() += projected;

        auto &array = summary.arrays[static_cast<std::size_t>(array_id)];
        for (Eigen::Index sample = 0; sample < projected.size(); ++sample) {
            const double value = projected(sample);
            if (value == 0.0) {
                continue;
            }
            ++array.projected_samples;
            ++summary.projected_samples;
            array.projected_sum += value;
            array.projected_square_sum += value * value;
            array.projected_peak =
                std::max(array.projected_peak, std::abs(value));
        }
    }
    return summary;
}

template <class Logger>
void log_fruit_loop_injected_source_summary(
    const Logger &logger, int iteration, Eigen::Index scan_index,
    const FruitLoopInjectedSourceSummary &summary) {
    if (!summary.applied) {
        return;
    }
    for (const auto &array : summary.arrays) {
        const double rms =
            array.projected_samples > 0
                ? std::sqrt(
                      array.projected_square_sum /
                      static_cast<double>(array.projected_samples))
                : 0.0;
        logger->info(
            "fruit-loop injected-source test iteration={} scan={} array={} "
            "amplitude_mJy_beam={} projected_samples={} projected_sum={} "
            "projected_rms={} projected_peak={}",
            iteration, scan_index + 1, array.array_id,
            array.amplitude_mjy_beam, array.projected_samples,
            array.projected_sum, rms, array.projected_peak);
    }
}

}  // namespace citlali::pipeline
