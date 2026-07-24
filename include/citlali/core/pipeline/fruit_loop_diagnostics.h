#pragma once

#include <algorithm>
#include <cmath>
#include <limits>
#include <string_view>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

struct FruitLoopNumericSummary {
    Eigen::Index count = 0;
    double sum = 0.0;
    double sum_squares = 0.0;
    double minimum = std::numeric_limits<double>::quiet_NaN();
    double maximum = std::numeric_limits<double>::quiet_NaN();
    double absolute_maximum = std::numeric_limits<double>::quiet_NaN();

    void add(double value) {
        if (!std::isfinite(value)) {
            return;
        }
        if (count == 0) {
            minimum = value;
            maximum = value;
            absolute_maximum = std::abs(value);
        }
        else {
            minimum = std::min(minimum, value);
            maximum = std::max(maximum, value);
            absolute_maximum = std::max(absolute_maximum, std::abs(value));
        }
        ++count;
        sum += value;
        sum_squares += value * value;
    }

    [[nodiscard]] double rms() const {
        return count > 0
                   ? std::sqrt(sum_squares / static_cast<double>(count))
                   : std::numeric_limits<double>::quiet_NaN();
    }
};

template <class Logger, class Data, class Calib>
void log_fruit_loop_tod_stage(
    const Logger &logger, bool enabled, std::string_view stage,
    Eigen::Index iteration, const Data &data, const Calib &calib) {
    if (!enabled || !logger) {
        return;
    }
    for (Eigen::Index array_pos = 0; array_pos < calib.arrays.size();
         ++array_pos) {
        const auto array_id =
            static_cast<Eigen::Index>(calib.arrays(array_pos));
        FruitLoopNumericSummary signal;
        FruitLoopNumericSummary kernel;
        for (Eigen::Index det = 0; det < data.scans.data.cols(); ++det) {
            if (static_cast<Eigen::Index>(calib.apt.at("array")(det)) !=
                    array_id ||
                calib.apt.at("flag")(det) != 0.0) {
                continue;
            }
            for (Eigen::Index sample = 0; sample < data.scans.data.rows();
                 ++sample) {
                if (!data.flags.data(sample, det)) {
                    signal.add(data.scans.data(sample, det));
                    if (sample < data.kernel.data.rows() &&
                        det < data.kernel.data.cols()) {
                        kernel.add(data.kernel.data(sample, det));
                    }
                }
            }
        }
        logger->info(
            "fruit_loop_diag kind=tod iteration={} scan={} array={} stage={} "
            "signal_count={} signal_sum={:.17g} signal_rms={:.17g} "
            "signal_min={:.17g} signal_max={:.17g} "
            "kernel_count={} kernel_sum={:.17g} kernel_rms={:.17g} "
            "kernel_min={:.17g} kernel_max={:.17g}",
            iteration, data.index.data, array_id, stage, signal.count,
            signal.sum, signal.rms(), signal.minimum, signal.maximum,
            kernel.count, kernel.sum, kernel.rms(), kernel.minimum,
            kernel.maximum);
    }
}

template <class Logger, class Data, class Calib>
void log_fruit_loop_detector_weights(
    const Logger &logger, bool enabled, std::string_view stage,
    Eigen::Index iteration, const Data &data, const Calib &calib) {
    if (!enabled || !logger) {
        return;
    }
    for (Eigen::Index array_pos = 0; array_pos < calib.arrays.size();
         ++array_pos) {
        const auto array_id =
            static_cast<Eigen::Index>(calib.arrays(array_pos));
        FruitLoopNumericSummary summary;
        for (Eigen::Index det = 0; det < data.weights.data.size(); ++det) {
            if (static_cast<Eigen::Index>(calib.apt.at("array")(det)) ==
                    array_id &&
                calib.apt.at("flag")(det) == 0.0) {
                summary.add(data.weights.data(det));
            }
        }
        logger->info(
            "fruit_loop_diag kind=detector_weight iteration={} scan={} "
            "array={} stage={} "
            "count={} sum={:.17g} rms={:.17g} min={:.17g} max={:.17g}",
            iteration, data.index.data, array_id, stage, summary.count,
            summary.sum, summary.rms(), summary.minimum, summary.maximum);
    }
}

template <class Logger>
void log_fruit_loop_projection_summary(
    const Logger &logger, bool enabled, Eigen::Index scan,
    Eigen::Index iteration, Eigen::Index array_id, std::string_view action,
    const FruitLoopNumericSummary &signal,
    const FruitLoopNumericSummary &kernel) {
    if (!enabled || !logger) {
        return;
    }
    logger->info(
        "fruit_loop_diag kind=projection iteration={} scan={} array={} "
        "action={} "
        "signal_count={} signal_sum={:.17g} signal_rms={:.17g} "
        "kernel_count={} kernel_sum={:.17g} kernel_rms={:.17g}",
        iteration, scan, array_id, action, signal.count, signal.sum,
        signal.rms(), kernel.count, kernel.sum, kernel.rms());
}

template <class Logger, class MapBuffer, class MapIndices, class Calib>
void log_fruit_loop_map_models(
    const Logger &logger, bool enabled, Eigen::Index scan,
    Eigen::Index iteration, std::string_view action, const MapBuffer &maps,
    const MapIndices &map_indices, const Calib &calib) {
    if (!enabled || !logger) {
        return;
    }
    std::vector<bool> logged(maps.signal.size(), false);
    for (Eigen::Index det = 0; det < map_indices.size(); ++det) {
        const auto map_index = static_cast<Eigen::Index>(map_indices(det));
        if (map_index < 0 ||
            map_index >= static_cast<Eigen::Index>(maps.signal.size()) ||
            logged[static_cast<std::size_t>(map_index)]) {
            continue;
        }
        logged[static_cast<std::size_t>(map_index)] = true;
        FruitLoopNumericSummary signal;
        for (Eigen::Index col = 0; col < maps.signal[map_index].cols(); ++col) {
            for (Eigen::Index row = 0; row < maps.signal[map_index].rows();
                 ++row) {
                signal.add(maps.signal[map_index](row, col));
            }
        }
        FruitLoopNumericSummary kernel;
        if (map_index < static_cast<Eigen::Index>(maps.kernel.size())) {
            for (Eigen::Index col = 0;
                 col < maps.kernel[map_index].cols(); ++col) {
                for (Eigen::Index row = 0;
                     row < maps.kernel[map_index].rows(); ++row) {
                    kernel.add(maps.kernel[map_index](row, col));
                }
            }
        }
        FruitLoopNumericSummary weight;
        if (map_index < static_cast<Eigen::Index>(maps.weight.size())) {
            for (Eigen::Index col = 0;
                 col < maps.weight[map_index].cols(); ++col) {
                for (Eigen::Index row = 0;
                     row < maps.weight[map_index].rows(); ++row) {
                    weight.add(maps.weight[map_index](row, col));
                }
            }
        }
        const auto array_id =
            static_cast<Eigen::Index>(calib.apt.at("array")(det));
        logger->info(
            "fruit_loop_diag kind=map_model iteration={} scan={} array={} "
            "map={} action={} "
            "signal_count={} signal_sum={:.17g} signal_rms={:.17g} "
            "signal_min={:.17g} signal_max={:.17g} signal_abs_max={:.17g} "
            "kernel_count={} kernel_sum={:.17g} kernel_rms={:.17g} "
            "kernel_abs_max={:.17g} "
            "weight_count={} weight_sum={:.17g} weight_rms={:.17g} "
            "weight_min={:.17g} weight_max={:.17g}",
            iteration, scan, array_id, map_index, action, signal.count,
            signal.sum, signal.rms(), signal.minimum, signal.maximum,
            signal.absolute_maximum, kernel.count, kernel.sum, kernel.rms(),
            kernel.absolute_maximum, weight.count, weight.sum, weight.rms(),
            weight.minimum, weight.maximum);
    }
}

}  // namespace citlali::pipeline
