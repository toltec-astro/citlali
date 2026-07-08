#pragma once

#include <fmt/core.h>
#include <Eigen/Core>

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace citlali::pipeline {

inline Eigen::Index find_first_sample_at_or_after(
    const Eigen::VectorXd &times, double target_time,
    const std::string &error_message) {
    if (times.size() == 0) {
        throw std::runtime_error(error_message);
    }
    Eigen::Index sample_index = 0;
    (times.array() - target_time).abs().minCoeff(&sample_index);
    while (sample_index < times.size() && times[sample_index] < target_time) {
        sample_index++;
    }
    if (sample_index >= times.size()) {
        throw std::runtime_error(error_message);
    }
    return sample_index;
}

inline Eigen::Index find_last_sample_at_or_before(
    const Eigen::VectorXd &times, double target_time,
    Eigen::Index start_index, const std::string &error_message) {
    if (times.size() == 0) {
        throw std::runtime_error(error_message);
    }
    Eigen::Index sample_index = 0;
    (times.array() - target_time).abs().minCoeff(&sample_index);
    while (sample_index >= 0 && times[sample_index] > target_time) {
        sample_index--;
    }
    if (sample_index < 0 || sample_index < start_index) {
        throw std::runtime_error(error_message);
    }
    return sample_index;
}

inline void validate_hwpr_alignment_inputs(
    const Eigen::VectorXd &recvt, const Eigen::VectorXd &angle,
    const std::string &alignment_label) {
    if (recvt.size() == 0 || angle.size() == 0) {
        throw std::runtime_error(
            "HWPR is enabled but HWP time/angle data are empty");
    }
    if (recvt.size() != angle.size()) {
        throw std::runtime_error(
            fmt::format("HWPR time and angle vectors have different lengths before {} alignment",
                        alignment_label));
    }
}

struct TimestreamOverlap {
    double max_start = std::numeric_limits<double>::lowest();
    double min_end = std::numeric_limits<double>::max();
    Eigen::Index max_start_index = 0;
    Eigen::Index min_end_index = 0;
};

template <class TimeVectors>
TimestreamOverlap find_common_timestream_overlap(
    const TimeVectors &times, const std::string &context_label) {
    TimestreamOverlap overlap;
    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(times.size());
         ++i) {
        if (times[i].size() == 0) {
            throw std::runtime_error(fmt::format(
                "empty time vector for interface index {} in {}",
                i, context_label));
        }
        const double initial_time = times[i](0);
        const double final_time = times[i](times[i].size() - 1);

        if (initial_time > overlap.max_start) {
            overlap.max_start = initial_time;
            overlap.max_start_index = i;
        }
        if (final_time < overlap.min_end) {
            overlap.min_end = final_time;
            overlap.min_end_index = i;
        }
    }
    return overlap;
}

template <class TimeVectors, class Logger>
std::vector<Eigen::VectorXi> build_common_time_grid_masks(
    const TimeVectors &times, const Eigen::VectorXd &t_common,
    double max_init_time, double dt, double tol, const Logger &logger) {
    std::vector<Eigen::VectorXi> masks;
    masks.reserve(times.size());

    for (const auto &t : times) {
        Eigen::VectorXi mask = Eigen::VectorXi::Zero(t_common.size());

        for (int i = 0; i < t.size(); ++i) {
            const double time = t(i);
            const int idx =
                static_cast<int>(std::round((time - max_init_time) / dt));
            if (idx >= 0 && idx < t_common.size() &&
                std::abs(time - t_common(idx)) <= tol) {
                mask(idx) = 1;
            }
        }

        logger->warn("{}/{} samples were not aligned to the common time grid",
                     mask.size() - mask.sum(), mask.size());
        masks.push_back(std::move(mask));
    }

    return masks;
}

}  // namespace citlali::pipeline
