#pragma once

#include <fmt/core.h>
#include <Eigen/Core>

#include <stdexcept>
#include <string>

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

}  // namespace citlali::pipeline
