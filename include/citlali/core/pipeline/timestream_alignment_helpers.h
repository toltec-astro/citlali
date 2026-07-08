#pragma once

#include <fmt/core.h>
#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

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

template <class TelData>
void interpolate_telescope_data_to_common_time(
    TelData &tel_data, const Eigen::VectorXd &common_time,
    bool skip_tel_utc_during_loop) {
    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << tel_data["TelTime"].size();

    for (const auto &tel_it : tel_data) {
        if (tel_it.first == "TelTime" ||
            (skip_tel_utc_during_loop && tel_it.first == "TelUTC")) {
            continue;
        }
        Eigen::VectorXd yd = tel_data[tel_it.first];
        Eigen::VectorXd yi(common_time.size());

        mlinterp::interp(nd.data(), common_time.size(),
                         yd.data(), yi.data(),
                         tel_data["TelTime"].data(), common_time.data());
        tel_data[tel_it.first] = std::move(yi);
    }

    tel_data["TelTime"] = common_time;
    tel_data["TelUTC"] = common_time;
}

inline void interpolate_hwpr_angle_to_common_time(
    Eigen::VectorXd &hwpr_angle, const Eigen::VectorXd &hwpr_time,
    const Eigen::VectorXd &common_time) {
    Eigen::Matrix<Eigen::Index,1,1> hwpr_nd;
    hwpr_nd << hwpr_time.size();
    Eigen::VectorXd yd = hwpr_angle;
    Eigen::VectorXd yi(common_time.size());

    mlinterp::interp(hwpr_nd.data(), common_time.size(),
                     yd.data(), yi.data(), hwpr_time.data(),
                     common_time.data());
    hwpr_angle = std::move(yi);
}

template <class TimeMatrix>
int count_packet_counter_gaps(const TimeMatrix &ts) {
    const Eigen::Index n_pts = ts.rows();
    if (n_pts <= 1) {
        return 0;
    }
    return ((ts.block(1,3,n_pts-1,1).array() -
             ts.block(0,3,n_pts-1,1).array()).array() > 1).count();
}

inline Eigen::VectorXd network_time_from_timestream_matrix(
    const Eigen::MatrixXd &ts_double, double fpga_freq,
    double interface_sync_offset) {
    auto sec = ts_double.col(0);
    auto nsec = ts_double.col(5);
    auto pps = ts_double.col(1);
    auto msec = ts_double.col(2) / fpga_freq;
    auto pps_msec = ts_double.col(4) / fpga_freq;

    double start_time_dbl = sec[0] + nsec[0] * 1e-9;
    const int start_time = int(start_time_dbl - 0.5);
    start_time_dbl = start_time;

    Eigen::VectorXd dt = msec - pps_msec;
    dt = (dt.array() < 0).select(
        msec.array() - pps_msec.array() +
            (std::pow(2.0,32) - 1) / fpga_freq,
        msec - pps_msec);

    return start_time_dbl + pps.array() + dt.array() +
           interface_sync_offset;
}

}  // namespace citlali::pipeline
