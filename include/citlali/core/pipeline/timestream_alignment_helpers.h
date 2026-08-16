#pragma once

#include <fmt/format.h>
#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/pipeline/timestream_native_pointing.h>

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

struct TimestreamSampleRange {
    Eigen::Index start_index = 0;
    Eigen::Index end_index = 0;

    Eigen::Index size() const {
        return end_index - start_index + 1;
    }
};

struct TimestreamSampleWindow {
    std::vector<Eigen::Index> start_indices;
    std::vector<Eigen::Index> end_indices;
    Eigen::Index min_size = 0;
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

inline TimestreamSampleRange find_common_sample_range(
    const Eigen::VectorXd &times, double max_start, double min_end,
    Eigen::Index interface_index) {
    const Eigen::Index start_index = find_first_sample_at_or_after(
        times, max_start,
        fmt::format("failed to find aligned start sample for interface index {}",
                    interface_index));
    const Eigen::Index end_index = find_last_sample_at_or_before(
        times, min_end, start_index,
        fmt::format("failed to find aligned end sample for interface index {}",
                    interface_index));
    if (end_index < start_index) {
        throw std::runtime_error(fmt::format(
            "invalid aligned sample range for interface index {}: start={} end={}",
            interface_index, start_index, end_index));
    }
    return {start_index, end_index};
}

template <class TimeVectors>
TimestreamSampleWindow find_common_sample_window(
    const TimeVectors &times, double max_start, double min_end) {
    if (times.empty()) {
        throw std::runtime_error("no timestreams available for sample alignment");
    }

    TimestreamSampleWindow window;
    window.start_indices.reserve(times.size());
    window.end_indices.reserve(times.size());
    window.min_size = times[0].size();

    for (Eigen::Index i = 0; i < static_cast<Eigen::Index>(times.size());
         ++i) {
        const auto range = find_common_sample_range(
            times[i], max_start, min_end, i);
        window.start_indices.push_back(range.start_index);
        window.end_indices.push_back(range.end_index);
        if (range.size() < window.min_size) {
            window.min_size = range.size();
        }
    }

    if (window.min_size <= 0) {
        throw std::runtime_error("aligned common timestream length is not positive");
    }
    return window;
}

inline Eigen::VectorXd build_common_gap_time_grid(
    double max_init_time, double min_final_time, double dt,
    const std::string &context_label) {
    if (!std::isfinite(max_init_time) || !std::isfinite(min_final_time) ||
        max_init_time > min_final_time) {
        throw std::runtime_error(fmt::format(
            "no common time overlap across input timestreams with gap interpolation: max_start={} min_end={}",
            max_init_time, min_final_time));
    }
    const Eigen::Index n_samples =
        static_cast<int>((min_final_time - max_init_time) / dt) + 1;
    if (n_samples <= 0) {
        throw std::runtime_error(fmt::format(
            "invalid common sample count in {}: {}",
            context_label, n_samples));
    }
    return Eigen::VectorXd::LinSpaced(
        n_samples, max_init_time, max_init_time + dt * (n_samples - 1));
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

        const auto n_unaligned = mask.size() - mask.sum();
        if (n_unaligned > 0) {
            logger->warn("{}/{} samples were not aligned to the common time grid",
                         n_unaligned, mask.size());
        }
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

inline NativeTelescopeData evaluate_raw_telescope_trajectory_at(
    const RawTelescopeTrajectory &raw_telescope_trajectory,
    const Eigen::VectorXd &target_times_unix_sec) {
    native_pointing_detail::require_strictly_increasing(
        target_times_unix_sec, "telescope evaluation target times");
    if (target_times_unix_sec(0) <
            raw_telescope_trajectory.support_start_unix_sec() ||
        target_times_unix_sec(target_times_unix_sec.size() - 1) >
            raw_telescope_trajectory.support_end_unix_sec()) {
        throw std::out_of_range(
            "telescope evaluation target is outside raw telescope support");
    }

    const auto &raw = raw_telescope_trajectory.telescope_data();
    const auto &support =
        raw_telescope_trajectory.support_times_unix_sec();
    Eigen::Matrix<Eigen::Index, 1, 1> support_shape;
    support_shape << support.size();

    NativeTelescopeData evaluated;
    for (const auto &[key, source] : raw) {
        if (key == "TelTime" || key == "TelUTC") {
            continue;
        }
        Eigen::VectorXd values(target_times_unix_sec.size());
        mlinterp::interp(
            support_shape.data(), values.size(), source.data(), values.data(),
            support.data(), target_times_unix_sec.data());
        if (!values.array().isFinite().all()) {
            throw std::logic_error(
                "telescope interpolation produced nonfinite series '" +
                key + "'");
        }
        evaluated.emplace(key, std::move(values));
    }
    evaluated["TelTime"] = target_times_unix_sec;
    evaluated["TelUTC"] = target_times_unix_sec;
    return evaluated;
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
