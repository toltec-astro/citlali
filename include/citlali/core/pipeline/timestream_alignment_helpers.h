#pragma once

#include <fmt/core.h>
#include <Eigen/Core>
#include <tula/algorithm/mlinterp/mlinterp.hpp>

#include <citlali/core/pipeline/sci_align_contract.h>
#include <citlali/core/pipeline/sci_align_field_registry.h>
#include <citlali/core/pipeline/timestream_alignment_state.h>

#include <algorithm>
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

// Select rows on the closed native telescope-support interval. This is an
// admission operation, not a timestamp transformation.
inline TimestreamSampleRange find_telescope_supported_detector_range(
    const Eigen::VectorXd &detector_time,
    const Eigen::VectorXd &native_telescope_time,
    const std::string &interface_id) {
    sci_align::require_strictly_increasing(
        detector_time, "detector interface time");
    sci_align::require_strictly_increasing(
        native_telescope_time, "native telescope TelTime");
    const auto begin_it = std::lower_bound(
        detector_time.data(), detector_time.data() + detector_time.size(),
        native_telescope_time[0]);
    const auto end_it = std::upper_bound(
        detector_time.data(), detector_time.data() + detector_time.size(),
        native_telescope_time[native_telescope_time.size() - 1]);
    const auto start = static_cast<Eigen::Index>(
        begin_it - detector_time.data());
    const auto stop = static_cast<Eigen::Index>(end_it - detector_time.data());
    if (start >= stop) {
        throw std::runtime_error(fmt::format(
            "detector interface '{}' has no rows within native telescope support",
            interface_id));
    }
    return {start, stop - 1};
}

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
        sci_align::require_strictly_increasing(t, "detector interface time");
        Eigen::VectorXi mask = Eigen::VectorXi::Zero(t_common.size());

        for (int i = 0; i < t.size(); ++i) {
            const double time = t(i);
            const auto idx64 = sci_align::round_half_up_slot(
                (time - max_init_time) / dt);
            if (idx64 < std::numeric_limits<int>::min() ||
                idx64 > std::numeric_limits<int>::max()) {
                throw std::runtime_error("common-grid slot exceeds integer mask range");
            }
            const int idx = static_cast<int>(idx64);
            if (idx >= 0 && idx < t_common.size() &&
                std::abs(time - t_common(idx)) < tol) {
                if (mask(idx) != 0) {
                    throw std::runtime_error(
                        "multiple native detector rows collide on one common-grid slot");
                }
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
    bool skip_tel_utc_during_loop,
    TimestreamAlignmentState *alignment_state = nullptr) {
    (void)skip_tel_utc_during_loop;
    sci_align::require_strictly_increasing(common_time,
                                           "detector common time");
    const auto coordinate_it = tel_data.find("TelTime");
    if (coordinate_it == tel_data.end()) {
        throw std::runtime_error("native telescope TelTime is unavailable");
    }
    const Eigen::VectorXd native_time = coordinate_it->second;
    sci_align::require_strictly_increasing(native_time,
                                           "native telescope TelTime");
    if (common_time[0] < native_time[0] ||
        common_time[common_time.size() - 1] >
            native_time[native_time.size() - 1]) {
        throw std::runtime_error(
            "required telescope support does not bracket the detector common grid; extrapolation is prohibited");
    }

    AlignmentTelescopeSummary telescope_summary;
    telescope_summary.initialized = true;
    telescope_summary.native_row_count = native_time.size();
    telescope_summary.native_first_coordinate_sec = native_time[0];
    telescope_summary.native_last_coordinate_sec =
        native_time[native_time.size() - 1];
    telescope_summary.native_tel_utc_available =
        tel_data.find("TelUTC") != tel_data.end();
    telescope_summary.native_pps_time_available =
        tel_data.find("PpsTime") != tel_data.end();
    double minimum_bracket_span = std::numeric_limits<double>::max();
    Eigen::Index upper = 0;
    for (Eigen::Index target = 0; target < common_time.size(); ++target) {
        while (upper < native_time.size() &&
               native_time[upper] < common_time[target]) {
            ++upper;
        }
        if (upper < native_time.size() &&
            native_time[upper] == common_time[target]) {
            ++telescope_summary.exact_target_count;
            continue;
        }
        if (upper == 0 || upper >= native_time.size()) {
            throw std::runtime_error(
                "telescope target lacks a valid adjacent native bracket");
        }
        const double span = native_time[upper] - native_time[upper - 1];
        if (!std::isfinite(span) || !(span > 0.0)) {
            throw std::runtime_error(
                "telescope target has an invalid adjacent native bracket");
        }
        ++telescope_summary.interpolated_target_count;
        minimum_bracket_span = std::min(minimum_bracket_span, span);
        telescope_summary.maximum_used_bracket_span_sec = std::max(
            telescope_summary.maximum_used_bracket_span_sec, span);
    }
    if (telescope_summary.interpolated_target_count != 0) {
        telescope_summary.minimum_used_bracket_span_sec =
            minimum_bracket_span;
    }

    Eigen::Matrix<Eigen::Index,1,1> nd;
    nd << native_time.size();
    TelData aligned;

    for (const auto &entry : sci_align::active_field_registry) {
        const std::string key{entry.canonical_name};
        if (entry.permitted_operator ==
                sci_align::FieldOperator::native_coordinate ||
            entry.permitted_operator ==
                sci_align::FieldOperator::exact_diagnostic) {
            continue;
        }
        const auto source_it = tel_data.find(key);
        if (source_it == tel_data.end()) {
            if (entry.required_for_admitted_intensity_profile) {
                throw std::runtime_error(fmt::format(
                    "required active telescope field '{}' is unavailable",
                    key));
            }
            continue;
        }
        const Eigen::VectorXd &yd = source_it->second;
        if (yd.size() != native_time.size() || !yd.allFinite()) {
            throw std::runtime_error(fmt::format(
                "active telescope field '{}' has invalid shape or nonfinite values",
                key));
        }

        Eigen::VectorXd interpolation_source = yd;
        const bool is_circular = entry.permitted_operator ==
            sci_align::FieldOperator::bracketed_shortest_arc;
        if (is_circular) {
            constexpr double pi_value = 3.141592653589793238462643383279502884;
            constexpr double two_pi_value = 2.0 * pi_value;
            double branch_offset = 0.0;
            for (Eigen::Index i = 1; i < native_time.size(); ++i) {
                const double raw_difference = yd(i) - yd(i - 1);
                double difference = std::fmod(raw_difference, two_pi_value);
                if (difference <= -pi_value) {
                    difference += two_pi_value;
                }
                else if (difference > pi_value) {
                    difference -= two_pi_value;
                }
                if (sci_align::machine_equal(std::abs(difference), pi_value)) {
                    throw std::runtime_error(fmt::format(
                        "active circular telescope field '{}' has an ambiguous antipodal bracket",
                        key));
                }
                branch_offset += difference - raw_difference;
                // With no wrap, this is exactly the native value rather than
                // a recurrence that can accumulate rounding differences.
                interpolation_source(i) = yd(i) + branch_offset;
            }
        }

        Eigen::VectorXd yi(common_time.size());

        mlinterp::interp(nd.data(), common_time.size(),
                         interpolation_source.data(), yi.data(),
                         native_time.data(), common_time.data());
        if (is_circular) {
            constexpr double two_pi_value =
                6.283185307179586476925286766559005768;
            // Both axes are strictly increasing.  Walk the native bracket
            // once for this field instead of performing a binary search for
            // every target sample in this established setup hot path.
            Eigen::Index upper = 0;
            for (Eigen::Index target = 0; target < yi.size(); ++target) {
                while (upper < native_time.size() &&
                       native_time(upper) < common_time(target)) {
                    ++upper;
                }
                if (upper < native_time.size() &&
                    native_time(upper) == common_time(target)) {
                    // Exact coincidences preserve the producer's numerical
                    // representation bit-for-bit.
                    yi(target) = yd(upper);
                    continue;
                }
                if (upper <= 0 || upper >= native_time.size()) {
                    throw std::runtime_error(fmt::format(
                        "active circular telescope field '{}' target lacks an adjacent bracket",
                        key));
                }
                const Eigen::Index lower = upper - 1;
                const double lambda =
                    (common_time(target) - native_time(lower)) /
                    (native_time(upper) - native_time(lower));
                const double raw_linear_reference =
                    (1.0 - lambda) * yd(lower) + lambda * yd(upper);
                yi(target) = sci_align::nearest_periodic_equivalent(
                    yi(target), raw_linear_reference, two_pi_value);
            }
        }
        if (!yi.allFinite()) {
            throw std::runtime_error(fmt::format(
                "alignment produced a nonfinite telescope field '{}'",
                key));
        }
        aligned[key] = std::move(yi);
    }

    aligned["TelTime"] = common_time;
    aligned["TelUTC"] = common_time;
    tel_data = std::move(aligned);
    if (alignment_state != nullptr) {
        alignment_state->telescope = std::move(telescope_summary);
    }
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
