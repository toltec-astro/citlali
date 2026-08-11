#pragma once

#include <map>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

inline constexpr double rtc_sampling_source_max_speed_arcsec_s = 3600.0;
inline constexpr double rtc_sampling_source_min_eligible_speed_arcsec_s = 1.0;
inline constexpr double rtc_sampling_source_max_interval_s = 0.1;
inline constexpr double rtc_sampling_source_max_pointing_step_rad = 0.01;
inline constexpr std::size_t rtc_sampling_source_boundary_guard_rows = 1;
inline constexpr const char *rtc_sampling_source_guard_version =
    "native-source-row-gap-jump-and-one-row-scan-boundary-v1";

// Compact observation-scoped authority for RTC learned-sampling diagnostics.
// It is populated from source telescope rows before detector-grid
// interpolation.  It deliberately retains intervals, not the source samples,
// and is never consumed by RTC, PTC, or mapmaking.
struct RtcSamplingSourceMotionInterval {
    std::size_t start_row_index = 0;
    std::size_t stop_row_index = 0;
    double start_time_s = std::numeric_limits<double>::quiet_NaN();
    double stop_time_s = std::numeric_limits<double>::quiet_NaN();
    double duration_s = std::numeric_limits<double>::quiet_NaN();
    double speed_arcsec_s = std::numeric_limits<double>::quiet_NaN();
    bool valid = false;
    bool eligible = false;
    std::string reason;
};

struct RtcSamplingSourceMotionSupport {
    std::string authority{
        "source-telescope-rows-before-detector-grid-interpolation"};
    std::string coordinate_identity{
        "delta-source-altaz-tangent-plane-v1"};
    std::string status{"unavailable"};
    std::string reason{"missing_source_motion_columns"};
    bool observation_identity_available = false;
    std::size_t observation_index = std::numeric_limits<std::size_t>::max();
    std::string observation_obsnum;
    std::string telescope_source_path;
    std::size_t source_row_count = 0;
    std::size_t interval_count = 0;
    std::size_t valid_interval_count = 0;
    std::size_t rejected_interval_count = 0;
    std::size_t eligible_interval_count = 0;
    std::size_t low_velocity_excluded_count = 0;
    double valid_duration_s = 0.0;
    double eligible_duration_s = 0.0;
    double low_velocity_excluded_duration_s = 0.0;
    std::vector<RtcSamplingSourceMotionInterval> intervals;
};

inline void bind_rtc_sampling_source_observation_identity(
    RtcSamplingSourceMotionSupport &support, std::size_t observation_index,
    std::string observation_obsnum, std::string telescope_source_path) {
    support.observation_index = observation_index;
    support.observation_obsnum = std::move(observation_obsnum);
    support.telescope_source_path = std::move(telescope_source_path);
    support.observation_identity_available =
        !support.observation_obsnum.empty() &&
        !support.telescope_source_path.empty();
}

inline bool rtc_sampling_source_observation_matches(
    const RtcSamplingSourceMotionSupport &support,
    std::size_t observation_index, const std::string &observation_obsnum,
    const std::string &telescope_source_path) {
    return support.observation_identity_available &&
           support.observation_index == observation_index &&
           support.observation_obsnum == observation_obsnum &&
           support.telescope_source_path == telescope_source_path;
}

struct TimestreamAlignmentState {
    Eigen::VectorXd common_time;
    std::vector<Eigen::VectorXi> masks;
    std::map<Eigen::Index, Eigen::VectorXi> network_masks;
    std::vector<Eigen::VectorXd> network_times;
    std::map<std::string, int> gaps;
    std::vector<Eigen::Index> start_indices;
    std::vector<Eigen::Index> end_indices;
    Eigen::Index hwpr_start_index = 0;
    Eigen::Index hwpr_end_index = 0;
    RtcSamplingSourceMotionSupport rtc_sampling_source_motion;
};

inline void clear_alignment_windows(TimestreamAlignmentState &state) {
    state.start_indices.clear();
    state.end_indices.clear();
}

inline void clear_gap_alignment_state(TimestreamAlignmentState &state) {
    clear_alignment_windows(state);
    state.network_masks.clear();
    state.gaps.clear();
}

inline void reset_rtc_sampling_source_motion(
    TimestreamAlignmentState &state) {
    state.rtc_sampling_source_motion = {};
}

}  // namespace citlali::pipeline
