#pragma once

#include <map>
#include <string>
#include <vector>

#include <Eigen/Core>

namespace citlali::pipeline {

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

}  // namespace citlali::pipeline
