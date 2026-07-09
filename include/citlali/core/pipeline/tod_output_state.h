#pragma once

#include <Eigen/Core>

namespace citlali::pipeline {

struct TodOutputState {
    Eigen::VectorXI rtc_scan_to_output_scan;
    Eigen::VectorXI ptc_scan_to_output_scan;
    Eigen::Index n_rtc_output_scans = 0;
    Eigen::Index n_ptc_output_scans = 0;
};

inline void reset(TodOutputState &state) {
    state.rtc_scan_to_output_scan.resize(0);
    state.ptc_scan_to_output_scan.resize(0);
    state.n_rtc_output_scans = 0;
    state.n_ptc_output_scans = 0;
}

}  // namespace citlali::pipeline
