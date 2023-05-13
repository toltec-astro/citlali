#pragma once

#include <string>
#include <map>

#include <Eigen/Core>

#include <citlali/core/utils/utils.h>

#include <citlali/core/timestream/timestream.h>

namespace engine {

class Diagnostics {
public:
    std::map<std::string, Eigen::MatrixXd> stats;

    // header for tpt table
    std::vector<std::string> tpt_header = {
        "rms",
        "stddev",
        "median"
    };

    // tpt header units
    std::map<std::string,std::string> tpt_header_units;

    template <timestream::TCDataKind tcdata_kind>
    void calc_stats(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &);
};

template <timestream::TCDataKind tcdata_kind>
void Diagnostics::calc_stats(timestream::TCData<tcdata_kind, Eigen::MatrixXd> &in) {
    Eigen::Index n_pts = in.scans.data.rows();
    Eigen::Index n_dets = in.scans.data.cols();

    for (Eigen::Index i=0; i<n_dets; i++) {
        // make Eigen::Maps for each detector's scan
        Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, 1>> scans(
            in.scans.data.col(i).data(), in.scans.data.rows());
        stats["rms"](i,in.index.data) = engine_utils::calc_rms(scans);
        stats["stddev"](i,in.index.data) = engine_utils::calc_std_dev(scans);
        stats["median"](i,in.index.data) = tula::alg::median(scans);
    }
}


} // namespace engine
