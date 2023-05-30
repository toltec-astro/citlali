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

    double fsmp;

    // header for tpt table
    std::vector<std::string> det_stats_header = {
        "rms",
        "stddev",
        "median",
        "flagged_frac",
        "weights",
    };

    std::vector<std::string> grp_stats_header = {
        "median_weight"
    };

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
        Eigen::Map<Eigen::Matrix<bool, Eigen::Dynamic, 1>> flags(
            in.flags.data.col(i).data(), in.flags.data.rows());
        // calc rms
        stats["rms"](i,in.index.data) = engine_utils::calc_rms(scans);
        // calc stddev
        stats["stddev"](i,in.index.data) = engine_utils::calc_std_dev(scans);
        // calc median
        stats["median"](i,in.index.data) = tula::alg::median(scans);
        // fraction of detectors that were flagged
        stats["flagged_frac"](i,in.index.data) = flags.cast<double>().sum()/static_cast<double>(n_pts);
    }

    if (in.weights.data.size()!=0) {
        // add weights
        stats["weights"].col(in.index.data) = in.weights.data;
        stats["median_weights"].col(in.index.data) = Eigen::Map<Eigen::VectorXd>(in.median_weights.data(),in.median_weights.size());
    }
}

} // namespace engine
