#pragma once

#include <Eigen/Core>

namespace timestream {

template <typename DerivedA, typename DerivedB, typename DerivedC, typename DerivedD>
void calibrate(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &flxscale,
               Eigen::DenseBase<DerivedC> &map_index_vector, Eigen::DenseBase<DerivedC> &det_index_vector,
               Eigen::DenseBase<DerivedD> &cflux) {

    for (Eigen::Index det=0; det<in.cols(); det++) {
        Eigen::Index di = det_index_vector(det);
        Eigen::Index mi = map_index_vector(det);

        in.col(det) = in.col(det)*flxscale(di)*cflux(mi);
    }
}

// estimate opacity
template <typename DerivedA, typename DerivedB>
auto estimate_tau(Eigen::DenseBase<DerivedB> &scans, Eigen::DenseBase<DerivedB> &dc2tau) {

    Eigen::MatrixXd in_temp = scans.colwise().mean();
    Eigen::VectorXd estimated_tau(scans.cols());

    if (dc2tau.row(0) < 9998){
        estimated_tau = dc2tau.row(2).array()*pow(in_temp.array(),2) + dc2tau.row(1).array()*in_temp.array() + dc2tau.row(0).array();
    }

    else {
        estimated_tau.setConstant(-9999.);
    }
    return estimated_tau;
}

} // namespace
