#pragma once

#include <Eigen/Core>

namespace timestream {

template <typename DerivedA, typename DerivedB, typename DerivedC>
void calibrate(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &flxscale, Eigen::DenseBase<DerivedC> &det_index_vector) {
    for (Eigen::Index det=0; det<in.cols(); det++) {
        Eigen::Index di = det_index_vector(det);
        in.col(det) = in.col(det)*flxscale(di);
    }
}

} // namespace
