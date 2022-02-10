#pragma once

#include <Eigen/Core>

namespace timestream {

template <typename DerivedA, typename DerivedB>
void calibrate(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &flxscale) {
    for (Eigen::Index det=0; det<in.cols(); det++) {
        in.col(det) = in.col(det)*flxscale(det);
    }
}

} // namespace
