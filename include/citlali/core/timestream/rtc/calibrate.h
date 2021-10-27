#pragma once

#include <Eigen/Core>
namespace timestream {

template <typename DerivedA, typename DerivedB>
void calibrate(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &flxscale) {

    for (Eigen::Index i=0; i<in.cols();i ++) {
        in.col(i) = in.col(i)*flxscale(i);
    }
}

} // namespace
