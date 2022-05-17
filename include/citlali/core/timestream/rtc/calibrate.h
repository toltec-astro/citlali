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

} // namespace
