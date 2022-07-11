#pragma once

#include <Eigen/Core>

#include <citlali/core/utils/constants.h>

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
void estimate_tau(Eigen::DenseBase<DerivedA> &in, Eigen::DenseBase<DerivedB> &el, double tau0) {

    // z is zenith angle
    auto cz = cos(pi/2 - el.derived().array());

    // This is David Tholen’s approximation
    auto A = sqrt(235225.0*cz*cz + 970.0 + 1.0) - 485*cz;

    // observed tau
    auto obs_taui = A*tau0;

    SPDLOG_INFO("obs_taui {}", obs_taui);

    // multiply scan cols by observed tau vector
    in = (in.derived().array().colwise()*exp(obs_taui.array())).eval();
}

} // namespace timestream
