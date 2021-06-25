#pragma once

namespace timestream {
template <typename Derived>
auto calibrate(TCData<LaliDataKind::PTC,MatrixXd>&in, Eigen::DenseBase<Derived> &fluxscale) {
    in.scans.data.array().colwise()*fluxscale.derived().array();
    // in.kernelscans.data.array().colwise()*fluxscale.derived().array();
}
} // namespace timestream
