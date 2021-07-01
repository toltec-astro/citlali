#pragma once

namespace timestream {
template <typename Derived>
auto calibrate(TCData<LaliDataKind::PTC,MatrixXd>&in, Eigen::DenseBase<Derived> &fluxscale) {
    //Eigen::Map<Eigen::Vector<double, Eigen::Dynamic,Eigen::RowMajor>> rowMajor_fluxscale(fluxscale.derived().data(), fluxscale.derived().size());
}
} // namespace timestream
