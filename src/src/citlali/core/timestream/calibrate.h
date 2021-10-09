#pragma once

namespace timestream {
template <typename Derived>
auto calibrate(TCData<LaliDataKind::PTC,MatrixXd>&in, Eigen::DenseBase<Derived> &fluxscale) {
    //Eigen::Map<Eigen::Vector<double, Eigen::Dynamic,Eigen::RowMajor>> rowMajor_fluxscale(fluxscale.derived().data(), fluxscale.derived().size());
    for(Eigen::Index i=0;i<in.scans.data.cols();i++) {
        in.scans.data.col(i) = in.scans.data.col(i)*fluxscale(i);
    }
}
} // namespace timestream
